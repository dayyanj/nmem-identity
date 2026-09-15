"""Text-style perception — a person's chat text → a 512-d LUAR authorship vector.

The authorship analogue of perception.face: it turns what someone *writes* into a fixed-dim style
vector the store can cluster, so an agent can recognise a returning interlocutor by writing style
alone (design: nmem-identity-chat-style-modality). This is the WEAKEST identity modality by design —
a soft prior fused below voice/face (Phase 5), never an authorization on its own.

**LUAR is episodic.** A LUAR-family author encoder does not embed one blob of text; it embeds an
*episode* — a set of short documents (here, a person's recent chat turns) — and pools them into one
author vector. So `embed` takes a list of documents (turns); `embed_text` splits a single blob into
pseudo-documents for the one-shot case. Each document is truncated to LUAR's native short window
(`text_doc_max_tokens`, 32) and the episode to `text_episode_len` documents.

**Authored text only.** Pasted / relayed spans are the caller's to strip before calling here (Phase
4): fingerprinting text the speaker merely relayed would attribute someone else's style — and someone
else's accountability — to them. This module embeds whatever authored text it is given.

Heavy ML deps (torch, transformers) are **lazy-imported inside the class**, so
`import nmem_identity.perception.text_style` succeeds in a store-only install (numpy is core); the deps
are required only when you actually construct a `TextStyleEmbedder`.

Model provenance: this produces vectors for the store's `text_style` embedder — dimension comes from
`nmem_identity.models` (512), and `model_id` self-labels from the checkpoint so a non-default
checkpoint cannot silently mix generations against the store's `luar_v1` partial index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .. import models

TEXT_MODEL_ID: str = models.model_id("text_style")     # "luar_v1" — the schema-bound store literal
EMBED_DIM: int = models.dim("text_style")               # 512

# Known LUAR checkpoints → the store model_id they are stored under. Only the checkpoint whose id is
# the store's active `luar_v1` matches the live partial HNSW index; another checkpoint self-labels a
# DIFFERENT id, so the embed sidecar's desync guard rejects it (a checkpoint swap is a migration:
# add a new model_id + index and re-embed, not an env flip). Mirrors face's pack→model_id mapping.
LUAR_MODEL_ID: dict[str, str] = {
    "rrivera1849/LUAR-MUD": "luar_v1",
    "rrivera1849/LUAR-CRUD": "luar_crud_v1",
}
DEFAULT_TEXT_MODEL = "rrivera1849/LUAR-MUD"

# Split a single blob into pseudo-documents on sentence / newline boundaries for the one-shot path, so
# a multi-sentence paragraph becomes several short LUAR documents (good episode coverage — one 32-token
# document would discard the rest of a paragraph). Formatting-based paste detection does NOT rely on
# this split: the sidecar scores contamination on the ORIGINAL input, where line/fence structure is
# intact (see text_style_embed_server + text_quality.heuristic_contamination). A blob with no boundary
# is a single document.
import re as _re
_DOC_SPLIT = _re.compile(r"[\r\n]+|(?<=[.!?])\s+")


@dataclass
class TextObservation:
    """One authored-text observation → an author vector. `embedding` is L2-normalised (cosine == dot),
    matching how the store compares centroids. `quality` is filled by perception.text_quality (0..1);
    None if quality scoring was skipped."""
    embedding: np.ndarray                   # (512,) float32, L2-normalised
    n_documents: int                        # how many docs (turns) went into the episode
    authored_chars: int                     # total authored characters embedded
    embed_version: int = 1
    quality: Optional[float] = None
    quality_parts: dict[str, float] = field(default_factory=dict)

    def as_template(self) -> dict[str, Any]:
        """Shape for store insertion: the embedder→vector map + provenance the matcher/ingest want."""
        return {
            "embeddings": {"text_style": self.embedding.tolist()},
            "modality": "text_style",
            "quality": self.quality,
            "embed_versions": {"text_style": self.embed_version},
        }


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n == 0.0 else (v / n).astype(np.float32)


def split_documents(text: str) -> list[str]:
    """Split a single text blob into pseudo-documents for the one-shot embed path, on sentence /
    newline boundaries (see `_DOC_SPLIT`); empties dropped. A blob with no boundary is one document."""
    parts = [p.strip() for p in _DOC_SPLIT.split(text or "") if p and p.strip()]
    return parts or ([text.strip()] if (text or "").strip() else [])


class TextStyleEmbedder:
    """A LUAR-family author encoder behind a fixed contract: authored text → 512-d style vector.

    Args:
        model_name: HuggingFace LUAR checkpoint (default MUD). Self-labels its store model_id via
            LUAR_MODEL_ID; an unknown checkpoint gets a derived, safe label so nothing silently
            mislabels against the store's `luar_v1` index.
        device: torch device string ('cuda' / 'cpu'); auto-selects cuda when available if None.
        episode_len / doc_max_tokens: episode shape (docs per episode, tokens per doc).
    """

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, *,
                 device: Optional[str] = None,
                 episode_len: int = 16, doc_max_tokens: int = 32,
                 embed_version: int = 1) -> None:
        # Lazy, isolated imports so the package imports without ML deps present.
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover - only hit without the `text` extra installed
            raise RuntimeError(
                "TextStyleEmbedder needs the 'text' extra (torch, transformers). "
                "Install with: pip install 'nmem-identity[text]'"
            ) from e

        self._torch = torch
        self.model_name = model_name
        self.episode_len = int(episode_len)
        self.doc_max_tokens = int(doc_max_tokens)
        self.embed_version = int(embed_version)
        # This checkpoint's store model_id — what a vector from this embedder must be stored under.
        self.model_id = LUAR_MODEL_ID.get(model_name, "luar_" + _safe_suffix(model_name))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # LUAR ships custom modeling code → trust_remote_code. Loaded once, eager, so a missing
        # checkpoint / bad name fails at construction (surfaced by /health), not per request.
        self._tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self._model.eval().to(self.device)

    def embed(self, documents: list[str]) -> np.ndarray:
        """Embed an episode (a person's recent authored documents/turns) → (512,) L2-normalised.
        Truncates the episode to `episode_len` documents and each document to `doc_max_tokens`."""
        torch = self._torch
        docs = [d.strip() for d in (documents or []) if d and d.strip()][: self.episode_len]
        if not docs:
            raise ValueError("embed needs at least one non-empty authored document")
        enc = self._tok(docs, padding="max_length", truncation=True,
                        max_length=self.doc_max_tokens, return_tensors="pt")
        # LUAR consumes (batch, episode_len, tokens): one author (batch=1), E documents.
        e = len(docs)
        input_ids = enc["input_ids"].reshape(1, e, -1).to(self.device)
        attention_mask = enc["attention_mask"].reshape(1, e, -1).to(self.device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        vec = np.asarray(_to_tensor(out).reshape(-1).float().cpu().numpy(), dtype=np.float32)
        if vec.shape[0] != EMBED_DIM:  # pragma: no cover - a checkpoint dim mismatch is a config error
            raise ValueError(f"expected {EMBED_DIM}-d LUAR embedding, got {vec.shape[0]}")
        return _l2norm(vec)

    def embed_text(self, text: str) -> np.ndarray:
        """One-shot convenience: split a single blob into pseudo-documents, then `embed`."""
        return self.embed(split_documents(text))

    def _retained_documents(self, documents: list[str]) -> list[str]:
        """The per-document text the embedding ACTUALLY saw after truncation to `doc_max_tokens`
        (decoded from the same truncated token ids `embed` uses, special tokens stripped). Returned as
        a LIST so callers join with spaces (sufficiency/diversity) or newlines (line-anchored
        contamination detection). Quality must be scored on THIS, not the raw documents: a repetitive
        32-token prefix followed by long diverse prose (which truncation discards) must not pass the
        learning gate on content the model never embedded (codex)."""
        enc = self._tok(documents, truncation=True, max_length=self.doc_max_tokens)
        return [self._tok.decode(ids, skip_special_tokens=True).strip()
                for ids in enc["input_ids"]]

    def observe_spans(self, spans, *, score_quality: bool = True) -> TextObservation:
        """Observe from an authored/pasted span list (design Phase 4): embed ONLY the authored spans
        and discount quality by the MEASURED pasted fraction. `spans` is a list of `text_spans.Span`
        (or the parsed HTTP body). Each coalesced authored RUN is sentence/newline-split into episode
        documents (like the `text` path) so a long fully-typed message isn't crushed into one 32-token
        document (codex). Raises ValueError if no authored text remains (a fully-pasted turn produces no
        observation — the caller/sidecar treats that as text_insufficient)."""
        from .text_spans import authored_documents, pasted_ratio
        docs = [d for run in authored_documents(spans) for d in split_documents(run)]
        return self.observe(docs, pasted_ratio=pasted_ratio(spans), score_quality=score_quality)

    def observe(self, documents: list[str], *, pasted_ratio: float = 0.0,
                score_quality: bool = True) -> TextObservation:
        """Full pipeline: embed the authored episode (→ quality). `pasted_ratio` is the pasted fraction
        that discounts quality — MEASURED from the span contract, or a heuristic estimate the CALLER
        computed on the ORIGINAL input, where line/fence structure survives (see the embed sidecar +
        text_quality.heuristic_contamination). Two distinct axes, two distinct inputs: SUFFICIENCY /
        diversity are scored on the RETAINED (post-truncation) text so discarded content cannot
        authorize a centroid update (codex), while CONTAMINATION reflects how relayed the submitted turn
        looks (the caller's `pasted_ratio`) — the precise per-embedded-content path is the span
        contract. The caller must have already removed pasted spans (see observe_spans)."""
        docs = [d.strip() for d in (documents or []) if d and d.strip()][: self.episode_len]
        emb = self.embed(docs)
        retained_docs = self._retained_documents(docs)   # exactly what the embedding consumed, per doc
        joined = " ".join(retained_docs)
        o = TextObservation(embedding=emb, n_documents=len(docs), authored_chars=len(joined),
                            embed_version=self.embed_version)
        if score_quality:
            from .text_quality import text_quality
            o.quality, o.quality_parts = text_quality(joined, pasted_ratio=float(pasted_ratio or 0.0))
        return o


def _to_tensor(out: Any):
    """LUAR's forward returns the embedding tensor; some HF wrappers box it (tuple / obj with
    .pooler_output / .last_hidden_state). Unwrap to the author-vector tensor."""
    # A plain tensor has .reshape and .float; use it directly.
    if hasattr(out, "reshape") and hasattr(out, "float"):
        return out
    for attr in ("pooler_output", "last_hidden_state"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"unexpected LUAR output type {type(out)!r}")


def _safe_suffix(model_name: str) -> str:
    """A `[a-z0-9_]+` suffix derived from a checkpoint name, for self-labelling an unknown checkpoint's
    store model_id (must satisfy the store's _SAFE_MODEL_ID guard)."""
    tail = model_name.rsplit("/", 1)[-1].lower()
    return _re.sub(r"[^a-z0-9]+", "_", tail).strip("_") or "unknown"
