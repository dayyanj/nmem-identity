"""Text-style embedding sidecar — authored chat text → 512-d LUAR author vector, symmetric with the
voice / face `/embed` contract.

A host POSTs a person's authored text (a single blob, or a list of recent turns as an episode) here;
the service embeds it with LUAR, scores authorship quality, and returns the vector the matcher wants.
Keeping the model in a warm sidecar (rather than in the host process) matches the other channels: the
host stays thin and just makes an HTTP call.

    POST /embed
        {"spans": [{"text": "...", "source": "typed"|"pasted"}, ...]}   # authored-vs-pasted (Phase 4)
        {"documents": ["turn1", "turn2", ...]}                          # native LUAR episode
        {"text": "..."}                                                 # one-shot (split server-side)
        (+ optional pasted_ratio to override the measured/heuristic value)
        -> { embedding:[512 floats], dim, model_id, embed_version, n_documents, authored_chars,
             quality, quality_parts, regime, pasted_ratio, source }
        -> { regime:"text_insufficient", ... }   # NO 'embedding' — a sub-threshold or fully-pasted
                                                  # turn produces NO observation (Phase 4 gate)
    GET  /health

The 512-d embedding is L2-normalised (cosine == dot). PASTE-AWARENESS (design §4-5): when `spans` are
given, ONLY authored spans are embedded — pasted/quoted/relayed text provably never enters the author
vector — and the pasted fraction discounts quality. When no spans are given (raw-API / uninstrumented
channel), a heuristic estimates contamination (fenced code / URLs / quoted lines) and discounts only,
never hard-labelling. Below the authored-length floor, no observation is emitted at all.

PRIVACY: this sidecar does NOT persist raw text — the authored text is quasi-biometric. If a
deployment stores raw text for re-embedding on a model swap, it must do so under a `raw_ref` passed to
the matcher so `delete_person` captures it in its delete cascade (design §5).

Run with the `text` + `serve` extras (torch, transformers, fastapi, uvicorn). Configure the port with
NMEM_IDENTITY_TEXT_EMBED_PORT and the checkpoint with NMEM_IDENTITY_TEXT_MODEL_NAME:

    NMEM_IDENTITY_TEXT_EMBED_PORT=9406 python services/text_style_embed_server.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmem_identity import config, models                       # noqa: E402
from nmem_identity.perception import text_quality as tq        # noqa: E402
from nmem_identity.perception import text_spans as ts_spans    # noqa: E402
from nmem_identity.perception import text_style as text_mod    # noqa: E402

PORT = int(config.get("text_embed_port"))
# Upper bound on characters scanned by the heuristic contamination pass (per turn) — it is a rough,
# discount-only estimate, so bounding the scan keeps /embed responsive on a pathologically large body.
_MAX_CONTAM_CHARS = 8000
_log = logging.getLogger("text_embed")

app = FastAPI()
_embedder: text_mod.TextStyleEmbedder | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _embedder
    # Load once, warm. A failure here (missing extra / weights) should surface immediately, not per
    # request — construct eagerly and let /health report readiness.
    name = str(config.get("text_model_name"))
    _embedder = text_mod.TextStyleEmbedder(
        model_name=name,
        episode_len=int(config.get("text_episode_len")),
        doc_max_tokens=int(config.get("text_doc_max_tokens")))
    # Fail fast on a checkpoint↔store desync: vectors this sidecar produces are stored under the
    # text_style model_id, so the running checkpoint MUST match the store's active model_id (else KNN
    # silently never matches, or mixes generations). A checkpoint swap is a migration, not a typo.
    if _embedder.model_id != models.model_id("text_style"):
        raise RuntimeError(
            f"text checkpoint {name!r} → model_id {_embedder.model_id!r} but the store's active "
            f"text_style model_id is {models.model_id('text_style')!r}. Swapping the live checkpoint "
            f"is a migration: update the store model_id + partial index and re-embed, don't just "
            f"change NMEM_IDENTITY_TEXT_MODEL_NAME.")
    _log.info("text embedder ready (checkpoint=%s model_id=%s device=%s)",
              name, _embedder.model_id, _embedder.device)


@app.get("/health")
async def health() -> dict:
    return {"ok": _embedder is not None,
            "model_id": (_embedder.model_id if _embedder else models.model_id("text_style")),
            "checkpoint": (_embedder.model_name if _embedder else config.get("text_model_name")),
            "dim": text_mod.EMBED_DIM,
            "device": (_embedder.device if _embedder else None)}


def _insufficient(authored_chars: int, n_documents: int, pasted_ratio: float, source: str) -> dict:
    """A 'no observation' result (HTTP 200, NO 'embedding' key) — a sub-threshold or fully-pasted turn
    emits no author vector, so the caller never submits unreliable/relayed text to /match."""
    return {"regime": "text_insufficient", "authored_chars": authored_chars,
            "n_documents": n_documents, "pasted_ratio": pasted_ratio, "source": source,
            "model_id": _embedder.model_id, "dim": text_mod.EMBED_DIM}


@app.post("/embed")
async def embed(req: dict) -> dict:
    if _embedder is None:                                # pragma: no cover - startup failed
        raise HTTPException(503, "embedder not ready")
    req = req or {}
    spans_raw = req.get("spans")
    if spans_raw is not None:
        # Explicit authored/pasted spans: embed ONLY the authored ones; pasted text never enters the
        # vector. pasted_ratio is MEASURED from the spans (Phase 4 core guarantee) → no heuristic.
        spans = ts_spans.parse_spans(spans_raw)
        # Coalesced authored runs, each sentence/newline-split into episode documents (like the `text`
        # path) so a long fully-typed message isn't crushed into one 32-token document (codex).
        documents = [d for run in ts_spans.authored_documents(spans)
                     for d in text_mod.split_documents(run)]
        pasted_ratio: float = ts_spans.pasted_ratio(spans)
        source = "spans"
    else:
        documents_in = req.get("documents")
        text = req.get("text")
        if documents_in is None and text is None:
            raise HTTPException(400, "provide 'spans', 'documents' (list), or 'text' (string)")
        # Contamination is estimated on the ORIGINAL input, where line / fence structure is intact —
        # decoupled from the sentence-chunked embedding documents (design §5; codex). A discount only.
        # `text` is ONE turn (a spanning fence is caught); a `documents` episode is independent turns,
        # scored per turn (episode-limited) so one turn's truncated fence can't consume the next.
        window = int(getattr(_embedder, "episode_len", 16))
        if documents_in is None:
            documents = text_mod.split_documents(str(text))
            pasted_ratio = tq.heuristic_contamination(str(text)[:_MAX_CONTAM_CHARS])[0]
        elif not isinstance(documents_in, list):
            raise HTTPException(400, "'documents' must be a list of strings")
        else:
            documents = [str(d) for d in documents_in if str(d).strip()]
            turns = [d[:_MAX_CONTAM_CHARS] for d in documents[:window]]
            pasted_ratio = tq.heuristic_contamination_docs(turns)[0]
        source = "heuristic"
    # An explicit measured pasted_ratio (a caller that computed it another way) overrides the heuristic.
    if req.get("pasted_ratio") is not None:
        try:
            pasted_ratio = max(0.0, min(1.0, float(req["pasted_ratio"])))
        except (TypeError, ValueError):
            pass
    pasted_ratio = float(pasted_ratio)

    documents = [d for d in documents if d.strip()]
    joined = " ".join(documents)
    # Sufficiency gate (Phase 4): below the authored-length floor — or with no authored text at all
    # (a fully-pasted turn) — emit NO observation. Cheap pre-check before the model call.
    if not tq.sufficient_length(joined):
        return _insufficient(len(joined.strip()), len(documents), pasted_ratio, source)

    # Blocking model call off the event loop, mirroring the voice/face sidecars.
    try:
        o = await asyncio.to_thread(
            _embedder.observe, documents, pasted_ratio=pasted_ratio, score_quality=True)
    except ValueError as e:                              # empty episode etc. → client error
        raise HTTPException(400, str(e)) from e
    # Retained (post-truncation) content may still fall below the floor → no observation.
    used_ratio = float(o.quality_parts.get("pasted_ratio", pasted_ratio))
    if tq.regime(o.quality_parts) == "text_insufficient":
        return _insufficient(o.authored_chars, o.n_documents, used_ratio, source)
    return {
        "embedding": o.embedding.tolist(),
        "dim": text_mod.EMBED_DIM,
        "model_id": _embedder.model_id,
        "embed_version": o.embed_version,
        "n_documents": o.n_documents,
        "authored_chars": o.authored_chars,
        "quality": o.quality,
        "quality_parts": o.quality_parts,
        "regime": tq.regime(o.quality_parts),
        "pasted_ratio": used_ratio,
        "source": source,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
