"""Authored-vs-pasted span contract for text-style identity (design: nmem-identity chat-style
modality, Phase 4).

Fingerprint the SPEAKER's own writing, not text they relayed. A chat turn is annotated as a list of
spans, each tagged by how it entered the message — typed/dictated by the speaker vs pasted or quoted
from elsewhere. Only authored spans feed the author embedding; pasted spans are excluded from BOTH the
style signature (it is someone else's style) and, downstream, the accountability chain (a pasted "the
CEO said X" is not the paster's commitment). The pasted fraction is also returned as a confidence
DISCOUNT.

The contract is transport-agnostic: a browser `paste` event fills it precisely; a raw-API caller may
send a single authored span, or omit spans entirely and let the heuristic fallback estimate
contamination (`perception.text_quality.heuristic_contamination`). Conservative by construction — a
span with an unknown/missing source is treated as PASTED, so uncertain text is never credited to the
speaker's signature. Pure (no ML deps), so it imports and unit-tests without torch present.
"""
from __future__ import annotations

from dataclasses import dataclass

# How a span entered the message. Authored = the speaker's own keystrokes / dictation (their style);
# everything else — pasted, quoted, relayed, or unknown — is NOT their style and is excluded from the
# embedding. Unknown is deliberately in the pasted set (never credit uncertain text to the speaker).
AUTHORED_SOURCES = frozenset({"typed", "authored", "dictated"})


@dataclass(frozen=True)
class Span:
    """One contiguous piece of a turn, tagged by how it entered the message."""
    text: str
    source: str

    @property
    def is_authored(self) -> bool:
        return self.source in AUTHORED_SOURCES


def parse_spans(raw) -> list[Span]:
    """Parse the HTTP-body span list ``[{text, source}, ...]`` → ``[Span]``. Non-list input, non-dict
    entries, and empty-text entries are dropped (fail-open to fewer spans, never raise on the request
    path). A missing/blank source becomes ``"unknown"`` → treated as pasted (conservative)."""
    if not isinstance(raw, list):
        return []
    out: list[Span] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text:
            continue
        source = str(item.get("source") or "unknown").strip().lower() or "unknown"
        out.append(Span(text=text, source=source))
    return out


def authored_documents(spans: list[Span]) -> list[str]:
    """The authored text as LUAR documents. Pasted / quoted / unknown spans are EXCLUDED — the core
    Phase-4 guarantee: relayed text never enters the author embedding.

    Adjacent authored spans are COALESCED into one document (concatenated verbatim, no inserted
    separators) so keystroke/edit fragments from an instrumented client don't each become a separate
    document — which would blow past the episode limit (spurious text_insufficient) and split the very
    writing being fingerprinted mid-word (codex). A pasted span BREAKS the run, starting a new
    authored document, so relayed material never bridges two authored stretches."""
    docs: list[str] = []
    run: list[str] = []
    for s in spans:
        if s.is_authored:
            run.append(s.text)
        elif run:
            docs.append("".join(run).strip())
            run = []
    if run:
        docs.append("".join(run).strip())
    return [d for d in docs if d]


def pasted_ratio(spans: list[Span]) -> float:
    """Fraction of the turn's characters that were NOT authored (0..1) — a confidence DISCOUNT, never
    a hard label. No spans / empty → 0.0 (nothing known to be pasted)."""
    total = sum(len(s.text) for s in spans)
    if total <= 0:
        return 0.0
    pasted = sum(len(s.text) for s in spans if not s.is_authored)
    return pasted / total
