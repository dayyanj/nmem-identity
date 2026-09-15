"""Per-observation text-style quality — the authorship analogue of perception.quality (face).

Quality has the same two jobs the voice/face channels already use it for:
  1. **Regime labelling** — a thin or relayed sample is a `text_degraded` / `text_insufficient`
     observation; the benchmark stratifies EER by regime (and by authored length especially, since
     short chat text is where authorship signal is weakest), so quality feeds that label.
  2. **Learning gate** — an author vector from too little (or pasted) text is unreliable and must NOT
     move a learned centroid or persist as a template. `cluster.match_or_create` gates centroid updates
     on `quality >= min_quality`; this module produces the scalar it gates on.

The score is a **product** of interpretable [0,1] credits, so a hard failure on any one axis drags
quality down (you cannot buy back "ok sounds good" with high lexical diversity):

  * **sufficiency** — authored character count vs a floor/ceiling ramp. The dominant axis: below the
    floor there is simply not enough text to fingerprint a style. This is what kills grounding on
    trivial acknowledgements.
  * **authored fraction** — 1 − pasted_ratio (Phase 4). A turn that is mostly relayed/quoted text is
    not the speaker's own style; until paste instrumentation exists the caller passes 0.0 and this is
    a no-op (credit 1.0), so the axis slots in without changing the shape here.
  * **diversity** — a gentle guard against degenerate input ("aaaaaa", "ok ok ok ok"): distinct-word
    ratio with a low floor, tuned to leave normal prose at ~full credit and only dock spam.

Pure stdlib + a tiny bit of math — no ML deps, so it imports and unit-tests without torch present.
"""
from __future__ import annotations

import re
from typing import Any

# Formatting artifacts a speaker typically RELAYS rather than authors in their own style — the basis
# of the heuristic contamination estimate used when a channel has NO paste instrumentation (design
# §5). Fenced code, URLs, and quoted (`>`) lines. This only ever DISCOUNTS confidence; it never hard-
# labels a span authored/pasted (that requires the real span contract, text_spans).
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_FENCE_MARK_RE = re.compile(r"```")            # individual fence marks, to catch an unclosed opener
_URL_RE = re.compile(r"https?://\S+")
# Leading indent is [ \t]* (NOT \s*): \s includes newlines, so \s* at a line start backtracks across
# runs of blank lines, making detection quadratic on newline-heavy input (codex). [ \t]* can't cross
# a line boundary, so each line-start match is bounded.
_QUOTE_LINE_RE = re.compile(r"^[ \t]*>.*$", re.MULTILINE)

# Tunable thresholds (documented defaults; a deployment re-thresholds by re-scoring or via config).
# Authorship signal is weak at chat length, so the sufficiency ramp is deliberately demanding: a
# single short turn is text_degraded (recognised maybe, but not centroid-updating); the gate is
# reached only once enough authored text has accumulated (~a solid multi-turn episode). This is the
# design's "short-text unreliability" guard, not an accident — recalibrate against the length-stratified
# EER (eval/text_benchmark.py), never by feel.
MIN_AUTHORED_CHARS = 120.0     # below this → text_insufficient (emit no observation / floor confidence)
GOOD_AUTHORED_CHARS = 600.0    # at/above this → full sufficiency credit
DIVERSITY_FLOOR = 0.20         # distinct-word ratio below this → degenerate (repetition/spam)
DIVERSITY_GOOD = 0.45          # at/above this → full diversity credit (normal prose clears this)

# Gate: an observation passes (may update the centroid / be stored) only if quality clears this.
TEXT_QUALITY_GATE = 0.50

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _lerp_credit(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 at/above hi, linear between — a soft floor/ceiling ramp (mirrors quality.py)."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return _clamp01((x - lo) / (hi - lo))


def diversity(text: str) -> tuple[float, dict[str, float]]:
    """Distinct-word ratio — a cheap degeneracy guard. Normal prose sits well above the floor; only
    highly repetitive input ("ok ok ok", "aaaa aaaa") scores low. Returns (ratio, parts)."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n = len(words)
    ratio = (len(set(words)) / n) if n else 0.0
    return ratio, {"n_words": float(n), "distinct_word_ratio": ratio}


def text_quality(text: str, *, pasted_ratio: float = 0.0) -> tuple[float, dict[str, float]]:
    """Combine sufficiency × authored-fraction × diversity into a [0,1] quality, plus the parts.

    `text` is the AUTHORED text to be embedded (pasted spans already removed by the caller, Phase 4);
    `pasted_ratio` (0..1) is how much of the ORIGINAL turn was relayed, used only to discount
    confidence — never to hard-label. Product combination so any single hard failure dominates."""
    authored = text or ""
    authored_chars = float(len(authored.strip()))
    sufficiency = _lerp_credit(authored_chars, MIN_AUTHORED_CHARS, GOOD_AUTHORED_CHARS)

    authored_fraction = _clamp01(1.0 - float(pasted_ratio))

    div_ratio, parts = diversity(authored)
    diversity_credit = _lerp_credit(div_ratio, DIVERSITY_FLOOR, DIVERSITY_GOOD)

    quality = float(sufficiency * authored_fraction * diversity_credit)
    parts.update({
        "authored_chars": authored_chars, "sufficiency": sufficiency,
        "pasted_ratio": float(pasted_ratio), "authored_fraction": authored_fraction,
        "diversity_credit": diversity_credit, "quality": quality,
    })
    return quality, parts


def regime(parts: dict[str, Any]) -> str:
    """Map quality parts to a benchmark regime label. Below the authored-char floor there is not
    enough text to trust (text_insufficient); otherwise split on the gate (authentic vs degraded)."""
    if float(parts.get("authored_chars", 0.0)) < MIN_AUTHORED_CHARS:
        return "text_insufficient"
    if float(parts.get("quality", 0.0)) >= TEXT_QUALITY_GATE:
        return "text_authentic"
    return "text_degraded"


def passes_gate(quality: float) -> bool:
    return quality >= TEXT_QUALITY_GATE


def sufficient_length(text: str) -> bool:
    """True if there is enough authored text to attempt an observation at all. The embed boundary
    uses this as the "sub-threshold turns produce no observation" gate (design Phase 4) — cheaper than
    embedding first, and a hard floor below which no author vector is emitted."""
    return len((text or "").strip()) >= MIN_AUTHORED_CHARS


def heuristic_contamination(text: str) -> tuple[float, dict[str, float]]:
    """Estimate the pasted/relayed fraction of `text` from formatting artifacts, for channels with NO
    paste instrumentation (raw API / nmem-exchange). Returns ``(ratio, parts)`` where ``ratio`` ∈
    [0,1] is a confidence DISCOUNT ONLY — never a hard authored/pasted label (design §5): fenced code,
    URLs, and quoted (`>`) lines are content a speaker typically relays rather than authors in their
    own voice. Overlapping matches are unioned over character positions so the ratio can't exceed 1."""
    if not text:
        return 0.0, {"flagged_chars": 0.0, "total_chars": 0.0, "contamination": 0.0}
    n = len(text)
    flagged = bytearray(n)                          # 1 per char covered by any artifact (unioned)
    for rx in (_CODE_FENCE_RE, _URL_RE, _QUOTE_LINE_RE):
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                flagged[i] = 1
    # An opening ``` whose closing fence was truncated away (a retained prefix of a long code block)
    # leaves an ODD number of fences — flag from the last one to the end so truncated code still
    # counts as relayed rather than earning authored credit (codex).
    fences = [m.start() for m in _FENCE_MARK_RE.finditer(text)]
    if len(fences) % 2 == 1:
        for i in range(fences[-1], n):
            flagged[i] = 1
    covered = float(sum(flagged))
    ratio = _clamp01(covered / n)
    return ratio, {"flagged_chars": covered, "total_chars": float(n), "contamination": ratio}


def heuristic_contamination_docs(documents: list[str]) -> tuple[float, dict[str, float]]:
    """Aggregate `heuristic_contamination` over INDEPENDENT documents (turns): each turn is scored on
    its own text, so an unclosed code fence (or any fence state) stays scoped to that turn and can't
    bleed into later, unrelated turns — otherwise a truncated code block in turn 1 would mark every
    following clean turn as pasted (codex). Continuity is retained WITHIN a turn (its own newlines).
    Flagged and total chars are summed across turns."""
    flagged = 0.0
    total = 0.0
    for d in documents:
        _, parts = heuristic_contamination(d)
        flagged += parts["flagged_chars"]
        total += parts["total_chars"]
    ratio = _clamp01(flagged / total) if total > 0 else 0.0
    return ratio, {"flagged_chars": flagged, "total_chars": total, "contamination": ratio}
