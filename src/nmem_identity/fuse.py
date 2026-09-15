"""Multi-modal identity fusion — combine per-channel votes and contextual priors into one calibrated
identity decision, under the existing grounding levels.

The matcher (`cluster.match_or_create`) is single-channel: it answers "does this ONE voiceprint /
faceprint / author-vector match a cluster?" Fusion is the layer above it (design: chat-style modality,
Phase 5): given whatever signals a turn produced — a text_style author vote, maybe a concurrent voice
match, plus contextual priors (a self-declared name, session continuity, an overlapping voice session)
— decide **who** the interlocutor is and **how grounded** that belief is.

**The math is additive in log-odds.** Independent evidence combines by adding log-likelihood ratios:

    log_odds(candidate) = prior_log_odds(candidate)            # from context signals
                        + Σ_channel  calibrated_LLR(channel vote for candidate)

`calibrate.Calibration` turns each channel's raw cosine into that LLR on a common scale;
`posterior = sigmoid(log_odds)`. The best candidate is the arg-max; a margin guard (like the matcher's)
keeps an ambiguous near-tie from grounding.

**Text is a structurally weak vote — enforced here, not by tuning.** Two guardrails make "text can
only raise a *prior*, never authorise on its own" (design §6, non-negotiable) a property of the code:

  1. A *weak* channel's LLR is clamped to ``TEXT_LLR_CAP`` — even a perfectly-fit curve on long text
     cannot contribute unbounded evidence.
  2. Grounding to ``confident`` (and ``authorize_ok``) requires **non-weak corroboration**: a non-text
     channel vote that itself favours the candidate, or a strong context signal (an authenticated
     principal / a concurrent voice-session match). Text alone caps at ``speculative`` and never
     authorises, no matter how high its posterior climbs.

**Uncalibrated ⇒ abstain.** A channel with no calibration curve contributes **zero** LLR (it abstains)
rather than guessing — trust is never shipped ahead of the eval. With no signals at all, fusion returns
an empty (ungrounded, no-candidate) result. Everything here is a pure function of its inputs: no I/O,
no DB — the matcher/`/resolve` gathers votes and calibrations and calls `fuse`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .calibrate import Calibration

# Channels whose LLR is hard-clamped (the weak-vote guardrail). text_style is weak by design; voice /
# face are not. A weak channel can never, alone, push a candidate past the speculative grounding floor.
WEAK_CHANNELS = frozenset({"text_style"})
# Max |LLR| a weak channel may contribute to fusion, in nats (~e^1.5 ≈ 4.5:1 odds from one text vote).
# Deliberately small: authorship at chat length is a soft prior, not proof.
TEXT_LLR_CAP = 1.5

# Grounding thresholds in log-odds (posterior = sigmoid). speculative ≈ P>0.62, confident ≈ P>0.90.
SPECULATIVE_LOGODDS = 0.5
CONFIDENT_LOGODDS = 2.2
# The winner must beat the runner-up by this many nats, else the decision is an ambiguous tie and is
# held at (at most) speculative — mirrors the matcher's top1-vs-top2 margin guard.
MARGIN_LOGODDS = 1.0


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class ChannelVote:
    """One channel's similarity of the current observation to a candidate identity. ``candidate`` is a
    stable key the caller chose (e.g. ``"person:42"`` for an enrolled person, ``"cluster:17"`` for an
    unbound proto-cluster). ``cosine`` is the raw centroid similarity; ``length`` is the retained
    authored-char count for length-banded text calibration (None for voice/face)."""
    channel: str
    candidate: str
    cosine: float
    length: Optional[int] = None
    quality: Optional[float] = None


@dataclass(frozen=True)
class ContextSignal:
    """A non-embedding prior toward a candidate: a self-declared name, session continuity, an
    overlapping voice session, a learned prior relation. ``log_odds`` is its prior contribution (nats);
    ``strong`` marks a signal that counts as non-weak corroboration for grounding (an authenticated
    principal or a concurrent voice-session match — NOT a claimed name or bare continuity)."""
    kind: str
    candidate: str
    log_odds: float
    strong: bool = False


@dataclass(frozen=True)
class FusionConfig:
    """Fusion policy. Defaults are the module constants; a deployment overrides via the matcher's
    config so the guardrails and thresholds are tunable without touching this math."""
    text_llr_cap: float = TEXT_LLR_CAP
    speculative_logodds: float = SPECULATIVE_LOGODDS
    confident_logodds: float = CONFIDENT_LOGODDS
    margin_logodds: float = MARGIN_LOGODDS
    weak_channels: frozenset[str] = WEAK_CHANNELS


@dataclass(frozen=True)
class FusionResult:
    """The fused decision. ``candidate`` is the arg-max identity (None when there was no signal);
    ``grounding`` ∈ {ungrounded, speculative, confident}; ``authorize_ok`` is True only when a
    non-weak channel/signal corroborates a confident belief — the single bit a caller checks before
    letting identity gate anything consequential. ``per_channel_llr`` / ``candidates`` are for
    provenance and debugging."""
    candidate: Optional[str]
    posterior: float
    log_odds: float
    grounding: str
    authorize_ok: bool
    margin: float
    has_strong: bool
    per_channel_llr: dict[str, float] = field(default_factory=dict)
    candidates: dict[str, float] = field(default_factory=dict)


_EMPTY = FusionResult(candidate=None, posterior=0.0, log_odds=float("-inf"),
                      grounding="ungrounded", authorize_ok=False, margin=0.0, has_strong=False)


def fuse(votes: list[ChannelVote], signals: list[ContextSignal],
         calibrations: dict[str, Calibration], *,
         config: Optional[FusionConfig] = None) -> FusionResult:
    """Fuse channel votes + context signals into one identity decision. Pure; see module docstring.

    Accumulates log-odds per candidate (context priors + calibrated per-channel LLRs, weak channels
    clamped), picks the arg-max, and classifies its grounding under the margin + corroboration rules.
    A channel with no entry in ``calibrations`` abstains (contributes 0). Returns the empty ungrounded
    result when there is nothing to go on."""
    cfg = config or FusionConfig()
    if not votes and not signals:
        return _EMPTY

    log_odds: dict[str, float] = {}
    has_strong: dict[str, bool] = {}
    per_channel_llr: dict[str, float] = {}
    # LLR contributed by an UNFITTED (default/analytic) weak-channel curve — kept for the soft
    # (speculative) signal + posterior, but excluded from the confident/authorise threshold so a
    # seeded default can never be the corroboration that authorises (trust follows the eval).
    unfitted_weak: dict[str, float] = {}

    for sig in signals:
        log_odds[sig.candidate] = log_odds.get(sig.candidate, 0.0) + float(sig.log_odds)
        # `strong` only counts as corroboration when it actually FAVOURS the candidate (log_odds > 0),
        # mirroring the non-weak channel check below — a strong signal that opposes the identity
        # (e.g. a voice session that says "not them") must never enable authorisation (codex).
        if sig.strong and float(sig.log_odds) > 0.0:
            has_strong[sig.candidate] = True
        has_strong.setdefault(sig.candidate, False)

    for v in votes:
        cal = calibrations.get(v.channel)
        if cal is None:
            # Uncalibrated channel abstains rather than guessing (trust follows the eval).
            log_odds.setdefault(v.candidate, 0.0)
            has_strong.setdefault(v.candidate, False)
            continue
        llr = cal.llr(v.cosine, v.length)
        if v.channel in cfg.weak_channels:
            llr = max(-cfg.text_llr_cap, min(cfg.text_llr_cap, llr))
        log_odds[v.candidate] = log_odds.get(v.candidate, 0.0) + llr
        per_channel_llr[f"{v.candidate}:{v.channel}"] = llr
        # A non-weak channel FAVOURING this candidate is corroboration for grounding/authorisation.
        if v.channel not in cfg.weak_channels and llr > 0.0:
            has_strong[v.candidate] = True
        # A weak channel calibrated only by a DEFAULT curve is soft-only: track its contribution so it
        # is excluded from the authoritative total below (it may still lift the speculative signal).
        elif v.channel in cfg.weak_channels and not cal.fitted:
            unfitted_weak[v.candidate] = unfitted_weak.get(v.candidate, 0.0) + llr
        has_strong.setdefault(v.candidate, False)

    if not log_odds:
        return _EMPTY

    ranked = sorted(log_odds.items(), key=lambda kv: kv[1], reverse=True)
    top_c, top_lo = ranked[0]
    runner_lo = ranked[1][1] if len(ranked) > 1 else float("-inf")
    margin = top_lo - runner_lo if runner_lo != float("-inf") else float("inf")
    strong = has_strong.get(top_c, False)

    # Authoritative (fitted-only) scores drop each candidate's UNFITTED weak-channel evidence. The
    # confident/authorise decision runs ENTIRELY on these — threshold, winner, AND margin — so a
    # seeded default can neither reach the bar nor reverse/resolve the winner into authorisation; it
    # only ever lifts the soft speculative hint (top_lo, below).
    auth = {c: log_odds[c] - unfitted_weak.get(c, 0.0) for c in log_odds}
    auth_top_c = max(auth, key=auth.get)
    auth_top_lo = auth[auth_top_c]
    auth_runner = max((v for c, v in auth.items() if c != auth_top_c), default=float("-inf"))
    auth_margin = auth_top_lo - auth_runner if auth_runner != float("-inf") else float("inf")

    # Grounding ladder. Confident REQUIRES non-weak corroboration AND that the reported winner is also
    # the authoritative winner, crosses the threshold, and beats every competitor by the margin on
    # fitted-only scores — so neither a high text posterior alone nor a seeded default can ever reach
    # confident/authorize (the design's non-negotiable guardrail).
    if (top_c == auth_top_c and auth_top_lo >= cfg.confident_logodds
            and auth_margin >= cfg.margin_logodds and strong):
        grounding = "confident"
    elif top_lo >= cfg.speculative_logodds:
        grounding = "speculative"
    else:
        grounding = "ungrounded"
    authorize_ok = grounding == "confident" and strong

    return FusionResult(
        candidate=top_c, posterior=_sigmoid(top_lo), log_odds=top_lo,
        grounding=grounding, authorize_ok=authorize_ok,
        margin=(float(margin) if margin != float("inf") else float("inf")),
        has_strong=strong, per_channel_llr=per_channel_llr, candidates=dict(log_odds))
