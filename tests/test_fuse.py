"""Fusion policy — the guardrails that make text a structurally weak vote, plus the additive log-odds
combination and grounding ladder. Uses hand-built Calibration curves so the math is exercised without
any model or DB."""
import math

import pytest

from nmem_identity import fuse
from nmem_identity.calibrate import Calibration, Curve


def _cal(channel, a=10.0, b=-3.0):
    """A steep pooled-only curve: cosine near 0.5 ⇒ strongly positive LLR, near 0 ⇒ negative."""
    return Calibration(channel=channel, pooled=Curve(a=a, b=b, prior_logodds=0.0,
                                                     n_genuine=500, n_impostor=500, clip=6.0))


def test_nan_cosine_vote_cannot_authorize():
    # A NaN biometric similarity must abstain (0 LLR via the calibration guard), never become evidence.
    # Without the guard a single NaN voice vote would clip to +6 nats and authorize on one candidate.
    cals = {"campp": _cal("campp", a=12.0, b=-3.0)}
    votes = [fuse.ChannelVote(channel="campp", candidate="person:1", cosine=float("nan"))]
    r = fuse.fuse(votes, [], cals)
    assert r.candidate == "person:1"
    assert r.log_odds == pytest.approx(0.0)              # abstained
    assert r.grounding == "ungrounded"
    assert not r.authorize_ok


def test_no_signal_is_empty_ungrounded():
    r = fuse.fuse([], [], {})
    assert r.candidate is None
    assert r.grounding == "ungrounded"
    assert not r.authorize_ok


def test_uncalibrated_channel_abstains():
    # A vote for a channel with no curve contributes zero — the candidate exists but at log-odds 0.
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:1", cosine=0.9, length=1000)]
    r = fuse.fuse(votes, [], {})                        # no calibrations supplied
    assert r.candidate == "person:1"
    assert r.log_odds == pytest.approx(0.0)
    assert r.grounding == "ungrounded"
    assert r.per_channel_llr == {}


def test_text_alone_never_confident_or_authorized():
    # A maximally confident text vote (high cosine, long text) still cannot cross to confident/authorize
    # because text is a WEAK channel with no non-weak corroboration — the non-negotiable guardrail.
    cals = {"text_style": _cal("text_style", a=30.0, b=-3.0)}
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:1", cosine=0.95, length=5000)]
    r = fuse.fuse(votes, [], cals)
    assert r.candidate == "person:1"
    assert r.log_odds <= fuse.TEXT_LLR_CAP + 1e-9        # clamped to the weak-vote cap
    assert r.grounding in ("speculative", "ungrounded")
    assert r.grounding != "confident"
    assert not r.authorize_ok
    assert not r.has_strong


def _default_cal(channel):
    """An UNFITTED default/analytic curve (zero sample provenance) — e.g. the one the appliance seeds
    so text_style isn't inert. Same shape as `_cal` but n_genuine=n_impostor=0."""
    return Calibration(channel=channel,
                       pooled=Curve(a=5.0, b=-1.5, prior_logodds=0.0,
                                    n_genuine=0, n_impostor=0, clip=1.5))


def test_unfitted_default_text_curve_cannot_corroborate_to_authorize():
    # An unfitted DEFAULT curve may lift the soft signal but must NEVER be the corroboration that
    # crosses confident/authorize alongside one borderline-strong signal. A FITTED curve in the exact
    # same scenario is allowed to (trust follows the eval). Regression for the codex finding.
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:1", cosine=0.6)]
    signals = [fuse.ContextSignal(kind="voice_session", candidate="person:1",
                                  log_odds=0.8, strong=True)]     # borderline; needs corroboration

    r_default = fuse.fuse(votes, signals, {"text_style": _default_cal("text_style")})
    assert r_default.grounding == "speculative"      # soft hint retained…
    assert not r_default.authorize_ok                # …but the default cannot authorize

    r_fitted = fuse.fuse(votes, signals, {"text_style": _cal("text_style", a=5.0, b=-1.5)})
    assert r_fitted.grounding == "confident"         # a real fit is allowed to corroborate
    assert r_fitted.authorize_ok


def test_unfitted_default_cannot_reverse_the_winner_into_authorization():
    # Two close fitted-strong candidates that correctly abstain (margin too small). A default text vote
    # for the loser must not reverse the winner AND resolve the ambiguity into authorization — the
    # authoritative winner/margin ignore unfitted evidence. Regression for the codex multi-candidate case.
    signals = [fuse.ContextSignal(kind="voice_session", candidate="person:1", log_odds=2.3, strong=True),
               fuse.ContextSignal(kind="voice_session", candidate="person:2", log_odds=2.4, strong=True)]
    assert not fuse.fuse([], signals, {}).authorize_ok               # baseline: too close → abstain

    votes = [fuse.ChannelVote(channel="text_style", candidate="person:1", cosine=0.6)]
    r = fuse.fuse(votes, signals, {"text_style": _default_cal("text_style")})
    assert not r.authorize_ok                                        # default can't authorize the flip
    assert r.grounding != "confident"


def test_text_llr_is_clamped_to_cap():
    cals = {"text_style": _cal("text_style", a=50.0, b=0.0)}   # would fit a huge LLR
    votes = [fuse.ChannelVote(channel="text_style", candidate="c:1", cosine=0.8, length=2000)]
    r = fuse.fuse(votes, [], cals)
    llr = r.per_channel_llr["c:1:text_style"]
    assert llr == pytest.approx(fuse.TEXT_LLR_CAP)       # positive side clamped exactly to the cap


def test_voice_can_ground_confident():
    # A strong voice vote (non-weak channel) with a clear margin grounds to confident + authorizes.
    cals = {"campp": _cal("campp", a=12.0, b=-3.0)}
    votes = [fuse.ChannelVote(channel="campp", candidate="person:7", cosine=0.62)]
    r = fuse.fuse(votes, [], cals)
    assert r.candidate == "person:7"
    assert r.log_odds >= fuse.CONFIDENT_LOGODDS
    assert r.has_strong
    assert r.grounding == "confident"
    assert r.authorize_ok


def test_text_plus_voice_corroboration_authorizes():
    # Text alone is capped; add a corroborating voice vote for the SAME candidate and it can authorize.
    cals = {"text_style": _cal("text_style", a=20.0, b=-2.0), "campp": _cal("campp", a=12.0, b=-3.0)}
    votes = [
        fuse.ChannelVote(channel="text_style", candidate="person:9", cosine=0.7, length=1500),
        fuse.ChannelVote(channel="campp", candidate="person:9", cosine=0.6),
    ]
    r = fuse.fuse(votes, [], cals)
    assert r.candidate == "person:9"
    assert r.has_strong                                 # voice corroborated
    assert r.grounding == "confident"
    assert r.authorize_ok
    # And text did contribute (capped) evidence on top of voice.
    assert r.per_channel_llr["person:9:text_style"] == pytest.approx(fuse.TEXT_LLR_CAP)


def test_strong_context_signal_counts_as_corroboration():
    # An authenticated-principal / voice-session context signal is non-weak corroboration on its own.
    cals = {"text_style": _cal("text_style", a=20.0, b=-2.0)}
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:3", cosine=0.7, length=1500)]
    signals = [fuse.ContextSignal(kind="voice_session", candidate="person:3", log_odds=2.0, strong=True)]
    r = fuse.fuse(votes, signals, cals)
    assert r.candidate == "person:3"
    assert r.has_strong
    assert r.grounding == "confident"
    assert r.authorize_ok


def test_weak_context_signal_is_not_corroboration():
    # A claimed name / bare session continuity raises the prior but is NOT strong: text+weak-context
    # can reach speculative but never confident/authorize.
    cals = {"text_style": _cal("text_style", a=20.0, b=-2.0)}
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:5", cosine=0.7, length=1500)]
    signals = [fuse.ContextSignal(kind="declared_name", candidate="person:5", log_odds=1.5, strong=False)]
    r = fuse.fuse(votes, signals, cals)
    assert r.candidate == "person:5"
    assert not r.has_strong
    assert r.grounding == "speculative"
    assert not r.authorize_ok


def test_strong_but_opposing_signal_does_not_corroborate():
    # A strong context signal that OPPOSES the candidate (negative log-odds) must not enable
    # authorization — corroboration requires the strong evidence to actually favour the identity.
    cals = {"text_style": _cal("text_style", a=20.0, b=-2.0)}
    votes = [fuse.ChannelVote(channel="text_style", candidate="person:1", cosine=0.7, length=1500)]
    signals = [
        fuse.ContextSignal(kind="declared_name", candidate="person:1", log_odds=2.0, strong=False),
        fuse.ContextSignal(kind="voice_session", candidate="person:1", log_odds=-1.0, strong=True),
    ]
    r = fuse.fuse(votes, signals, cals)
    assert r.candidate == "person:1"
    assert not r.has_strong                              # the only strong signal opposes → no corrob.
    assert r.grounding != "confident"
    assert not r.authorize_ok


def test_margin_guard_holds_ambiguous_tie():
    # Two candidates near-tied on strong voice evidence → margin below MARGIN_LOGODDS holds it out of
    # confident even though the top log-odds clears the confident threshold.
    cals = {"campp": _cal("campp", a=12.0, b=-3.0)}
    votes = [
        fuse.ChannelVote(channel="campp", candidate="person:1", cosine=0.62),
        fuse.ChannelVote(channel="campp", candidate="person:2", cosine=0.61),
    ]
    r = fuse.fuse(votes, [], cals)
    assert r.candidate == "person:1"
    assert r.log_odds >= fuse.CONFIDENT_LOGODDS
    assert r.margin < fuse.MARGIN_LOGODDS
    assert r.grounding == "speculative"                 # tie ⇒ not confident
    assert not r.authorize_ok


def test_additive_log_odds_and_posterior():
    cals = {"campp": _cal("campp", a=0.0, b=0.0)}       # llr == 0 for any cosine (flat curve)
    votes = [fuse.ChannelVote(channel="campp", candidate="p", cosine=0.5)]
    signals = [fuse.ContextSignal(kind="x", candidate="p", log_odds=0.8, strong=False)]
    r = fuse.fuse(votes, signals, cals)
    assert r.log_odds == pytest.approx(0.8)
    assert r.posterior == pytest.approx(1.0 / (1.0 + math.exp(-0.8)), abs=1e-9)


def test_config_override_changes_cap():
    cfg = fuse.FusionConfig(text_llr_cap=0.3)
    cals = {"text_style": _cal("text_style", a=20.0, b=-2.0)}
    votes = [fuse.ChannelVote(channel="text_style", candidate="c", cosine=0.7, length=1500)]
    r = fuse.fuse(votes, [], cals, config=cfg)
    assert r.per_channel_llr["c:text_style"] == pytest.approx(0.3)
