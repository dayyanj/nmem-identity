"""Calibration math on synthetic score distributions — the LUAR forward is deferred to the eval
benchmark, so here we feed known genuine/impostor cosine draws and check the fitter recovers a sane,
monotone, prior-free LLR curve that (de)serialises through the store's JSONB shape unchanged."""
import math

import numpy as np
import pytest

from nmem_identity import calibrate


def _draw(mean_gen, mean_imp, *, n_gen=400, n_imp=400, sd=0.08, seed=0):
    rng = np.random.default_rng(seed)
    gen = rng.normal(mean_gen, sd, n_gen).tolist()
    imp = rng.normal(mean_imp, sd, n_imp).tolist()
    return gen, imp


def test_fit_curve_is_monotone_and_separates():
    # Well-separated classes → higher cosine ⇒ higher LLR, and the LLR crosses zero between the means.
    gen, imp = _draw(0.55, 0.10)
    c = calibrate.fit_curve(gen, imp)
    assert c.a > 0                                      # higher similarity ⇒ more "same-person" evidence
    lo, hi = c.llr(0.10), c.llr(0.55)
    assert hi > lo
    assert c.llr(0.10) < 0 < c.llr(0.55)               # impostor region negative, genuine region positive
    # Monotone across a sweep.
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    llrs = [c.llr(x) for x in xs]
    assert all(b >= a for a, b in zip(llrs, llrs[1:]))


def test_llr_is_prior_free_of_fit_class_balance():
    # The SAME underlying distributions, but one fit set is class-imbalanced. Because prior_logodds is
    # subtracted, the recovered LLR at a fixed cosine should be close, not shifted by the imbalance.
    gen1, imp1 = _draw(0.5, 0.1, n_gen=400, n_imp=400, seed=1)
    gen2, imp2 = _draw(0.5, 0.1, n_gen=100, n_imp=800, seed=1)
    c1 = calibrate.fit_curve(gen1, imp1)
    c2 = calibrate.fit_curve(gen2, imp2)
    # At the genuine mean the two prior-free LLRs agree to within ~1 nat despite 8× vs 1× imbalance.
    assert abs(c1.llr(0.5) - c2.llr(0.5)) < 1.0
    # And prior_logodds actually captured the imbalance the class counts imply.
    assert c2.prior_logodds == pytest.approx(math.log(100 / 800), abs=1e-9)


def test_llr_clip_bounds_evidence():
    gen, imp = _draw(0.9, 0.0, sd=0.01)                 # near-separable → large raw slope
    c = calibrate.fit_curve(gen, imp, clip=2.0)
    assert -2.0 <= c.llr(2.0) <= 2.0                    # extreme cosine can't exceed the clip
    assert -2.0 <= c.llr(-2.0) <= 2.0


def test_non_finite_cosine_abstains():
    # A NaN / inf similarity is not evidence — llr must return 0.0, never clip to +clip and become
    # maximum supporting evidence (a zero-norm embedding can yield NaN pgvector cosine distance).
    gen, imp = _draw(0.5, 0.1)
    c = calibrate.fit_curve(gen, imp)
    assert c.llr(float("nan")) == 0.0
    assert c.llr(float("inf")) == 0.0
    assert c.llr(float("-inf")) == 0.0


def test_fit_curve_requires_both_classes():
    with pytest.raises(ValueError):
        calibrate.fit_curve([0.5, 0.6], [])
    with pytest.raises(ValueError):
        calibrate.fit_curve([], [0.1, 0.2])


def test_banded_fit_picks_band_by_length():
    # Short text is LESS separable than long text (authorship reliability collapses at chat length):
    # build scored rows where the short band overlaps and the long band separates cleanly.
    rng = np.random.default_rng(3)
    scored = []
    # long band (>280): clean separation
    for _ in range(120):
        scored.append((float(rng.normal(0.55, 0.05)), True, 600))
        scored.append((float(rng.normal(0.10, 0.05)), False, 600))
    # short band (<=40): heavy overlap
    for _ in range(120):
        scored.append((float(rng.normal(0.30, 0.12)), True, 30))
        scored.append((float(rng.normal(0.25, 0.12)), False, 30))
    cal = calibrate.fit_banded(scored, band_edges=[40, 120, 280], channel="text_style")
    assert "<=40" in cal.bands and ">280" in cal.bands
    # The long-band curve is steeper (more discriminative) than the short-band one.
    assert cal.bands[">280"].a > cal.bands["<=40"].a
    # A mid-cosine is credited MORE at long length than at short length.
    assert cal.llr(0.45, length=600) > cal.llr(0.45, length=30)
    # curve_for resolves the right band, and an unknown length falls back to pooled.
    assert cal.curve_for(30) is cal.bands["<=40"]
    assert cal.curve_for(None) is cal.pooled


def test_sparse_band_falls_back_to_pooled():
    scored = [(0.5, True, 600), (0.1, False, 600)]      # far below min_per_band
    cal = calibrate.fit_banded(scored, band_edges=[40, 280], channel="text_style", min_per_band=20)
    assert cal.bands == {}                              # nothing fit per-band
    assert cal.curve_for(600) is cal.pooled


def test_calibration_json_roundtrip():
    gen, imp = _draw(0.5, 0.1)
    cal = calibrate.fit_banded([(s, True, 600) for s in gen] + [(s, False, 600) for s in imp],
                               band_edges=[40, 280], channel="text_style", min_per_band=10)
    blob = cal.as_curve_json()
    back = calibrate.Calibration.from_curve_json("text_style", blob)
    assert back.channel == "text_style"
    assert back.band_edges == (40, 280)
    for x in (0.0, 0.2, 0.5):
        assert back.llr(x, length=600) == pytest.approx(cal.llr(x, length=600), abs=1e-9)
        assert back.llr(x, length=None) == pytest.approx(cal.llr(x, length=None), abs=1e-9)
