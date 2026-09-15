"""Per-channel score calibration — turn a raw cosine similarity into a prior-free log-likelihood
ratio (LLR) that fusion can add up across modalities.

Every channel lives on its own cosine scale (CAM++ same-speaker ~0.6, ArcFace ~0.3, LUAR chat-length
much lower and length-dependent), and a bare cosine is not comparable across channels or interpretable
as evidence. Calibration fixes both: it maps `cosine -> LLR`, where

    LLR(s) = log[ p(s | same person) / p(s | different person) ]

is the *weight of evidence* the observation carries, on a common (nats) scale, independent of how
common the person is. Fusion (`fuse.py`) then does the principled thing — sum the LLRs and add a
context prior — because independent evidence adds in log-odds space.

**Why this exists as its own layer (design: chat-style modality, Phase 5).** The store persists these
curves in `identity_calibration` keyed by `(model_id, embed_version, channel)` — a preprocessing or
weights change re-fits rather than silently reusing a stale curve. This module is the pure math:
fit a curve from a labelled score set, evaluate it, and (de)serialise it to the JSONB the store holds.
The DB round-trip is `store/calibration.py`; the operator command that runs an eval and persists is
`eval/fit_calibration.py`.

**Fit method — logistic (Platt) calibration.** A one-feature logistic regression on the cosine, with
genuine=1 / impostor=0, gives `logit P(same | s) = a·s + b`. By Bayes that posterior log-odds is the
LLR plus the *training* prior log-odds, so we subtract the training class balance to recover a
prior-free LLR:

    LLR(s) = a·s + b − log(n_genuine / n_impostor)

Fusion supplies its own context prior, so the fitter's class balance must not leak in. Simple, smooth,
monotone in the cosine, and only two parameters — robust on the small labelled sets a household eval
produces (a richer monotone/binned fit can slot in behind the same `Calibration.llr` contract later).

**Length banding (text_style).** LUAR authorship reliability collapses as the authored text shrinks,
so a single curve would over-credit short turns. `fit_banded` fits one curve per authored-length band
(the bands the length-stratified EER benchmark already reports) plus a pooled fallback; `Calibration`
picks the band by the observation's *retained* authored length at eval time. Voice/face pass no length
and use the single pooled curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# The absolute cap applied to any single channel's LLR after fitting. A logistic curve is unbounded in
# principle; on a small eval set a near-separable region can fit an implausibly large slope, so one
# calibrated observation could otherwise dominate the whole fusion. Clamping keeps any single piece of
# evidence bounded (fusion additionally caps *weak* channels harder — see fuse.TEXT_LLR_CAP). In nats:
# ±6 ≈ evidence odds of ~400:1, already generous for one observation.
DEFAULT_LLR_CLIP = 6.0

# Ridge penalty on the logistic slope/intercept. Keeps the fit finite when genuine/impostor are
# (near-)separable — common for face/voice on clean data — instead of the slope running to infinity.
_RIDGE = 1e-3
_MAX_ITERS = 100
_TOL = 1e-8


@dataclass(frozen=True)
class Curve:
    """One fitted logistic calibration: ``LLR(s) = clip(a·s + b − prior_logodds)``.

    ``prior_logodds`` is ``log(n_genuine / n_impostor)`` on the fit set, subtracted so the returned
    value is a prior-free LLR (fusion adds its own prior). ``n_genuine``/``n_impostor`` are kept for
    provenance and so a caller can down-weight a curve fit on too little data."""
    a: float
    b: float
    prior_logodds: float
    n_genuine: int
    n_impostor: int
    clip: float = DEFAULT_LLR_CLIP

    def llr(self, cosine: float) -> float:
        c = float(cosine)
        # A non-finite similarity is NOT evidence — abstain (0.0). Without this guard a NaN cosine
        # (e.g. a zero-norm embedding → NaN pgvector cosine distance) would clip to +clip and become
        # MAXIMUM supporting evidence, enough to authorize on a single candidate (codex). min/max don't
        # sanitise NaN, so reject it explicitly before clipping.
        if not math.isfinite(c):
            return 0.0
        raw = self.a * c + self.b - self.prior_logodds
        if not math.isfinite(raw):
            return 0.0
        return float(max(-self.clip, min(self.clip, raw)))

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "prior_logodds": self.prior_logodds,
                "n_genuine": self.n_genuine, "n_impostor": self.n_impostor, "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Curve":
        return cls(a=float(d["a"]), b=float(d["b"]), prior_logodds=float(d.get("prior_logodds", 0.0)),
                   n_genuine=int(d.get("n_genuine", 0)), n_impostor=int(d.get("n_impostor", 0)),
                   clip=float(d.get("clip", DEFAULT_LLR_CLIP)))


@dataclass(frozen=True)
class Calibration:
    """A channel's calibration: a pooled curve plus optional per-length-band curves (text_style).

    ``llr(cosine, length=None)`` picks the band whose edge the retained authored ``length`` falls in
    (using ``band_edges``), falling back to the pooled curve when no band matches or no length is given
    — so voice/face (which pass no length) transparently use the pooled curve. Serialises to/from the
    JSONB the store keeps in ``identity_calibration.curve``."""
    channel: str
    pooled: Curve
    bands: dict[str, Curve] = field(default_factory=dict)
    band_edges: tuple[int, ...] = ()

    def _band_label(self, length: int) -> str:
        lo = 0
        for hi in self.band_edges:
            if length <= hi:
                return f"{lo + 1}-{hi}" if lo else f"<={hi}"
            lo = hi
        return f">{self.band_edges[-1]}" if self.band_edges else ""

    def curve_for(self, length: Optional[int]) -> Curve:
        """The curve that governs an observation of this retained authored length (pooled if no band
        applies). Exposed so callers/tests can see which band an observation resolved to."""
        if length is not None and self.bands and self.band_edges:
            c = self.bands.get(self._band_label(int(length)))
            if c is not None:
                return c
        return self.pooled

    def llr(self, cosine: float, length: Optional[int] = None) -> float:
        return self.curve_for(length).llr(cosine)

    @property
    def fitted(self) -> bool:
        """True when this curve was fit from real data (has sample provenance). A DEFAULT/analytic
        curve (seeded so a channel isn't inert, e.g. `nmem-identity seed-text-calibration`) has zero
        samples — fusion lets it inform a soft/speculative hint but never count toward authorisation
        (trust follows the eval)."""
        return (self.pooled.n_genuine + self.pooled.n_impostor) > 0

    # ---- (de)serialisation to the store's JSONB curve ----
    def as_curve_json(self) -> dict:
        return {"type": "logistic", "channel": self.channel, "pooled": self.pooled.as_dict(),
                "bands": {k: v.as_dict() for k, v in self.bands.items()},
                "band_edges": list(self.band_edges)}

    @classmethod
    def from_curve_json(cls, channel: str, curve: dict) -> "Calibration":
        pooled = Curve.from_dict(curve["pooled"])
        bands = {k: Curve.from_dict(v) for k, v in (curve.get("bands") or {}).items()}
        edges = tuple(int(x) for x in (curve.get("band_edges") or ()))
        return cls(channel=channel, pooled=pooled, bands=bands, band_edges=edges)


# ---- fitting -------------------------------------------------------------------------------------

def _fit_logistic(cos: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Ridge-regularised 1-D logistic regression via IRLS on features [1, cos] → (a, b) such that
    ``logit P(y=1 | s) = a·s + b``. The ridge term keeps (a, b) finite under (near-)separable data.
    Falls back to the last stable estimate if a step goes non-finite."""
    x = np.asarray(cos, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x])            # [1, s]
    w = np.zeros(2, dtype=np.float64)                    # [b, a]
    ridge = _RIDGE * np.eye(2)
    for _ in range(_MAX_ITERS):
        eta = np.clip(X @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (p - yv) + _RIDGE * w
        s = p * (1.0 - p)
        H = X.T @ (X * s[:, None]) + ridge
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)):
            break
        if float(np.max(np.abs(w_new - w))) < _TOL:
            w = w_new
            break
        w = w_new
    b, a = float(w[0]), float(w[1])
    return a, b


def fit_curve(genuine: Sequence[float], impostor: Sequence[float], *,
              clip: float = DEFAULT_LLR_CLIP) -> Curve:
    """Fit one prior-free LLR curve from genuine (same-person) and impostor (different-person) cosine
    scores. Needs at least one of each; raises ValueError otherwise (a curve fit on one class is
    meaningless). ``prior_logodds = log(n_gen / n_imp)`` is baked in so ``llr`` returns evidence, not
    a posterior."""
    g = np.asarray([float(s) for s in genuine], dtype=np.float64)
    im = np.asarray([float(s) for s in impostor], dtype=np.float64)
    if g.size == 0 or im.size == 0:
        raise ValueError("fit_curve needs at least one genuine and one impostor score")
    cos = np.concatenate([g, im])
    y = np.concatenate([np.ones_like(g), np.zeros_like(im)])
    a, b = _fit_logistic(cos, y)
    prior_logodds = math.log(g.size / im.size)
    return Curve(a=a, b=b, prior_logodds=prior_logodds,
                 n_genuine=int(g.size), n_impostor=int(im.size), clip=float(clip))


def fit_banded(scored: Sequence[tuple[float, bool, int]], *, band_edges: Sequence[int],
               channel: str, min_per_band: int = 20, clip: float = DEFAULT_LLR_CLIP) -> Calibration:
    """Fit a length-banded calibration from scored pairs ``(cosine, same_author, retained_length)``.

    Always fits the pooled curve over all pairs. For each length band with at least ``min_per_band``
    pairs *and* both classes present, fits a band-specific curve; sparser bands fall back to pooled at
    eval time (no band entry written). Bands are labelled exactly as ``eval/text_benchmark`` /
    ``Calibration._band_label`` so a stored band key resolves correctly. Voice/face callers use
    ``fit_curve`` directly (single pooled curve, no length)."""
    edges = tuple(int(e) for e in band_edges)
    pooled = fit_curve([s for s, same, _ in scored if same],
                       [s for s, same, _ in scored if not same], clip=clip)
    # Group by band label, using the same labelling Calibration uses to look them up.
    probe = Calibration(channel=channel, pooled=pooled, band_edges=edges)
    buckets: dict[str, list[tuple[float, bool]]] = {}
    for s, same, length in scored:
        buckets.setdefault(probe._band_label(int(length)), []).append((float(s), bool(same)))
    bands: dict[str, Curve] = {}
    for label, rows in buckets.items():
        gen = [s for s, same in rows if same]
        imp = [s for s, same in rows if not same]
        if len(rows) >= min_per_band and gen and imp:
            bands[label] = fit_curve(gen, imp, clip=clip)
    return Calibration(channel=channel, pooled=pooled, bands=bands, band_edges=edges)
