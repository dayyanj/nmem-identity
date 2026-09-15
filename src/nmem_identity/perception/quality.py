"""Per-observation face quality — size / pose / blur / detector-confidence gates.

Quality has two jobs, matching how the voice channel already uses it:
  1. **Regime labelling** — a low-quality face is a `face_degraded` observation; the benchmark
     stratifies accuracy by regime, so quality feeds that label.
  2. **Learning gate** — a face may still be *recognised* from a poor crop, but a poor crop must NOT
     move a learned centroid or get persisted as a template (an unreliable vector pollutes identity).
     `cluster.match_or_create` already gates centroid updates on `quality >= min_quality`; this module
     produces the scalar it gates on.

The score is a heuristic in [0,1] built from four interpretable parts, each also returned raw so a
deployment can re-threshold without recomputing. cv2 is lazy-imported (blur needs a Laplacian); the
pose math is pure numpy over the 5-point landmarks SCRFD returns.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Tunable thresholds (documented defaults; a deployment overrides by re-scoring or config later).
MIN_FACE_PX = 60.0          # min(bbox side) below this → too small to trust
GOOD_FACE_PX = 130.0        # at/above this → full size credit
MIN_DET_SCORE = 0.5         # SCRFD confidence floor
BLUR_FLOOR = 30.0           # variance-of-Laplacian below this → too blurry
BLUR_GOOD = 150.0           # at/above this → full sharpness credit
YAW_FULL = 0.06             # |nose offset / inter-ocular| below this → fully frontal
YAW_ZERO = 0.35             # at/above this → treated as profile (0 frontality)

# Gate: an observation passes (may update the centroid / be stored) only if quality clears this.
QUALITY_GATE = 0.45


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _lerp_credit(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 at/above hi, linear between — a soft floor/ceiling ramp."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return _clamp01((x - lo) / (hi - lo))


def frontality(kps: np.ndarray) -> tuple[float, dict[str, float]]:
    """Estimate frontal-ness from the 5 landmarks [L-eye, R-eye, nose, L-mouth, R-mouth].

    Yaw proxy: horizontal offset of the nose from the eye-midpoint, normalised by inter-ocular
    distance — ~0 frontal, grows toward profile. Also returns roll (eye-line tilt) for diagnostics.
    Pure heuristic, but monotone in the thing we care about and free (no extra model)."""
    kps = np.asarray(kps, dtype=np.float64)
    l_eye, r_eye, nose = kps[0], kps[1], kps[2]
    eye_mid = (l_eye + r_eye) / 2.0
    iod = float(np.linalg.norm(r_eye - l_eye)) or 1e-6           # inter-ocular distance
    yaw_ratio = abs(float(nose[0] - eye_mid[0])) / iod
    front = _clamp01(1.0 - _lerp_credit(yaw_ratio, YAW_FULL, YAW_ZERO))
    roll_deg = float(np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0])))
    return front, {"yaw_ratio": yaw_ratio, "roll_deg": roll_deg, "iod_px": iod}


def sharpness(aligned_bgr: np.ndarray) -> float:
    """Variance of the Laplacian of the aligned crop — the standard cheap blur proxy (higher=sharper)."""
    try:
        import cv2  # type: ignore
    except Exception:  # pragma: no cover - blur credit falls back to 'unknown-good' without cv2
        return BLUR_GOOD
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(bgr: np.ndarray, det: dict[str, Any],
                 aligned_bgr: np.ndarray) -> tuple[float, dict[str, float]]:
    """Combine size + detector score + frontality + sharpness into a [0,1] quality, plus the parts.

    Combination is a **product** of the four credits, so a hard failure on any single axis (tiny,
    blurry, or profile) drags quality down — you cannot buy back a profile shot with a high detector
    score. That matches what we want the learning gate to enforce."""
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    face_px = min(x2 - x1, y2 - y1)
    size_credit = _lerp_credit(face_px, MIN_FACE_PX, GOOD_FACE_PX)

    det_score = float(det.get("det_score", 0.0))
    det_credit = _lerp_credit(det_score, MIN_DET_SCORE, 0.9)

    kps = det.get("kps")
    if kps is not None:
        front, parts = frontality(kps)
    else:
        front, parts = 1.0, {}                       # boxcrop path has no landmarks; don't penalise pose

    blur_var = sharpness(aligned_bgr)
    blur_credit = _lerp_credit(blur_var, BLUR_FLOOR, BLUR_GOOD)

    quality = float(size_credit * det_credit * front * blur_credit)
    parts.update({
        "face_px": face_px, "size_credit": size_credit,
        "det_score": det_score, "det_credit": det_credit,
        "frontality": front, "blur_var": blur_var, "blur_credit": blur_credit,
        "quality": quality,
    })
    return quality, parts


def regime(parts: dict[str, float]) -> str:
    """Map quality parts to a benchmark regime label. `no_face` is handled by the caller (empty
    detection); here we split present faces into frontal vs degraded."""
    if parts.get("quality", 0.0) >= QUALITY_GATE and parts.get("frontality", 1.0) >= 0.7:
        return "face_frontal"
    return "face_degraded"


def passes_gate(quality: float) -> bool:
    return quality >= QUALITY_GATE
