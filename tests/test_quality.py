"""Face quality heuristics — no model needed, pure numpy/geometry."""
import numpy as np

from nmem_identity.perception import quality as Q


def _kps(nose_x):
    # [L-eye, R-eye, nose, L-mouth, R-mouth]; eyes at 40/72, vary nose x to sweep yaw.
    return np.array([[40, 50], [72, 50], [nose_x, 66], [44, 82], [68, 82]], float)


def test_frontality_is_high_for_centred_nose():
    f, parts = Q.frontality(_kps(56))            # nose centred between eyes
    assert f > 0.95 and parts["yaw_ratio"] < 0.05


def test_frontality_decreases_monotonically_with_yaw():
    fs = [Q.frontality(_kps(x))[0] for x in (56, 52, 48, 44)]   # nose drifts toward left eye
    assert all(a >= b for a, b in zip(fs, fs[1:]))              # monotone non-increasing
    assert fs[0] > fs[-1]                                        # and actually drops


def test_roll_detected_from_eye_line_tilt():
    kps = np.array([[40, 50], [72, 62], [56, 70], [44, 86], [68, 92]], float)  # right eye lower
    _, parts = Q.frontality(kps)
    assert parts["roll_deg"] > 5                  # tilted head → non-zero roll


def test_quality_is_product_so_one_bad_axis_tanks_it():
    det = {"bbox": np.array([0, 0, 200, 200], float), "det_score": 0.95, "kps": _kps(56)}
    sharp = np.full((112, 112, 3), 0, np.uint8)
    sharp[::4] = 255                              # high-frequency content → sharp
    good, gp = Q.face_quality(sharp, det, sharp)
    # a tiny face (below MIN_FACE_PX) must drag quality down regardless of a high det score
    det_small = dict(det, bbox=np.array([0, 0, 30, 30], float))
    small_q, _ = Q.face_quality(sharp, det_small, sharp)
    assert small_q < good
    assert 0.0 <= small_q <= 1.0 and 0.0 <= good <= 1.0


def test_regime_split_and_gate():
    assert Q.regime({"quality": 0.8, "frontality": 0.9}) == "face_frontal"
    assert Q.regime({"quality": 0.2, "frontality": 0.3}) == "face_degraded"
    assert Q.passes_gate(Q.QUALITY_GATE + 0.01) is True
    assert Q.passes_gate(Q.QUALITY_GATE - 0.01) is False


def test_missing_landmarks_do_not_penalise_pose():
    # boxcrop path has no kps → frontality credit is neutral (1.0), not a profile penalty.
    det = {"bbox": np.array([0, 0, 200, 200], float), "det_score": 0.95, "kps": None}
    sharp = np.zeros((112, 112, 3), np.uint8); sharp[::4] = 255
    q, parts = Q.face_quality(sharp, det, sharp)
    assert parts["frontality"] == 1.0
