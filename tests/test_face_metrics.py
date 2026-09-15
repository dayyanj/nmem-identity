"""Verification metrics + LFW pair parsing — pure functions, no model/dataset needed."""
import numpy as np

from nmem_identity.eval import face_benchmark as FB


def test_metrics_on_separable_scores():
    sims = np.array([0.9, 0.85, 0.8, 0.1, 0.05, 0.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    m = FB.verification_metrics(sims, labels)
    assert m["n"] == 6 and m["eer"] == 0.0 and m["acc"] == 1.0


def test_metrics_empty_is_nan_not_crash():
    m = FB.verification_metrics(np.array([]), np.array([]))
    assert m["n"] == 0 and np.isnan(m["acc"]) and np.isnan(m["eer"])


def test_score_pairs_skips_missing_embeddings():
    embs = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    sims, labels, skipped = FB.score_pairs([("a", "b", True), ("a", "gone", False)], embs)
    assert skipped == 1 and len(sims) == 1 and labels[0] == 1
    assert sims[0] == 1.0                        # identical unit vectors → cosine 1


def test_lfw_pairs_parser(tmp_path):
    p = tmp_path / "pairs.txt"
    p.write_text("2 1\nAlice 1 2\nAlice 3 Bob 1\n")   # header (skipped) + genuine + impostor
    pairs = FB.load_lfw_pairs(str(p), "/root")
    assert len(pairs) == 2
    a, b, same = pairs[0]
    assert same is True and a.endswith("Alice/Alice_0001.jpg") and b.endswith("Alice/Alice_0002.jpg")
    a2, b2, same2 = pairs[1]
    assert same2 is False and a2.endswith("Alice/Alice_0003.jpg") and b2.endswith("Bob/Bob_0001.jpg")


def test_generic_pairs_parser(tmp_path):
    import json
    p = tmp_path / "pairs.json"
    p.write_text(json.dumps([["x/1.jpg", "x/2.jpg", True], ["x/1.jpg", "y/1.jpg", False]]))
    pairs = FB.load_generic_pairs(str(p), "/data")
    assert len(pairs) == 2 and pairs[0][2] is True and pairs[1][2] is False
    assert pairs[0][0].endswith("x/1.jpg")
