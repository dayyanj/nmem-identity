"""The recognition brain, tested with a STUBBED store — no DB, no GPU.

`cluster.match_or_create` is the live decision the whole system turns on: recognise an existing
identity, refuse to false-merge an ambiguous near-tie, learn only from good-quality observations, and
otherwise open a new proto-cluster. We stub the store layer so this exercises the *policy* (tau /
margin / quality / learn) in isolation and asserts both the returned verdict AND the write side effects
(did it actually learn, or just recognise?)."""
import asyncio

import pytest

import nmem_identity.cluster as C


class FakeStore:
    """Records store calls; knn returns a scripted hit list."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def knn_clusters(self, embedder, vec, k=2, modality="voice"):
        self.calls.append("knn")
        return self.hits

    async def insert_template(self, *a, **k):
        self.calls.append("insert")
        return 1

    async def ema_update_centroid(self, *a, **k):
        self.calls.append("ema")

    async def bump_member(self, *a, **k):
        self.calls.append("bump")

    async def create_cluster(self, *a, **k):
        self.calls.append("create")
        return 42


def _hit(cluster_id=7, sim=0.8, person_id=None, grounding="ungrounded", member_count=3):
    return {"cluster_id": cluster_id, "sim": sim, "person_id": person_id,
            "grounding_status": grounding, "member_count": member_count}


def _patch(monkeypatch, hits):
    fs = FakeStore(hits)
    for name in ("knn_clusters", "insert_template", "ema_update_centroid",
                 "bump_member", "create_cluster"):
        monkeypatch.setattr(C.T, name, getattr(fs, name))
    return fs


def _run(**kw):
    return asyncio.run(C.match_or_create("campp", [0.0] * 192, **kw))


def test_clear_match_recognises_and_learns(monkeypatch):
    fs = _patch(monkeypatch, [_hit(sim=0.80, person_id=5, grounding="confident"), _hit(cluster_id=8, sim=0.40)])
    r = _run(quality=3.0)                       # quality above min → learn
    assert r["action"] == "matched" and r["learned"] is True and r["is_new"] is False
    assert r["cluster_id"] == 7 and r["person_id"] == 5
    assert "insert" in fs.calls and "ema" in fs.calls and "bump" in fs.calls


def test_match_but_low_quality_recognises_without_learning(monkeypatch):
    fs = _patch(monkeypatch, [_hit(sim=0.80), _hit(cluster_id=8, sim=0.40)])
    r = _run(quality=0.5)                       # below min_quality (1.5)
    assert r["action"] == "matched_nolearn" and r["learned"] is False
    assert r["cluster_id"] == 7                 # still recognised
    assert "insert" not in fs.calls and "ema" not in fs.calls  # but centroid untouched


def test_near_tie_makes_a_new_cluster_not_a_false_merge(monkeypatch):
    # top1 clears tau (0.5) but the runner-up is within `margin` (0.1) → ambiguous → new cluster.
    fs = _patch(monkeypatch, [_hit(cluster_id=7, sim=0.80), _hit(cluster_id=8, sim=0.75)])
    r = _run(quality=3.0)
    assert r["action"] == "created" and r["is_new"] is True and r["cluster_id"] == 42
    assert "create" in fs.calls


def test_below_tau_makes_a_new_cluster(monkeypatch):
    fs = _patch(monkeypatch, [_hit(sim=0.30)])
    r = _run(quality=3.0)
    assert r["action"] == "created" and r["is_new"] is True


def test_empty_gallery_creates(monkeypatch):
    fs = _patch(monkeypatch, [])
    r = _run(quality=3.0)
    assert r["action"] == "created" and r["is_new"] is True


def test_miss_without_learn_stays_unknown_no_write(monkeypatch):
    fs = _patch(monkeypatch, [_hit(sim=0.30)])
    r = _run(quality=3.0, learn=False)
    assert r["action"] == "unknown_nolearn" and r["cluster_id"] is None
    assert "create" not in fs.calls           # a non-learning miss must not durably leak a cluster


def test_miss_with_low_quality_stays_unknown_no_write(monkeypatch):
    fs = _patch(monkeypatch, [_hit(sim=0.30)])
    r = _run(quality=0.5)                      # too short to seed a trustworthy new cluster
    assert r["action"] == "unknown_lowquality" and r["cluster_id"] is None
    assert "create" not in fs.calls


def test_margin_reported_is_gap_over_runner_up(monkeypatch):
    _patch(monkeypatch, [_hit(sim=0.80), _hit(cluster_id=8, sim=0.55)])
    r = _run(quality=3.0)
    assert r["sim"] == pytest.approx(0.80) and r["margin"] == pytest.approx(0.25)
