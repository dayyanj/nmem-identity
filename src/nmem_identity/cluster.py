"""Online incremental speaker matching — the live recognition core.

Given one utterance's voiceprint, decide whether it belongs to an existing cluster (recognise) or is a
new voice (create a proto-cluster), and — when learning is on and the utterance is good enough — attach
the template and nudge the centroid. This is identity-grade nearest-centroid matching with margin +
quality guards; it is NOT the nmem-sym word-clustering (that normalises speakers away — the opposite
goal). Thresholds here are PROVISIONAL (CAM++ same-speaker cosine ~0.6, different ~0.01); they get
tuned on real data at Gate 0.

Guards (plan Risks #3/#4):
  * `tau_match` — top-1 cosine must clear this to count as a match, else it's a new voice.
  * `margin`    — top-1 must beat top-2 by this, so an ambiguous near-tie makes a new cluster rather
    than a false-merge into whichever centroid happened to be marginally closer.
  * `min_quality` — short/low-quality utterances (< ~1.5s) still get RECOGNISED, but do not attach a
    template or move the centroid (an unreliable vector must not pollute a learned identity).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .store import templates as T

# Provisional; tuned at Gate 0. quality here = utterance duration (seconds), as the tap records it.
DEFAULT_TAU_MATCH = 0.50
DEFAULT_MARGIN = 0.10
DEFAULT_MIN_QUALITY = 1.5


async def match_or_create(embedder: str, vec: Sequence[float], *,
                          session_id: Optional[str] = None, modality: str = "voice",
                          quality: Optional[float] = None, raw_ref: Optional[str] = None,
                          regime: Optional[str] = None, embed_version: int = 1,
                          tau_match: float = DEFAULT_TAU_MATCH, margin: float = DEFAULT_MARGIN,
                          min_quality: float = DEFAULT_MIN_QUALITY, learn: bool = True) -> dict[str, Any]:
    """Recognise or enrol a voiceprint. Returns:
        {action, cluster_id, person_id, is_new, sim, top2_sim, margin, grounding_status, learned}
    where action ∈ {matched, created, matched_nolearn}. Never raises on a normal miss — a miss is a
    new proto-cluster, which is the enrolment substrate."""
    hits = await T.knn_clusters(embedder, vec, k=2, modality=modality)
    top = hits[0] if hits else None
    top2 = hits[1] if len(hits) > 1 else None
    sim = float(top["sim"]) if top else -1.0
    top2_sim = float(top2["sim"]) if top2 else None
    gap = (sim - top2_sim) if top2_sim is not None else sim   # margin over the runner-up (or absolute)

    matched = bool(top) and sim >= tau_match and gap >= margin
    quality_ok = quality is None or quality >= min_quality

    if matched:
        cid = int(top["cluster_id"])
        person_id = top["person_id"]
        grounding = top["grounding_status"]
        learned = False
        if learn and quality_ok:
            await T.insert_template(cid, {embedder: vec}, modality=modality, quality=quality,
                                    raw_ref=raw_ref, session_id=session_id, regime=regime,
                                    embed_versions={embedder: embed_version})
            await T.ema_update_centroid(cid, embedder, vec, embed_version=embed_version)
            await T.bump_member(cid)
            learned = True
        return {"action": "matched" if learned else "matched_nolearn",
                "cluster_id": cid, "person_id": person_id, "is_new": False,
                "sim": sim, "top2_sim": top2_sim, "margin": gap,
                "grounding_status": grounding, "learned": learned}

    # Miss. Only CREATE a proto-cluster when we can also seed it (learn + good quality). Creating an
    # unseeded cluster would be a durable leak: KNN filters `WHERE centroid IS NOT NULL`, so a
    # centroid-less cluster is invisible to all future matches — the same voice would just spawn
    # another cluster next time. A too-short / non-learning first sighting instead reports 'unknown'
    # and waits for a good utterance to enrol the voice.
    if not (learn and quality_ok):
        return {"action": "unknown_lowquality" if not quality_ok else "unknown_nolearn",
                "cluster_id": None, "person_id": None, "is_new": False,
                "sim": sim, "top2_sim": top2_sim, "margin": gap,
                "grounding_status": None, "learned": False}
    # The three writes below aren't wrapped in one transaction; ema_update_centroid is itself atomic
    # and bump_member is a single statement, so a mid-way failure can at worst leave member_count off
    # by one (a soft signal, self-healing on the next batch rebuild) — never a corrupt centroid.
    cid = await T.create_cluster(modality=modality, grounding_status="ungrounded")
    await T.insert_template(cid, {embedder: vec}, modality=modality, quality=quality,
                            raw_ref=raw_ref, session_id=session_id, regime=regime,
                            embed_versions={embedder: embed_version})
    await T.ema_update_centroid(cid, embedder, vec, embed_version=embed_version)
    await T.bump_member(cid)
    return {"action": "created", "cluster_id": cid, "person_id": None, "is_new": True,
            "sim": sim, "top2_sim": top2_sim, "margin": gap,
            "grounding_status": "ungrounded", "learned": True}
