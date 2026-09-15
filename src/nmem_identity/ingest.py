"""Ingest hand-labelled voice captures into the identity store.

Reads a capture log (`captures.jsonl`, one CAM++ voiceprint per utterance) + a `labels.json`
(utt_key -> {speaker, regime}) and loads them into the identity DB:

  * one `identity_person` per distinct speaker label (get-or-create by display_name),
  * one voice `identity_cluster` per person (get-or-create; the person's confident cluster),
  * one `identity_template` per labelled utterance (CAM++ channel, raw_ref = wav path, regime),
  * EMA-updated centroid + member_count per person.

This is the offline bridge from Phase-0 capture to a populated store (Phase 1 / live matching). The
high-frequency live loop never calls this — capture writes disk, ingest is a deliberate batch step.

Idempotency: re-running APPENDS templates (it does not dedupe utterances). Use `--reset` to delete the
listed speakers' existing voice clusters first for a clean reload.

    python -m nmem_identity.ingest \
        --captures <capture-dir>/captures.jsonl \
        --labels   <capture-dir>/labels.json  [--reset]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .store import db
from .store import templates as T


def _utt_key(rec: dict) -> str:
    wav = rec.get("wav") or ""
    return os.path.basename(wav) if wav else f"{rec.get('session_id')}:{rec.get('ts')}"


async def _get_or_create_person(display_name: str) -> int:
    pool = await db.get_pool()
    pid = await pool.fetchval(
        "SELECT person_id FROM identity_person WHERE display_name = $1 ORDER BY person_id LIMIT 1",
        display_name)
    if pid is not None:
        return pid
    return await T.create_person(display_name, nmem_entity_id=f"person:{display_name}")


async def _get_or_create_voice_cluster(person_id: int) -> int:
    pool = await db.get_pool()
    cid = await pool.fetchval(
        "SELECT cluster_id FROM identity_cluster WHERE person_id = $1 AND modality = 'voice' "
        "ORDER BY cluster_id LIMIT 1", person_id)
    if cid is not None:
        return cid
    return await T.create_cluster_for_person(person_id, modality="voice",
                                             grounding_status="confident")  # atomic create+bind


async def _reset_speakers(speakers: set[str]) -> None:
    pool = await db.get_pool()
    # Delete voice clusters for ALL persons sharing each display_name (no uniqueness constraint on
    # display_name, so a get-or-create could in principle have produced duplicates). templates CASCADE.
    for sp in speakers:
        await pool.execute(
            "DELETE FROM identity_cluster WHERE modality = 'voice' AND person_id IN "
            "(SELECT person_id FROM identity_person WHERE display_name = $1)", sp)
    print(f"[reset] cleared voice clusters for {sorted(speakers)}")


async def run(captures: str, labels: str, reset: bool = False) -> None:
    lab = json.loads(Path(labels).read_text())
    recs = []
    for line in Path(captures).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("campp_emb"):
            continue
        meta = lab.get(_utt_key(rec))
        if meta and meta.get("speaker"):
            recs.append((rec, meta))

    speakers = {m["speaker"] for _, m in recs}
    if not recs:
        print("no labelled utterances to ingest")
        return
    if reset:
        await _reset_speakers(speakers)

    person_id: dict[str, int] = {}
    cluster_id: dict[str, int] = {}
    for sp in sorted(speakers):
        person_id[sp] = await _get_or_create_person(sp)
        cluster_id[sp] = await _get_or_create_voice_cluster(person_id[sp])

    n = 0
    per_spk = {sp: 0 for sp in speakers}
    for rec, meta in recs:
        sp = meta["speaker"]
        cid = cluster_id[sp]
        vec = rec["campp_emb"]
        # Insert templates only in the loop (a single atomic op each). Centroids + member_count are
        # rebuilt authoritatively per cluster afterwards, so a partial run stays self-consistent.
        await T.insert_template(cid, {"campp": vec}, modality="voice",
                                quality=rec.get("duration_s"), raw_ref=rec.get("wav"),
                                session_id=rec.get("session_id"), regime=meta.get("regime", "clean"))
        n += 1
        per_spk[sp] += 1

    for sp in sorted(speakers):
        await T.rebuild_cluster_centroid(cluster_id[sp], "campp")   # atomic recompute from templates

    print(f"[ingest] loaded {n} utterances into {len(speakers)} person/cluster(s):")
    for sp in sorted(speakers):
        print(f"   {sp:15s} person={person_id[sp]} cluster={cluster_id[sp]} templates={per_spk[sp]}")
    await db.close_pool()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--reset", action="store_true",
                    help="delete the listed speakers' existing voice clusters before loading")
    args = ap.parse_args()
    asyncio.run(run(args.captures, args.labels, args.reset))


if __name__ == "__main__":
    main()
