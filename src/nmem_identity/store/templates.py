"""Data-access layer for the nmem-identity store (the identity_* tables).

Everything here is the *store* — no matching policy, no calibration, no fusion (those live in
cluster.py / calibrate.py / fuse.py). The invariants this module upholds:

  * Every similarity query filters `WHERE <e>_model_id = :model_id` so cross-model cosine is
    impossible (design §2.2). The caller passes the active model_id; there is no implicit default.
  * A template's per-embedder triple (emb, model_id, embed_version) is written atomically or not at
    all — the schema CHECKs reject a partial triple, so we build the column set dynamically per turn.
  * `delete_person` is the CANONICAL delete: it removes the person's clusters (templates CASCADE),
    returns the raw_ref blob keys the caller must unlink, and leaves no orphan vectors. The
    cluster->person FK is ON DELETE SET NULL only as a degradation guard, never the delete path.

pgvector's asyncpg codec: we register `vector` as text on each connection (send/recv as the
'[1,2,3]' literal) so we don't hard-depend on the pgvector-python asyncpg helper being installed.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from .. import models
from .db import get_pool

# model_id + dim are the schema-bound constants; they live in one place (nmem_identity.models) so the
# store, perception, and the DDL literals can never drift. Here we add each embedder's SQL column map.
CAMPP_MODEL_ID = models.model_id("campp")
FACE_MODEL_ID = models.model_id("face")

# name -> (centroid column, emb column, model_id col, version col, model_id, dim)
_COLUMNS: dict[str, dict[str, str]] = {
    "campp": {"centroid": "centroid_campp", "emb": "campp_emb",
              "mid_col": "campp_model_id", "ver_col": "campp_embed_version"},
    "face": {"centroid": "centroid_face", "emb": "face_emb",
             "mid_col": "face_model_id", "ver_col": "face_embed_version"},
    # LUAR authorship (chat-writing style). Column prefix is `text` (the embedder name is `text_style`);
    # everything below is data-driven off this map, so KNN/EMA/insert/rebuild/delete work unchanged.
    "text_style": {"centroid": "centroid_text", "emb": "text_emb",
                   "mid_col": "text_model_id", "ver_col": "text_embed_version"},
}
_EMBEDDERS: dict[str, dict[str, Any]] = {
    name: {**cols, "model_id": models.model_id(name), "dim": models.dim(name)}
    for name, cols in _COLUMNS.items()
}


def _vec_literal(vec: Sequence[float]) -> str:
    """pgvector text form: '[0.1,0.2,...]'. Compact, no spaces (pgvector parses either)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _check_dim(embedder: str, vec: Sequence[float]) -> None:
    exp = _EMBEDDERS[embedder]["dim"]
    if len(vec) != exp:
        raise ValueError(f"{embedder} expects dim {exp}, got {len(vec)}")


_SAFE_MODEL_ID = re.compile(r"[a-z0-9_]+")


def _model_literal(embedder: str) -> str:
    """The embedder's active model_id as a SQL string literal, for inlining into the KNN predicate.
    Postgres only uses the partial HNSW index when the query predicate is a *literal* that implies
    the index predicate — a bound `$n` yields a generic plan that can't prove the match, so ANN
    silently degrades to a seqscan. model_id comes from the internal `_EMBEDDERS` constants (never
    user input) and is asserted `[a-z0-9_]+`, so inlining is injection-safe."""
    e = _EMBEDDERS[embedder]
    mid = e["model_id"]
    if mid is None:
        raise ValueError(f"{embedder} has no active model_id (not yet enabled)")
    if not _SAFE_MODEL_ID.fullmatch(mid):        # defence-in-depth; constants already conform
        raise ValueError(f"unsafe model_id {mid!r}")
    return f"'{mid}'"


# ---- person ---------------------------------------------------------------------------------

async def create_person(display_name: Optional[str] = None,
                        nmem_entity_id: Optional[str] = None) -> int:
    pool = await get_pool()
    return await pool.fetchval(
        "INSERT INTO identity_person (display_name, nmem_entity_id) VALUES ($1, $2) "
        "RETURNING person_id",
        display_name, nmem_entity_id,
    )


async def bind_cluster_person(cluster_id: int, person_id: int,
                              grounding_status: str = "confident") -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE identity_cluster SET person_id = $1, grounding_status = $2, "
        "grounded_at = now() WHERE cluster_id = $3",
        person_id, grounding_status, cluster_id,
    )


# Which embedder centroids belong to each modality (first present one is the snapshot baseline).
_MODALITY_EMBEDDERS: dict[str, tuple[str, ...]] = {
    "voice": ("campp",),
    "face": ("face",),
    "text_style": ("text_style",),
    "multi": ("face", "campp", "text_style"),
}


async def snapshot_centroid(cluster_id: int) -> bool:
    """Snapshot a cluster's centroid into `centroid_snapshot` as the rechallenge drift baseline,
    picking the centroid column by the cluster's modality (voice→CAM++, face→ArcFace, ...). Modality-
    agnostic so enrolment works for a face cluster exactly as for a voice one. Returns True if a
    non-null centroid was snapshotted."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        modality = await conn.fetchval(
            "SELECT modality FROM identity_cluster WHERE cluster_id = $1", cluster_id)
        if modality is None:
            return False
        for name in _MODALITY_EMBEDDERS.get(modality, ("campp",)):
            cen = _EMBEDDERS[name]["centroid"]
            updated = await conn.fetchval(
                f"UPDATE identity_cluster SET centroid_snapshot = {cen}::text "
                f"WHERE cluster_id = $1 AND {cen} IS NOT NULL RETURNING cluster_id",
                cluster_id)
            if updated is not None:
                return True
    return False


async def delete_person(person_id: int) -> list[str]:
    """Canonical delete. Removes the person's clusters (templates CASCADE) and the person row,
    inside one transaction. Returns the raw_ref blob keys the caller MUST unlink from the blob
    store, and the caller separately clears the nmem entity + curiosity rows AND the nmem-sym
    relational person-aliases (via ``SymbolBridge.forget_person_aliases`` — Phase 6-A dossier
    convergence) for this person's refs (all cross-schema). Leaves zero orphan vectors."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT t.raw_ref FROM identity_template t "
                "JOIN identity_cluster c ON c.cluster_id = t.cluster_id "
                "WHERE c.person_id = $1 AND t.raw_ref IS NOT NULL",
                person_id,
            )
            # Deleting clusters CASCADEs their templates; then the person row.
            await conn.execute("DELETE FROM identity_cluster WHERE person_id = $1", person_id)
            await conn.execute("DELETE FROM identity_person WHERE person_id = $1", person_id)
    return [r["raw_ref"] for r in rows]


# ---- cluster --------------------------------------------------------------------------------

async def create_cluster(modality: str = "voice",
                         grounding_status: str = "ungrounded") -> int:
    """Create an empty proto-cluster (no centroid yet — set on first template via EMA update)."""
    pool = await get_pool()
    return await pool.fetchval(
        "INSERT INTO identity_cluster (modality, grounding_status) VALUES ($1, $2) "
        "RETURNING cluster_id",
        modality, grounding_status,
    )


async def create_cluster_for_person(person_id: int, modality: str = "voice",
                                    grounding_status: str = "confident") -> int:
    """Create a cluster already BOUND to a person, in one atomic INSERT (no create-then-bind window
    that could orphan a cluster if the bind failed)."""
    pool = await get_pool()
    return await pool.fetchval(
        "INSERT INTO identity_cluster (modality, grounding_status, person_id, grounded_at) "
        "VALUES ($1, $2, $3, now()) RETURNING cluster_id",
        modality, grounding_status, person_id,
    )


async def rebuild_cluster_centroid(cluster_id: int, embedder: str, embed_version: int = 1) -> None:
    """Recompute a cluster's centroid + member_count from its templates in one transaction. Used by
    batch ingest instead of per-utterance EMA: it's order-independent and resilient to a mid-run
    failure (member_count/centroid always reflect exactly the templates that landed). member_count is
    the cluster's total template (turn) count; the centroid is the mean of this embedder's vectors."""
    e = _EMBEDDERS[embedder]
    if e["model_id"] is None:
        raise ValueError(f"{embedder} has no active model_id (not yet enabled)")
    cen, emb, mid, ver = e["centroid"], e["emb"], e["mid_col"], e["ver_col"]
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"SELECT (avg({emb}) FILTER (WHERE {mid} = $2))::text AS mean, "
                f"       count(*) FILTER (WHERE {emb} IS NOT NULL AND {mid} = $2) AS n_e, "
                f"       count(*) AS n_all "
                f"FROM identity_template WHERE cluster_id = $1",
                cluster_id, e["model_id"],
            )
            n_all = int(row["n_all"])
            if row["mean"] is None or int(row["n_e"]) == 0:
                # no vectors for this embedder — set only member_count, keep the all-or-none triple.
                await conn.execute(
                    "UPDATE identity_cluster SET member_count = $1 WHERE cluster_id = $2",
                    n_all, cluster_id)
                return
            await conn.execute(
                f"UPDATE identity_cluster SET {cen} = $1::vector, {mid} = $2, {ver} = $3, "
                f"member_count = $4 WHERE cluster_id = $5",
                row["mean"], e["model_id"], int(embed_version), n_all, cluster_id)


async def knn_clusters(embedder: str, vec: Sequence[float], k: int = 5,
                       modality: str = "voice") -> list[dict[str, Any]]:
    """K nearest clusters by cosine on this embedder's centroid, FILTERED to its model_id (so only
    same-generation centroids are ever compared). Returns [{cluster_id, sim, member_count,
    grounding_status, person_id}] with sim = 1 - cosine_distance, descending."""
    _check_dim(embedder, vec)
    e = _EMBEDDERS[embedder]
    cen, mid = e["centroid"], e["mid_col"]
    mid_lit = _model_literal(embedder)           # literal so the partial HNSW index is used
    pool = await get_pool()
    rows = await pool.fetch(
        f"SELECT cluster_id, member_count, grounding_status, person_id, "
        f"       1 - ({cen} <=> $1::vector) AS sim "
        f"FROM identity_cluster "
        f"WHERE {cen} IS NOT NULL AND {mid} = {mid_lit} AND modality = $2 "
        f"ORDER BY {cen} <=> $1::vector "
        f"LIMIT $3",
        _vec_literal(vec), modality, k,
    )
    return [dict(r) for r in rows]


async def ema_update_centroid(cluster_id: int, embedder: str, vec: Sequence[float],
                              alpha: float = 0.1, embed_version: int = 1) -> None:
    """Online EMA centroid update for ONE embedder: c <- (1-a)*c + a*v.

    Reseed (c <- v) when there's nothing to blend into the SAME generation: centroid NULL, or the
    stored (model_id, embed_version) differs from the incoming one. This makes a model/version bump
    replace the centroid instead of silently averaging vectors across generations (a stale-blend
    correctness bug). Renormalisation is not applied (pgvector cosine ops are scale-invariant).

    member_count is NOT touched here — a turn runs *both* embedders, so counting per-embedder would
    double-count the observation. The matcher bumps member_count once per accepted turn via
    `bump_member`."""
    _check_dim(embedder, vec)
    e = _EMBEDDERS[embedder]
    if e["model_id"] is None:
        raise ValueError(f"{embedder} has no active model_id (not yet enabled)")
    cen, mid, ver = e["centroid"], e["mid_col"], e["ver_col"]
    a = float(alpha)
    # pgvector has no scalar*vector operator, so the blend is computed in Python under a row lock
    # (SELECT ... FOR UPDATE) — atomic with the model_id/version write and safe against concurrent
    # updates to the same cluster. Vectors are tiny (192/512) so the extra round-trip is negligible.
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"SELECT {cen}::text AS c, {mid} AS m, {ver} AS v "
                f"FROM identity_cluster WHERE cluster_id = $1 FOR UPDATE",
                cluster_id,
            )
            if row is None:
                raise ValueError(f"cluster {cluster_id} not found")
            stale = row["c"] is None or row["m"] != e["model_id"] or row["v"] != int(embed_version)
            if stale:                                   # reseed: nothing to blend into this generation
                new_vec = [float(x) for x in vec]
            else:
                cur = json.loads(row["c"])              # pgvector '[...]' text is valid JSON
                new_vec = [(1.0 - a) * cu + a * vv for cu, vv in zip(cur, vec)]
            await conn.execute(
                f"UPDATE identity_cluster SET {cen} = $1::vector, {mid} = $2, {ver} = $3 "
                f"WHERE cluster_id = $4",
                _vec_literal(new_vec), e["model_id"], int(embed_version), cluster_id,
            )


async def bump_member(cluster_id: int, n: int = 1) -> None:
    """Increment a cluster's member_count once per accepted observation (turn), independent of how
    many embedders fired. Called by the matcher, not by ema_update_centroid."""
    pool = await get_pool()
    await pool.execute(
        "UPDATE identity_cluster SET member_count = member_count + $1 WHERE cluster_id = $2",
        int(n), cluster_id,
    )


# ---- template (one observation / one turn) --------------------------------------------------

async def insert_template(cluster_id: int, embeddings: dict[str, Sequence[float]], *,
                          modality: str = "voice", quality: Optional[float] = None,
                          raw_ref: Optional[str] = None, session_id: Optional[str] = None,
                          regime: Optional[str] = None,
                          embed_versions: Optional[dict[str, int]] = None) -> int:
    """Insert one observation. `embeddings` maps embedder name -> vector for whichever embedders fired
    (e.g. {'campp': [...]} for a voice turn, {'face': [...]} for a detection); each writes its full
    (emb, model_id, embed_version) triple. At least one embedder is required (schema ck_tmpl_any)."""
    if not embeddings:
        raise ValueError("insert_template needs at least one embedder's vector")
    embed_versions = embed_versions or {}
    cols = ["cluster_id", "modality", "quality", "raw_ref", "session_id", "regime"]
    vals: list[Any] = [cluster_id, modality, quality, raw_ref, session_id, regime]
    casts = ["$1", "$2", "$3", "$4", "$5", "$6"]
    n = 6
    for name, vec in embeddings.items():
        e = _EMBEDDERS.get(name)
        if e is None:
            raise ValueError(f"unknown embedder {name!r}")
        if e["model_id"] is None:
            raise ValueError(f"{name} has no active model_id (not yet enabled)")
        _check_dim(name, vec)
        # emb (vector cast)
        n += 1; cols.append(e["emb"]); casts.append(f"${n}::vector"); vals.append(_vec_literal(vec))
        # model_id
        n += 1; cols.append(e["mid_col"]); casts.append(f"${n}"); vals.append(e["model_id"])
        # embed_version
        n += 1; cols.append(e["ver_col"]); casts.append(f"${n}")
        vals.append(int(embed_versions.get(name, 1)))
    sql = (f"INSERT INTO identity_template ({', '.join(cols)}) "
           f"VALUES ({', '.join(casts)}) RETURNING id")
    pool = await get_pool()
    return await pool.fetchval(sql, *vals)


async def iter_templates(embedder: Optional[str] = None,
                         modality: str = "voice") -> list[dict[str, Any]]:
    """Bulk-read templates for the offline benchmark. If `embedder` is given, returns only rows
    where that embedder fired, with its vector parsed to list[float] under key 'emb'."""
    pool = await get_pool()
    if embedder is None:
        rows = await pool.fetch(
            "SELECT id, cluster_id, session_id, quality, regime, captured_at "
            "FROM identity_template WHERE modality = $1 ORDER BY captured_at",
            modality,
        )
        return [dict(r) for r in rows]
    e = _EMBEDDERS[embedder]
    col, mid = e["emb"], e["mid_col"]
    rows = await pool.fetch(
        f"SELECT id, cluster_id, session_id, quality, regime, captured_at, "
        f"       {col}::text AS emb, {e['ver_col']} AS embed_version "
        f"FROM identity_template "
        f"WHERE modality = $1 AND {col} IS NOT NULL AND {mid} = $2 "
        f"ORDER BY captured_at",
        modality, e["model_id"],
    )
    out = []
    for r in rows:
        d = dict(r)
        d["emb"] = json.loads(d["emb"])  # pgvector '[...]' text is valid JSON
        out.append(d)
    return out
