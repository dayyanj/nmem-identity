"""Identity-matcher HTTP service — the live recognition endpoint, run next to the DB.

A host agent computes an embedding (a voice CAM++ vector, a face ArcFace vector, ...) and POSTs it
here; this service does the DB-side nearest-centroid match / proto-cluster creation
(nmem_identity.cluster.match_or_create) and returns who it thinks the person is. Keeping the DB
writes behind this service is deliberate: the host only embeds + calls in over HTTP and never opens
an asyncpg pool to the identity DB itself.

    POST /match  { embedding:[floats], embedder?, session_id?, quality?, learn? }
       -> { action, cluster_id, person_id, person_name, is_new, sim, top2_sim, margin,
            grounding_status, learned }
    GET  /health

`person_name` is resolved from `identity_person.display_name` when the matched cluster is bound to a
person (i.e. an enrolled/known identity); it is null for an as-yet-unnamed one. Best-effort by
design — a caller should treat any failure as "unknown" and never block on it.

Run with a Python that has the store deps (asyncpg) and can reach the DB. Configure the DB via
NMEM_IDENTITY_DSN and the port via NMEM_IDENTITY_MATCHER_PORT (see config.py):
    PYTHONPATH=<repo>/src python -m uvicorn matcher_server:app  # or run this file directly
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from nmem_identity import cluster, config, fuse, models, viz
from nmem_identity.store import calibration as calib_store, db, templates as T

PORT = int(config.get("matcher_port"))
CAMPP_DIM = 192
_VALID_EMBEDDERS = ("campp", "face", "text_style")
# match_or_create's `modality` column uses different names than the embedder itself (historical:
# "voice" predates the "campp" embedder name). Must be passed explicitly -- its default ("voice")
# would silently mislabel every face cluster otherwise. text_style's cluster modality IS 'text_style'.
_EMBEDDER_MODALITY = {"campp": "voice", "face": "face", "text_style": "text_style"}
# An unknown (unbound) cluster observed at least this many times is worth ASKING who it is (recurrence
# gate for conversational enrolment — the system decides WHEN to ask, so it isn't left to the LLM).
ASK_RECURRENCE = int(config.get("ask_recurrence"))
_log = logging.getLogger("matcher")

# Serialise match_or_create: it is read-then-create (KNN, then create a cluster on a miss). Two
# concurrent first-sightings of the SAME new person could both miss and both create a cluster (an
# over-split that then fails the margin test forever). All calls funnel through this one service, so a
# process-level lock makes the whole match+learn step atomic w.r.t. other matches. Cheap: matches are
# ~ms and a single-camera / single-mic host observes one person at a time.
_match_lock = asyncio.Lock()

# Calibration curves (channel -> calibrate.Calibration), preloaded at startup so fusion has evidence
# weights in memory. Empty until an operator runs eval/fit_calibration.py; an uncalibrated channel
# simply abstains in fusion (fuse.fuse contributes 0 LLR — trust follows the eval, never precedes it).
_CALIBRATIONS: dict = {}
# The read-only KNN modality per channel (mirrors _EMBEDDER_MODALITY, but keyed for the /resolve gather).
_CHANNEL_MODALITY = {"campp": "voice", "face": "face", "text_style": "text_style"}
# The embed_version the live embedders write (the default across match_or_create / the sidecars). A
# preprocessing change bumps this AND re-fits calibration under the new version — a coordinated
# migration, never a silent drift — so fusion only ever loads the curve for THIS generation.
_ACTIVE_EMBED_VERSION = 1
# /resolve over-fetches this multiple of resolve_k KNN rows before deduping to distinct candidates, so
# one person's several near clusters can't crowd a genuine runner-up out of the window (see /resolve).
_RESOLVE_FETCH_FACTOR = 5


def _fusion_config() -> fuse.FusionConfig:
    """Build the fusion policy from config so thresholds/cap are deployment-tunable. The weak-channel
    SET itself is not a knob — text being weak is a design invariant enforced in fuse.py."""
    return fuse.FusionConfig(
        text_llr_cap=float(config.get("fusion_text_llr_cap")),
        speculative_logodds=float(config.get("fusion_speculative_logodds")),
        confident_logodds=float(config.get("fusion_confident_logodds")),
        margin_logodds=float(config.get("fusion_margin_logodds")))


def _thresholds(embedder: str) -> dict:
    """Per-embedder accept policy. Each channel lives on its own cosine scale and its `quality` means
    something different (voice: utterance seconds; face: 0..1 crop score; text: 0..1 authored-text
    sufficiency), so they MUST NOT share thresholds — voice defaults would make another channel never
    learn (the bug this branch guards). All config-overridable. The text_style values are UNCALIBRATED
    placeholders until eval/text_benchmark.py fits them (design Phase 3/5); text is a weak vote and its
    thresholds are deliberately conservative until then."""
    if embedder == "face":
        return {"tau_match": float(config.get("face_tau_match")),
                "margin": float(config.get("face_margin")),
                "min_quality": float(config.get("face_min_quality"))}
    if embedder == "text_style":
        return {"tau_match": float(config.get("text_tau_match")),
                "margin": float(config.get("text_margin")),
                "min_quality": float(config.get("text_min_quality"))}
    return {"tau_match": float(config.get("tau_match")),
            "margin": float(config.get("margin")),
            "min_quality": float(config.get("min_quality"))}


class MatchRequest(BaseModel):
    embedding: list[float]
    session_id: Optional[str] = None
    quality: Optional[float] = None            # utterance duration (s); gates centroid learning
    raw_ref: Optional[str] = None
    regime: Optional[str] = None
    embedder: str = "campp"
    learn: bool = True


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    await db.get_pool()                        # warm the pool so the first match isn't cold
    await _reload_calibrations()               # preload fusion curves (empty until fitted)


async def _reload_calibrations() -> None:
    """(Re)load each channel's calibration curve for its ACTIVE model generation. Best-effort — a store
    hiccup leaves fusion abstaining (0 LLR) rather than crashing the service; call again to pick up a
    fresh fit. Loading by the channel's active (model_id, embed_version) — not the latest-fitted —
    means a curve fit on a different model/version is never applied to this generation's scores."""
    global _CALIBRATIONS
    try:
        active = {ch: (models.model_id(ch), _ACTIVE_EMBED_VERSION) for ch in _CHANNEL_MODALITY}
        _CALIBRATIONS = await calib_store.load_active_calibrations(active)
        _log.info("loaded %d calibration curve(s): %s",
                  len(_CALIBRATIONS), sorted(_CALIBRATIONS))
    except Exception:                            # noqa: BLE001 - fusion degrades to abstain, never 500s
        _log.warning("calibration preload failed; fusion channels will abstain", exc_info=True)
        _CALIBRATIONS = {}


@app.get("/health")
async def health() -> dict:
    # `calibrations` = the channels with a LOADED fusion curve (so a caller can tell recognition is
    # actually live, not just that the DB is reachable — an uncalibrated channel abstains in fusion).
    try:
        hc = await db.healthcheck()
        return {"ok": True, "calibrations": sorted(_CALIBRATIONS), **hc}
    except Exception as e:                       # noqa: BLE001 - health must never raise
        return {"ok": False, "calibrations": sorted(_CALIBRATIONS), "error": str(e)}


def _unknown(reason: str) -> dict:
    """A best-effort 'couldn't identify' result (HTTP 200) so the caller degrades to unknown
    rather than seeing a 500 — recognition must never break the host's main flow."""
    return {"action": "error", "cluster_id": None, "person_id": None, "person_name": None,
            "is_new": False, "sim": None, "top2_sim": None, "margin": None,
            "grounding_status": None, "learned": False, "error": reason}


@app.post("/match")
async def match(req: MatchRequest) -> dict:
    if req.embedder not in _VALID_EMBEDDERS:
        raise HTTPException(400, f"unknown embedder {req.embedder!r}")
    if req.embedder == "campp" and len(req.embedding) != CAMPP_DIM:
        raise HTTPException(400, f"campp embedding must be {CAMPP_DIM}-d, got {len(req.embedding)}")
    if req.embedder == "text_style" and len(req.embedding) != models.dim("text_style"):
        raise HTTPException(
            400, f"text_style embedding must be {models.dim('text_style')}-d, got {len(req.embedding)}")
    try:
        async with _match_lock:
            res = await cluster.match_or_create(
                req.embedder, req.embedding, session_id=req.session_id, quality=req.quality,
                raw_ref=req.raw_ref, regime=req.regime, learn=req.learn,
                modality=_EMBEDDER_MODALITY[req.embedder],
                **_thresholds(req.embedder))
    except Exception as e:                          # noqa: BLE001 - caller-facing: never leak a 500
        _log.exception("match failed")
        return _unknown(str(e)[:200])
    # Resolve a display name if the matched cluster is bound to a person (an enrolled/known voice).
    # NB: person_name None does NOT imply ungrounded — the person row could be unbound/deleted mid-flight
    # or have a NULL display_name; callers should key off person_id / grounding_status, not the name.
    person_name = None
    if res.get("person_id") is not None:
        try:
            pool = await db.get_pool()
            person_name = await pool.fetchval(
                "SELECT display_name FROM identity_person WHERE person_id = $1", res["person_id"])
        except Exception:                            # noqa: BLE001 - name is a nicety, not required
            person_name = None
    res["person_name"] = person_name
    # Recurrence gate for enrolment: flag an unknown cluster that's been heard enough to be worth
    # asking about. The host reads this to decide when to ask "who am I talking to / looking at?".
    member_count = None
    if res.get("cluster_id") is not None:
        try:
            pool = await db.get_pool()
            member_count = await pool.fetchval(
                "SELECT member_count FROM identity_cluster WHERE cluster_id = $1", res["cluster_id"])
        except Exception:                            # noqa: BLE001
            member_count = None
    res["member_count"] = member_count
    res["ask_identity"] = bool(
        res.get("person_id") is None and res.get("cluster_id") is not None
        and (member_count or 0) >= ASK_RECURRENCE)
    # Best-effort live viz (no-op unless NMEM_IDENTITY_VIZ_URL is set) — never blocks recognition.
    try:
        _payload = {
            "cluster_id": res.get("cluster_id"), "person_id": res.get("person_id"),
            "person_name": person_name, "grounding_status": res.get("grounding_status"),
            "sim": res.get("sim"), "margin": res.get("margin"),
            "member_count": member_count, "action": res.get("action"),
        }
        if res.get("action") == "created":
            viz.emit("identity.new_speaker", _payload)
        elif res.get("action") in ("matched", "matched_nolearn"):
            viz.emit("identity.match", _payload)
        if res.get("ask_identity"):
            viz.emit("identity.ask", _payload)
    except Exception:  # noqa: BLE001 - viz is best-effort
        pass
    return res


# ── Fusion / resolution (Phase 5) ────────────────────────────────────────────────────────────────

class ChannelInput(BaseModel):
    """One channel's embedding for a resolution turn. `length` is the retained authored-char count
    (text only) for length-banded calibration; `quality` gates learning on the primary channel."""
    embedding: list[float]
    length: Optional[int] = None
    quality: Optional[float] = None
    learn: bool = False


class ContextInput(BaseModel):
    """A non-embedding prior toward a candidate (a self-declared name, session continuity, an
    overlapping voice session). Supply either `candidate` (an explicit person:/cluster: key) or
    `name` (resolved here to the person). `strong` marks non-weak corroboration (an authenticated
    principal or concurrent voice-session match) — never set it for a merely claimed name."""
    kind: str
    candidate: Optional[str] = None
    name: Optional[str] = None
    log_odds: float = 0.0
    strong: bool = False

    @field_validator("log_odds")
    @classmethod
    def _finite_log_odds(cls, v: float) -> float:
        # Pydantic's bare float accepts "NaN"/"Infinity"; a non-finite prior would poison the fused
        # posterior and then fail JSON serialisation OUTSIDE the endpoint's try/except (a 500 the
        # caller can't degrade from). Reject it at the request boundary instead (422, fail-open client).
        if not math.isfinite(v):
            raise ValueError("log_odds must be finite")
        return v


class ResolveRequest(BaseModel):
    """Resolve WHO the interlocutor is by fusing whatever signals a turn produced. `text` is the
    primary chat-path channel (may learn its author cluster); `voice`/`face` are optional concurrent
    corroboration (read-only here — they learn via their own /match). `context` carries priors."""
    session_id: Optional[str] = None
    text: Optional[ChannelInput] = None
    voice: Optional[ChannelInput] = None
    face: Optional[ChannelInput] = None
    context: list[ContextInput] = []


def _cand_key(row: dict) -> str:
    """Stable fusion candidate key: an enrolled person (person:<id>) or an unbound cluster
    (cluster:<id>). A person key lets votes from different channels corroborate the same identity."""
    pid = row.get("person_id")
    if pid is not None:
        return f"person:{pid}"
    return f"cluster:{row['cluster_id']}"


async def _resolve_context_candidate(c: ContextInput) -> str:
    """The candidate key a context signal refers to. An explicit `candidate` wins; otherwise a `name`
    is resolved to an enrolled person (person:<id>) or, if no such person exists yet, kept as
    name:<name> — a candidate carrying only prior evidence (a just-declared, not-yet-enrolled name),
    which can reach speculative but never authorises without a real channel."""
    if c.candidate:
        # A cluster key that has since been bound to a person must normalise to that person key — else
        # a caller echoing a cluster:<id> after /enrol splits from the embedding hit (which _cand_key
        # now reports as person:<id>), preventing corroboration (codex). person:/name: pass through.
        if c.candidate.startswith("cluster:"):
            try:
                cid = int(c.candidate.split(":", 1)[1])
                pool = await db.get_pool()
                pid = await pool.fetchval(
                    "SELECT person_id FROM identity_cluster WHERE cluster_id = $1", cid)
                if pid is not None:
                    return f"person:{pid}"
            except Exception:                            # noqa: BLE001 - best-effort; keep the raw key
                pass
        return c.candidate
    name = (c.name or "").strip()
    if not name:
        return "name:"
    try:
        pool = await db.get_pool()
        pid = await pool.fetchval(
            "SELECT person_id FROM identity_person WHERE display_name = $1 "
            "ORDER BY person_id LIMIT 1", name)
        if pid is not None:
            return f"person:{pid}"
    except Exception:                                # noqa: BLE001 - name lookup is best-effort
        pass
    return f"name:{name}"


# Which fusion contributor maps to which Speaker `source` label (agent_core.speaker.SPEAKER_SOURCES).
# Both biometric channels are concurrent-session evidence → voice_session; text stays text_style.
_CHANNEL_SOURCE = {"text_style": "text_style", "campp": "voice_session", "face": "voice_session"}


def _source_for(candidate: str, result, signals: list) -> str:
    """The Speaker `source` label for the winning candidate — the label of its largest-magnitude
    contributor, compared on ONE scale (calibrated nats). Uses the fusion result's per-channel LLRs
    (which already exclude abstaining/uncalibrated channels and reflect the weak-vote cap) rather than
    raw cosines, so a channel that contributed no evidence is never named as the source. Falls back to
    'unknown' when nothing contributed to the winner."""
    best_src, best_mag = "unknown", -1.0
    prefix = f"{candidate}:"
    for key, llr in result.per_channel_llr.items():
        if key.startswith(prefix):
            ch = key[len(prefix):]
            if ch in _CHANNEL_SOURCE and abs(llr) > best_mag:
                best_src, best_mag = _CHANNEL_SOURCE[ch], abs(llr)
    for s in signals:
        if s.candidate == candidate and abs(s.log_odds) > best_mag:
            best_src, best_mag = s.kind, abs(s.log_odds)
    return best_src


@app.post("/resolve")
async def resolve(req: ResolveRequest) -> dict:
    """Fuse a turn's identity signals into one calibrated decision (design: chat-style modality,
    Phase 5). Read-mostly: the text channel MAY learn its author cluster (req.text.learn); voice/face
    are read-only corroboration. Returns a Speaker-shaped payload plus the fusion provenance. Never
    500s — recognition must never break the host's turn (best-effort, degrades to unknown)."""
    # Validate EVERY supplied embedding's dimension up front — before the text channel's
    # match_or_create, which WRITES (inserts a template + moves the centroid). Deferring the voice/face
    # dim checks until after that write meant a request rejected for a bad voice/face vector had already
    # mutated the text cluster, and the corrected retry would learn the same observation twice and
    # inflate recurrence (codex). All-or-nothing: a malformed request learns nothing.
    for ch, payload in (("text_style", req.text), ("campp", req.voice), ("face", req.face)):
        if payload is not None and len(payload.embedding) != models.dim(ch):
            raise HTTPException(400, f"{ch} embedding must be {models.dim(ch)}-d")
    try:
        votes: list[fuse.ChannelVote] = []
        meta: dict[str, dict] = {}

        # ── text: the primary chat-path channel. Route through match_or_create so the author cluster
        #    is maintained (learn side-effect + quality gate). Only a genuine MATCH is identity
        #    evidence; a freshly CREATED / unknown cluster means "new author" → no positive text vote. ──
        if req.text is not None:
            async with _match_lock:
                tres = await cluster.match_or_create(
                    "text_style", req.text.embedding, session_id=req.session_id,
                    quality=req.text.quality, learn=req.text.learn,
                    modality=_EMBEDDER_MODALITY["text_style"], **_thresholds("text_style"))
            if tres.get("action") in ("matched", "matched_nolearn") and tres.get("cluster_id"):
                key = _cand_key(tres)
                meta[key] = {"cluster_id": tres["cluster_id"], "person_id": tres.get("person_id"),
                             "grounding_status": tres.get("grounding_status")}
                votes.append(fuse.ChannelVote("text_style", key, float(tres["sim"]),
                                              length=req.text.length))

        # ── voice / face: optional concurrent corroboration (read-only KNN). ──
        k = int(config.get("resolve_k"))
        # Over-fetch, then dedupe to DISTINCT candidates: one person can own several nearby clusters
        # (over-split / multi-enrol), and if those fill the window a genuine runner-up from ANOTHER
        # person just outside it is never seen — leaving one candidate with an infinite margin that can
        # wrongly authorize (codex). Fetching k distinct identities out of a wider row window keeps a
        # near competitor visible so the margin guard can do its job.
        fetch = max(k * _RESOLVE_FETCH_FACTOR, k + 16)
        for ch, payload in (("campp", req.voice), ("face", req.face)):
            if payload is None:
                continue                             # dims already validated up front (all-or-nothing)
            try:
                hits = await T.knn_clusters(ch, payload.embedding, k=fetch,
                                            modality=_CHANNEL_MODALITY[ch])
            except Exception:                        # noqa: BLE001 - a channel KNN failure just abstains
                _log.warning("resolve: %s KNN failed (abstaining)", ch, exc_info=True)
                continue
            # Keep ONE vote per candidate per channel — the best-similarity hit. Two centroids enrolled
            # to the same person (/enrol supports it) collapse to one key via _cand_key; emitting a vote
            # for each would let fusion SUM the same channel's evidence and over-count toward authorize.
            best: dict[str, dict] = {}
            for h in hits:
                key = _cand_key(h)
                if key not in best or float(h["sim"]) > float(best[key]["sim"]):
                    best[key] = h
            # Keep the k best-similarity DISTINCT candidates (a near runner-up is retained for margin).
            top = sorted(best.values(), key=lambda h: float(h["sim"]), reverse=True)[:k]
            for h in top:
                key = _cand_key(h)
                meta.setdefault(key, {"cluster_id": h["cluster_id"], "person_id": h.get("person_id"),
                                      "grounding_status": h.get("grounding_status")})
                votes.append(fuse.ChannelVote(ch, key, float(h["sim"])))

        # ── context priors ──
        signals: list[fuse.ContextSignal] = []
        for c in req.context:
            key = await _resolve_context_candidate(c)
            signals.append(fuse.ContextSignal(c.kind, key, float(c.log_odds), bool(c.strong)))
            meta.setdefault(key, {})

        result = fuse.fuse(votes, signals, _CALIBRATIONS, config=_fusion_config())
    except HTTPException:
        raise
    except Exception as e:                           # noqa: BLE001 - caller-facing: never leak a 500
        _log.exception("resolve failed")
        return {"action": "error", "candidate": None, "person_id": None, "person_name": None,
                "grounding": None, "authorize_ok": False, "ask_identity": False,
                "speaker": None, "error": str(e)[:200]}

    return await _resolve_response(result, votes, signals, meta)


async def _resolve_response(result, votes: list, signals: list, meta: dict) -> dict:
    """Shape a FusionResult into the /resolve payload: resolve the winner to a person + display name,
    decide whether to ask who it is, and emit a Speaker the chat path can carry."""
    cand = result.candidate
    info = meta.get(cand, {}) if cand else {}
    person_id = info.get("person_id")
    # A winning candidate that IS a bare person key (person:<id>) but wasn't in a KNN row still resolves.
    if person_id is None and cand and cand.startswith("person:"):
        try:
            person_id = int(cand.split(":", 1)[1])
        except ValueError:
            person_id = None
    person_name = None
    if person_id is not None:
        try:
            pool = await db.get_pool()
            person_name = await pool.fetchval(
                "SELECT display_name FROM identity_person WHERE person_id = $1", person_id)
        except Exception:                            # noqa: BLE001
            person_name = None
    # A name-only candidate (declared, not enrolled) carries its name in the key.
    if person_name is None and cand and cand.startswith("name:"):
        person_name = cand.split(":", 1)[1] or None

    # ask_identity: worth asking WHO only for an unbound, speculative-but-not-confident belief that
    # has recurred enough — i.e. we keep meeting an author we can't name. A confident/authorised
    # identity needs no question; an ungrounded blip isn't worth interrupting for.
    member_count = None
    cid = info.get("cluster_id")
    if cid is not None:
        try:
            pool = await db.get_pool()
            member_count = await pool.fetchval(
                "SELECT member_count FROM identity_cluster WHERE cluster_id = $1", cid)
        except Exception:                            # noqa: BLE001
            member_count = None
    ask_identity = bool(
        person_id is None and cid is not None and result.grounding == "speculative"
        and (member_count or 0) >= ASK_RECURRENCE)

    source = _source_for(cand, result, signals) if cand else "unknown"
    # The Speaker the chat path carries. `id` is the person name when known, a human-readable declared
    # name for a name:<x> candidate, else the CANONICAL candidate key (cluster:<id> / person:<id>) — so
    # the caller can echo it back verbatim as a context `candidate` next turn and text + continuity
    # evidence converge on ONE identity instead of splitting (codex). confidence is the fused posterior
    # (already reflecting the weak-vote cap); authorize_ok / grounding gate anything consequential.
    speaker = None
    if cand and result.grounding != "ungrounded":
        if person_name:
            sid = person_name
        elif cand.startswith("name:"):
            sid = cand.split(":", 1)[1] or cand
        else:
            sid = cand                               # canonical cluster:/person: key, echoed verbatim
        speaker = {"id": sid, "confidence": round(result.posterior, 4), "source": source}

    payload = {
        "action": "resolved" if cand else "unknown",
        "candidate": cand, "person_id": person_id, "person_name": person_name,
        "grounding": result.grounding, "authorize_ok": result.authorize_ok,
        "posterior": round(result.posterior, 4),
        "log_odds": (None if not math.isfinite(result.log_odds) else round(result.log_odds, 4)),
        "margin": (None if not math.isfinite(result.margin) else round(result.margin, 4)),
        "member_count": member_count, "ask_identity": ask_identity,
        "source": source, "speaker": speaker,
        "per_channel_llr": result.per_channel_llr,
    }
    try:
        if cand:
            viz.emit("identity.resolve", {
                "candidate": cand, "person_id": person_id, "person_name": person_name,
                "grounding": result.grounding, "posterior": result.posterior,
                "authorize_ok": result.authorize_ok, "ask_identity": ask_identity})
    except Exception:                                # noqa: BLE001 - viz is best-effort
        pass
    return payload


class EnrolRequest(BaseModel):
    cluster_id: int
    name: str


@app.post("/enrol")
async def enrol(req: EnrolRequest) -> dict:
    """Bind a voice cluster to a named person (conversational enrolment write-back). Get-or-create the
    person by display_name, bind the cluster as 'confident', and snapshot its centroid as the
    rechallenge baseline. Idempotent: enrolling the same cluster+name again just re-binds."""
    name = (req.name or "").strip()[:60]
    if not name:
        raise HTTPException(400, "name required")
    try:
        pool = await db.get_pool()
        async with _match_lock:                      # serialise vs match's read-then-write
            pid = await pool.fetchval(
                "SELECT person_id FROM identity_person WHERE display_name = $1 "
                "ORDER BY person_id LIMIT 1", name)
            if pid is None:
                pid = await T.create_person(
                    name, nmem_entity_id="person:" + name.lower().replace(" ", "_"))
            await T.bind_cluster_person(req.cluster_id, pid, grounding_status="confident")
            # Snapshot the modality-appropriate centroid (voice→CAM++, face→ArcFace) as the
            # rechallenge drift baseline — works for a face cluster exactly as for a voice one.
            await T.snapshot_centroid(req.cluster_id)
        _log.info("enrolled cluster %s -> person %s (%s)", req.cluster_id, pid, name)
        viz.emit("identity.enrolled", {
            "cluster_id": req.cluster_id, "person_id": pid, "person_name": name,
            "grounding_status": "confident",
        })
        return {"ok": True, "person_id": pid, "name": name, "cluster_id": req.cluster_id}
    except Exception as e:                            # noqa: BLE001 - never 500 the caller
        _log.exception("enrol failed")
        return {"ok": False, "error": str(e)[:200]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
