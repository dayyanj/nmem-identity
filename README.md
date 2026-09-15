# nmem-identity

**Status: 0.2 — early, pre-1.0.** The store + matcher run wired into a live agent (Phase 0, voice); the API may still change. See the [changelog](CHANGELOG.md).

Self-supervised **multi-modal person identity** (voice + face) as an extension of
[nmem](https://github.com/dayyanj/nmem). A host agent learns *who it is interacting with* — recognising people by
voice and/or face over time — with **no manual tagging**: identities accumulate from evidence, and the
system asks a human *once*, only when it is already confident it keeps meeting the same person.

The package is **install- and agent-agnostic**. It names no particular host, database, or machine;
everything host-specific is configuration (`config.py` / `NMEM_IDENTITY_*` env / an optional
`nmem-identity.toml`). Any agent that can produce an embedding and make an HTTP call can use it.

**Design / blueprint:** `nmem/docs/design/multi-modal-identity.md` (the *why*) and
`-plan.md` (the *how* and *in what order*).

## Thesis

> Perception produces weak, uncertain signals; identity is resolved by memory and context, not by any
> single classifier.

Fusing face + voice + contextual priors under nmem's grounding levels should beat any one channel and
degrade gracefully when a channel is missing — a claim the phased build is designed to *falsify* early
if it is false for a given deployment.

## Architecture

```
host agent ──embed──▶ matcher service ──▶ identity DB (identity_* tables in pgvector)
   (voice/face)          /match, /enrol        clusters · templates · centroids · calibration · priors
```

- **`store/`** — the identity DB. `identity_cluster` / `identity_template` carry **per-embedder
  self-describing triples** `(vector, model_id, embed_version)`: CAM++ voice (192-d), ArcFace face
  (512-d), each in its own columns with its own model provenance. Cross-model cosine is structurally
  impossible (every KNN filters on the matching `*_model_id`, hitting a partial HNSW index). A model
  swap is a re-embedding migration, never a silent comparison. Deletability is built in:
  `delete_person(person_id)` cascades templates, returns raw-blob keys to unlink, leaves zero orphans.
- **`cluster.py`** — the live recognition core: incremental nearest-centroid `match_or_create` with
  margin + quality guards. **Modality-agnostic** — `match_or_create(embedder, vec, modality=...)`
  works identically for voice and face.
- **`perception/`** — embedders that turn raw signal into vectors: `face.py` (SCRFD detect → 5-pt
  landmark align → ArcFace → 512-d), `quality.py` (per-observation quality gates). Voice embedding is
  produced by an external speaker-embedding service behind the same `/embed` contract.
- **`services/`** — HTTP surface: `matcher_server.py` (`/match`, `/enrol`) and the embed sidecars.
- **`eval/`** — regime-stratified benchmarks and public-benchmark validation (LFW/IJB-C verification,
  EER, the alignment ablation).

### Host integration contract

A host stays thin — it embeds and calls in over HTTP; it never opens a DB pool itself:

1. On each observation, compute an embedding (voice or face) and
   `POST /match {embedding, embedder, session_id?, quality?}` → `{person_id, person_name,
   grounding_status, cluster_id, member_count, ask_identity, ...}`.
2. When `ask_identity` is true (an unbound cluster seen enough times — the *system* decides when to
   ask, it isn't left to the LLM), the host asks the human "who is this?" **once**.
3. On the answer, `POST /enrol {cluster_id, name}` binds the cluster to a named person and snapshots
   its centroid as the rechallenge baseline.

## Status

**Voice channel — built & live.**
- **CAM++ speaker embedder** (192-d) behind an `/embed` HTTP contract; validated discriminative.
- **Store** — schema applied; `db.py` + `templates.py` (person/cluster/template CRUD, model-scoped
  `knn_clusters`, EMA + batch centroid rebuild, zero-orphan `delete_person`). Smoke-tested end-to-end.
- **Matcher service** — `/match` (recognise or create) + `/enrol` (conversational write-back) with a
  recurrence gate (`ask_identity`).
- **Offline `ingest.py`** — labelled captures → populated store.
- **`eval/benchmark.py`** — regime-stratified Gate-0 report (per-regime EER + leave-one-session-out
  nearest-centroid accuracy + a verdict).

**Face channel — in progress (public-benchmark-first).** The face triple is present in the store from
day one; the perception pipeline and its public-benchmark validation are the current build:
- `perception/face.py` — SCRFD + 5-pt landmark align + ArcFace → 512-d. **Lightweight by default**
  (`buffalo_s`: SCRFD-500MF + MobileFaceNet) so the face channel can share one GPU with the rest of a
  host's stack (LLM, TTS); the heavy `buffalo_l` pack is kept for an offline accuracy-ceiling compare.
- `perception/quality.py` — face size / pose / blur gates.
- `services/face_embed_server.py` — `/embed` sidecar (image → 512-d + quality).
- `eval/face_benchmark.py` — LFW/IJB-C verification EER **and the alignment ablation** (landmark align
  vs naive box-crop — proving the ~2× gap that makes alignment non-negotiable). This runs against
  public data and needs no captured household data, so it validates the highest-risk component first.

## Footprint (designed to share one GPU)

The face channel is built to co-exist with a full host stack (LLM + TTS + ...) on a **single GPU**, so
the defaults favour low latency and small footprint over the last fraction of a percent of accuracy:

- **Light models by default** — `buffalo_s` (SCRFD-500MF detector + MobileFaceNet recogniser), a
  fraction of `buffalo_l`'s FLOPs/VRAM, still 512-d and ~99.5% LFW.
- **Latency levers**, all config-driven: `face_det_size` (640 accurate / 320 ~4× cheaper),
  `NMEM_IDENTITY_ORT_PROVIDERS` (set CPU to keep the GPU entirely for the LLM/TTS — the models are
  small enough to run real-time on CPU for single-subject capture), and host-side frame gating (run
  detection at ~2 FPS, not every frame — recognition needs no frame-rate throughput).
- **Swapping the pack is a migration, not a flag.** The pack fixes detector + recogniser + alignment,
  so it defines cosine-comparability; changing it means a new `model_id`, a new partial index, and
  re-embedding retained crops. The embed sidecar fails fast if the running pack ≠ the store's active
  face `model_id`.

## Install & configure

```bash
pip install -e '.[serve,face,eval]'          # core is dep-light; extras add services/face/eval
export NMEM_IDENTITY_DSN=postgresql://user:pass@host:5432/dbname   # asyncpg scheme (no +asyncpg)
python -c "import asyncio; from nmem_identity.store import db; asyncio.run(db.apply_schema())"
```

All knobs (DSN, ports, thresholds, blob dir) resolve as **defaults < `nmem-identity.toml` < env** —
see `config.py`.

## Phases & gates

Each phase ends with a go/no-go on real data; the ordering front-loads the cheapest experiment that
can kill the project.

- **Phase 0 (voice) — done.** CAM++ + gallery NN + conversational enrolment; capture → Gate 0 decides
  whether the light path is enough or the heavier apparatus is warranted.
- **Phase 0a (face) — current.** `perception/face.py` + **public-benchmark validation + alignment
  ablation**. Unblocked by hardware; de-risks the face channel before any camera exists.
- **Phase 1** — calibrated per-channel LLR + capped-log-prior **fusion**; adds the second live channel.
- **Phase 2** — enrolment loop + cross-session anchoring (persistent centroid) + rechallenge (drift /
  decay / confident→disputed).
- **Phase 3** — nmem dossier + per-channel provenance + temporal/spatial contradiction detection.

## The nmem suite

nmem-identity is one library in a family of composable, framework-agnostic cognitive layers for AI agents. Each is standalone — mix in only the ones you need.

| Repo | Layer | What it does |
|------|-------|--------------|
| [nmem](https://github.com/dayyanj/nmem) | Memory | Hierarchical, self-refining cognitive memory — 6 tiers + a consolidation engine |
| [nmem-sym](https://github.com/dayyanj/nmem-sym) | Reasoning | Symbolic cognition — typed knowledge graph, spreading activation, drives, prediction |
| [nmem-sym-sensor](https://github.com/dayyanj/nmem-sym-sensor) | Perception | Sensory memory — unlabeled visual/audio primitives clustered into grounded concepts |
| [nmem-identity](https://github.com/dayyanj/nmem-identity) | Perception | Self-supervised person identity (voice + face) — learns who it's talking to, no manual tagging |
| [nmem-act](https://github.com/dayyanj/nmem-act) | Action | Typed action/outcome contract + tiered autonomy gate — act to learn, safely |
| [nmem-sandbox](https://github.com/dayyanj/nmem-sandbox) | Action | LLM-agnostic computer-use sandbox — a headless desktop a vision model drives |
| [nmem-exchange](https://github.com/dayyanj/nmem-exchange) | Comms | End-to-end-secured message bus — agents talk without sharing memory |
| [nmem-immune](https://github.com/dayyanj/nmem-immune) | Integrity | Memory immune system — poisoning detection, drift, quarantine |
| [nmem-viz](https://github.com/dayyanj/nmem-viz) | Tooling | Real-time 3D "brain" visualization of any nmem agent |

**Just want to run one?** [**nmem-studio**](https://huggingface.co/dayyanj/nmem-studio) is the pull-and-run appliance — a single Docker image that stands up one fully-configured agent from a web wizard (memory + reasoning, plus optional identity and embodied perception), no config files by hand.
