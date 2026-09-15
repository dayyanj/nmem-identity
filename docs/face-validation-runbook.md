# Face channel — Phase 0a validation runbook (Gate 0a)

The face perception pipeline (`perception/face.py`), the embed sidecar
(`services/face_embed_server.py`), and the benchmark harness (`eval/face_benchmark.py`) are built and
unit-checked. What remains is a **runtime validation** that must happen on a host with the ML deps, a
GPU (or CPU + patience), and the downloaded public datasets — none of which live on a dev box.

This is intentionally the first face deliverable: it validates the highest-risk component against
public data, so any accuracy loss is a bug in *our* detection/alignment rather than something masked
by real-world data. It needs **no camera and no captured household data.**

## 1. Environment

```bash
pip install -e '.[face,eval,serve]'
# On a CUDA host, prefer the GPU ONNX runtime:
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

First construction of `FaceEmbedder()` auto-downloads the configured pack to `~/.insightface/models/`.
The **live default is the light pack `buffalo_s`** (SCRFD-500MF detector + MobileFaceNet recogniser) —
chosen so the face channel shares a single GPU with the rest of the host stack (LLM, TTS). Pre-warm it:

```bash
python -c "from nmem_identity.perception.face import FaceEmbedder; FaceEmbedder(); print('ok')"
```

**Shared-GPU knobs** (this is the whole point on a single-GPU host):
- `export NMEM_IDENTITY_ORT_PROVIDERS=CPUExecutionProvider` — run the (small) face models on CPU and
  leave the GPU entirely to the LLM/TTS. `CUDAExecutionProvider,CPUExecutionProvider` to use the GPU.
- `export NMEM_IDENTITY_FACE_DET_SIZE=320` — ~4× cheaper detection than 640; fine for close-range
  single-subject capture. The dominant per-frame cost is detection, so this is the main latency lever.
- Gate detection to ~2 FPS host-side; recognition needs no frame-rate throughput.

## 2. Datasets

- **LFW** (the tractable first target): download `lfw-deepfunneled` (or `lfw`) + the standard
  `pairs.txt` (6000 pairs). Point `--root` at the directory of per-person folders.
- **IJB-C** (harder, template/mixed-media; optional follow-up): use the `generic` loader with a
  pairs JSON (`[["a.jpg","b.jpg",true], ...]`) built from the protocol CSVs.

## 3. Run — reproduce the numbers + the alignment ablation

```bash
# validate the LIVE light pack (default) + the alignment ablation
nmem-identity face-benchmark --dataset lfw \
    --root /data/lfw/lfw-deepfunneled --pairs /data/lfw/pairs.txt \
    --align landmark,boxcrop

# optional: measure what the heavy pack would buy (accuracy ceiling, offline only)
nmem-identity face-benchmark --dataset lfw --root /data/lfw/lfw-deepfunneled \
    --pairs /data/lfw/pairs.txt --pack buffalo_l --align landmark
```

Read the report:

- **Gate 0a (reproduce known numbers).** For the light pack, `landmark` accuracy should be near the
  published `buffalo_s` LFW number (~0.995; `buffalo_l` ~0.998). If it is ~1%+ below, the bug is in
  *this* detection/alignment — fix it here before anything downstream. The harness prints this verdict
  against the pack's reference.
- **Light-vs-heavy tradeoff.** Compare the two runs: if `buffalo_s` is within a small margin of
  `buffalo_l` on your data, the light pack is the right live choice (it already is the default). Only
  if the gap is large *and* matters for the deployment should you migrate the live pack to a heavier
  one (a migration, per §4).
- **Alignment ablation.** `boxcrop` EER should be **well above** `landmark` EER (the design expects
  roughly a 2× gap). Confirming it is the evidence that makes 5-point-landmark alignment
  non-negotiable. A *small* gap means alignment isn't wired the way you think — investigate before
  trusting the pipeline.
- **`no_face` count.** Detection failures on a cooperative benchmark like LFW should be near zero; a
  high count points at a detector-size/threshold problem.

## 4. Wire the live path (once Gate 0a is green)

The store, matcher, and enrolment loop are already modality-agnostic and the face channel is enabled
(`face` embedder → `arcface_buffalo_s`, partial HNSW index on `centroid_face`). To run face
recognition live:

```bash
# 1. embed sidecar (warm model)
NMEM_IDENTITY_FACE_EMBED_PORT=9405 python services/face_embed_server.py
# 2. matcher (already running for voice) accepts embedder="face":
#    host POSTs the sidecar's 512-d vector to /match {embedder:"face", quality, session_id}
```

A host with a camera: frame → `POST /embed` (face sidecar) → `POST /match {embedder:"face"}` →
`{person_id, grounding_status, ask_identity}`; on `ask_identity`, ask once → `POST /enrol`. This is the
same recurrence→ask→enrol loop the voice channel uses; nothing about it is voice-specific.

## 5. What this does NOT cover (later phases, gated on real data)

- **Household capture + regime distribution (Phase 0b).** The public benchmark proves the pipeline is
  correct; it does not tell you how often *your* deployment is in a degraded regime — that is the
  kill-gate that decides whether fusion is worth building. Needs a camera + a labelling pass.
- **Calibrated LLR fusion of face + voice (Phase 1).** Only warranted if degraded regimes are
  materially frequent.
- **Cross-session re-ID, rechallenge, dossier (Phases 2–3).**
