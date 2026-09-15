"""Canonical embedder registry — the one authoritative map of embedder name → (model_id, dim).

Dependency-free on purpose so BOTH the store layer (store/templates.py) and the perception layer
(perception/face.py) import the same constants without either depending on the other, and without
pulling ML deps.

model_id is **schema-bound**: a partial HNSW index is defined with the model_id as a literal in the
DDL (store/schema.sql), and every KNN predicate inlines the *same* literal. So a model swap is a
migration (register a new id, re-embed, re-index), never a config flip — that is the versioning
guarantee the store enforces. `embed_version` (stored per-vector, not here) bumps on any preprocessing
/ alignment change even when the weights are unchanged, because alignment moves cosine distances as
much as weights do.

A `model_id` of None means the embedder is defined but not yet enabled (no vectors may be written for
it and it cannot be queried).
"""
from __future__ import annotations

# Face model packs → (recognition) model_id. A pack fixes the detector AND the recogniser AND the
# alignment, so it is the unit of comparability: two packs are never cosine-comparable. Switching the
# live pack is therefore a model-swap migration (new model_id, new partial index, re-embed), never a
# silent config flip. The DEFAULT is deliberately the LIGHT pack — the face channel is designed to
# share one GPU with the rest of a host's stack (LLM, TTS, ...), so low latency / small footprint wins
# over the last fraction of a percent of accuracy.
#
#   buffalo_s : SCRFD-500MF detector + MobileFaceNet recogniser — light, low-latency. DEFAULT / live.
#   buffalo_l : SCRFD-10G detector + ResNet50 recogniser — accuracy ceiling; heavy. OFFLINE compare only.
FACE_PACK_MODEL_ID: dict[str, str] = {
    "buffalo_s": "arcface_buffalo_s",
    "buffalo_l": "arcface_buffalo_l",
}
DEFAULT_FACE_PACK = "buffalo_s"

EMBEDDERS: dict[str, dict[str, object]] = {
    # CAM++ speaker embedding (ONNX). The live voice channel.
    "campp": {"model_id": "campplus_onnx_v1", "dim": 192},
    # ArcFace face embedding. model_id tracks the live pack (light by default); a swap is a migration.
    "face": {"model_id": FACE_PACK_MODEL_ID[DEFAULT_FACE_PACK], "dim": 512},
    # LUAR authorship embedding — the chat-writing-STYLE channel (design: nmem-identity-chat-style-
    # modality). A LUAR-family author encoder maps a person's short texts (an "episode" of turns) to a
    # 512-d authorship vector. This is the WEAKEST identity modality by design — a soft prior that can
    # raise confidence but must never authorize on its own; fusion (Phase 5) weights it below voice/
    # face. 512-d like face, but a SEPARATE column + model_id, so the two are never cosine-comparable.
    # model_id 'luar_v1' is schema-bound (partial-index literal); a weights/preprocessing change is a
    # re-embed migration, never a config flip. The HF checkpoint name is a runtime knob (config.py).
    "text_style": {"model_id": "luar_v1", "dim": 512},
}


def model_id(embedder: str) -> str | None:
    return EMBEDDERS[embedder]["model_id"]  # type: ignore[return-value]


def dim(embedder: str) -> int:
    return int(EMBEDDERS[embedder]["dim"])  # type: ignore[arg-type]
