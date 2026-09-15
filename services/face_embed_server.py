"""Face-embedding sidecar — image → 512-d ArcFace vector, symmetric with the voice `/embed` contract.

A host with a camera POSTs a frame (encoded image bytes) here; the service detects the largest face,
5-point-landmark-aligns it, embeds it with ArcFace, scores quality, and returns the vector the matcher
wants. Keeping the model in a warm sidecar (rather than in the host process) matches the voice channel:
the host stays thin and just makes an HTTP call.

    POST /embed   (raw image bytes: jpeg/png/... in the body; optional ?align=landmark|boxcrop)
        -> { embedding:[512 floats], dim, model_id, embed_version, det_score, bbox,
             quality, quality_parts, align, n_faces }        # n_faces:0 => no face detected
    GET  /health

The 512-d embedding is L2-normalised (cosine == dot). `align=boxcrop` exists for parity with the
benchmark's alignment ablation; the default and only production path is `landmark`.

Run with the `face` + `serve` extras installed (insightface, onnxruntime, opencv, fastapi, uvicorn).
Configure the port with NMEM_IDENTITY_FACE_EMBED_PORT and the ONNX providers with NMEM_IDENTITY_ORT_PROVIDERS
(comma-separated, e.g. "CUDAExecutionProvider,CPUExecutionProvider"):

    NMEM_IDENTITY_FACE_EMBED_PORT=9405 python services/face_embed_server.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmem_identity import config, models                 # noqa: E402
from nmem_identity.perception import face as face_mod    # noqa: E402
from nmem_identity.perception import quality as quality_mod  # noqa: E402

PORT = int(config.get("face_embed_port"))
_log = logging.getLogger("face_embed")

app = FastAPI()
_embedder: face_mod.FaceEmbedder | None = None


def _providers() -> list[str] | None:
    raw = os.environ.get("NMEM_IDENTITY_ORT_PROVIDERS")
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else None


@app.on_event("startup")
async def _startup() -> None:
    global _embedder
    # Load once, warm. A failure here (missing extra / weights) should surface immediately, not per
    # request — so we construct eagerly and let /health report readiness.
    pack = str(config.get("face_model_pack"))
    n = int(config.get("face_det_size"))
    _embedder = face_mod.FaceEmbedder(model_pack=pack, providers=_providers(), det_size=(n, n))
    # Fail fast on a pack↔store desync: vectors this sidecar produces are stored under the face
    # model_id, so the running pack MUST match the store's active face model_id (else KNN silently
    # never matches, or worse, mixes generations). A pack swap is a migration, not a config typo.
    if _embedder.model_id != models.model_id("face"):
        raise RuntimeError(
            f"face pack {pack!r} → model_id {_embedder.model_id!r} but the store's active face "
            f"model_id is {models.model_id('face')!r}. Swapping the live pack is a migration: update "
            f"the store model_id + partial index and re-embed, don't just change NMEM_IDENTITY_FACE_MODEL_PACK.")
    _log.info("face embedder ready (pack=%s model_id=%s det_size=%d)", pack, _embedder.model_id, n)


@app.get("/health")
async def health() -> dict:
    return {"ok": _embedder is not None,
            "model_id": (_embedder.model_id if _embedder else models.model_id("face")),
            "pack": (_embedder.model_pack if _embedder else config.get("face_model_pack")),
            "dim": face_mod.EMBED_DIM, "providers": _providers()}


def _decode(body: bytes) -> np.ndarray:
    import cv2  # type: ignore
    arr = np.frombuffer(body, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)            # -> BGR, the format ArcFace/SCRFD expect
    if img is None:
        raise HTTPException(400, "could not decode image bytes")
    return img


@app.post("/embed")
async def embed(request: Request) -> dict:
    if _embedder is None:                                # pragma: no cover - startup failed
        raise HTTPException(503, "embedder not ready")
    align = request.query_params.get("align", "landmark")
    if align not in ("landmark", "boxcrop"):
        raise HTTPException(400, f"unknown align {align!r}")
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty image body")
    img = _decode(body)
    # Blocking model call off the event loop, mirroring the voice service's non-blocking embed.
    obs = await asyncio.to_thread(
        _embedder.observe, img, align=align, largest_only=True, score_quality=True)
    if not obs:
        return {"n_faces": 0, "model_id": _embedder.model_id, "dim": face_mod.EMBED_DIM,
                "regime": "no_face"}
    o = obs[0]
    return {
        "embedding": o.embedding.tolist(),
        "dim": face_mod.EMBED_DIM,
        "model_id": _embedder.model_id,
        "embed_version": o.embed_version,
        "det_score": o.det_score,
        "bbox": o.bbox.tolist(),
        "quality": o.quality,
        "quality_parts": o.quality_parts,
        "regime": quality_mod.regime(o.quality_parts),
        "align": o.align,
        "n_faces": len(obs),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
