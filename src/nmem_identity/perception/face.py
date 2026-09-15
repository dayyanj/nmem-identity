"""Face perception — SCRFD detect + 5-point landmark align (112×112) + ArcFace → 512-d.

This is the highest-risk component of the visual channel, so it is built to be validated against
public benchmarks (LFW / IJB-C) BEFORE any captured data exists — if it doesn't reproduce known
numbers, the bug is in *this* detection/alignment, not hidden by real-world data.

**Lightweight by default.** The face channel is designed to share one GPU with the rest of a host's
stack (LLM, TTS, ...), so the default pack is `buffalo_s` (SCRFD-500MF detector + MobileFaceNet
recogniser) — a fraction of the FLOPs/VRAM of `buffalo_l`, still 512-d, ~99.5% LFW. Latency levers:
`det_size` (640 accurate / 320 ~4× cheaper), the ONNX provider (CPU keeps the GPU free for the LLM),
and host-side frame gating (run detection at ~2 FPS, not every frame). The heavier `buffalo_l` pack is
available for an OFFLINE accuracy-ceiling comparison in the benchmark, never as the live default.

The one non-negotiable lesson baked in here: **feed ArcFace a 5-point-landmark-aligned 112×112 crop,
never a raw detector box.** Naive box-crop roughly halves verification accuracy. To *prove* that for a
given deployment rather than assert it, alignment is a parameter (`align="landmark"` vs `"boxcrop"`)
so `eval/face_benchmark.py` can run the ablation on the same code path the live system uses.

Heavy ML deps (insightface / onnxruntime / opencv) are **lazy-imported inside the class**, so
`import nmem_identity.perception.face` succeeds in a store-only install; the deps are required only
when you actually construct a `FaceEmbedder`.

Model provenance: this produces vectors for the store's `face` embedder — `model_id` and dimension
come from `nmem_identity.models` so the value that lands in the DB is the same literal the partial
HNSW index is built on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .. import models

FACE_MODEL_ID: str = models.model_id("face")          # "arcface_buffalo_l"
EMBED_DIM: int = models.dim("face")                    # 512
ARCFACE_INPUT: int = 112                                # ArcFace expects a 112×112 aligned crop

AlignMethod = str  # "landmark" | "boxcrop"


@dataclass
class FaceObservation:
    """One detected+embedded face. `embedding` is L2-normalised (so cosine == dot), matching how the
    store compares centroids. `quality` is filled by perception.quality.face_quality (0..1); None if
    quality scoring was skipped."""
    embedding: np.ndarray                   # (512,) float32, L2-normalised
    bbox: np.ndarray                        # (4,) x1,y1,x2,y2
    kps: np.ndarray                         # (5,2) 5-point landmarks
    det_score: float                        # detector confidence
    align: AlignMethod                      # how the crop fed to ArcFace was produced
    embed_version: int = 1
    quality: Optional[float] = None
    quality_parts: dict[str, float] = field(default_factory=dict)

    def as_template(self) -> dict[str, Any]:
        """Shape for store insertion: the embedder→vector map + provenance the matcher/ingest want."""
        return {
            "embeddings": {"face": self.embedding.tolist()},
            "modality": "face",
            "quality": self.quality,
            "embed_versions": {"face": self.embed_version},
        }


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n == 0.0 else (v / n).astype(np.float32)


class FaceEmbedder:
    """SCRFD + ArcFace, with explicit alignment control.

    Wraps the insightface `buffalo_l` pack: `det_10g` (SCRFD, gives 5 landmarks) for detection and
    `w600k_r50` (ArcFace) for the embedding. We drive detection and recognition separately (rather
    than letting `FaceAnalysis.get` align internally) so the *alignment step is ours* and the ablation
    is real — the box-crop path is genuinely a box crop fed to the same ArcFace net.

    Args:
        model_pack: insightface model pack name (default 'buffalo_l' → ArcFace r50 + SCRFD).
        providers: onnxruntime execution providers, e.g. ['CUDAExecutionProvider','CPUExecutionProvider'].
        det_size: SCRFD input size; larger finds smaller faces at more cost.
        det_thresh: SCRFD score threshold.
    """

    def __init__(self, model_pack: str = models.DEFAULT_FACE_PACK,
                 providers: Optional[Sequence[str]] = None,
                 det_size: tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5,
                 embed_version: int = 1) -> None:
        # Lazy, isolated imports so the package imports without ML deps present.
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:  # pragma: no cover - only hit without the `face` extra installed
            raise RuntimeError(
                "FaceEmbedder needs the 'face' extra (insightface, onnxruntime, opencv). "
                "Install with: pip install 'nmem-identity[face]'"
            ) from e

        if providers is None:
            import onnxruntime as ort  # type: ignore
            providers = list(ort.get_available_providers())

        self.embed_version = int(embed_version)
        self.model_pack = model_pack
        # This pack's recognition model_id — what a vector from this embedder must be stored under.
        # Unknown packs are allowed (custom weights) but self-label so nothing silently mislabels.
        self.model_id = models.FACE_PACK_MODEL_ID.get(model_pack, f"arcface_{model_pack}")
        self._app = FaceAnalysis(name=model_pack, providers=list(providers))
        # ctx_id: 0 selects the first GPU if a CUDA provider is active; CPU provider ignores it.
        self._app.prepare(ctx_id=0, det_size=tuple(det_size), det_thresh=det_thresh)
        self._det = self._app.models["detection"]
        rec = self._app.models.get("recognition")
        if rec is None:  # pragma: no cover
            raise RuntimeError(f"model pack {model_pack!r} has no recognition (ArcFace) model")
        self._rec = rec
        # face_align.norm_crop implements the canonical 5-point similarity transform to 112×112.
        from insightface.utils import face_align  # type: ignore
        self._face_align = face_align

    # -- detection ---------------------------------------------------------------------------------
    def detect(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces. Returns [{bbox(4), kps(5,2), det_score}] sorted by area, largest first."""
        bboxes, kpss = self._det.detect(bgr, max_num=0, metric="default")
        out = []
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2, score = bboxes[i][:5]
            out.append({
                "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                "kps": (None if kpss is None else kpss[i].astype(np.float32)),
                "det_score": float(score),
            })
        out.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                 reverse=True)
        return out

    # -- alignment (the ablation lever) ------------------------------------------------------------
    def align(self, bgr: np.ndarray, det: dict[str, Any], method: AlignMethod = "landmark") -> np.ndarray:
        """Produce the 112×112 BGR crop fed to ArcFace.

        method="landmark": 5-point similarity transform (norm_crop) — the correct path.
        method="boxcrop":  crop the detector bbox and resize to 112×112 — the naive baseline the
                           ablation exists to shame. Requires no landmarks.
        """
        if method == "landmark":
            kps = det.get("kps")
            if kps is None:
                raise ValueError("landmark alignment needs 5-point kps (SCRFD provides them)")
            return self._face_align.norm_crop(bgr, landmark=np.asarray(kps), image_size=ARCFACE_INPUT)
        if method == "boxcrop":
            import cv2  # type: ignore
            x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
            h, w = bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("degenerate bbox for boxcrop")
            return cv2.resize(bgr[y1:y2, x1:x2], (ARCFACE_INPUT, ARCFACE_INPUT))
        raise ValueError(f"unknown align method {method!r}")

    # -- embedding ---------------------------------------------------------------------------------
    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """ArcFace embedding of an already-112×112-aligned BGR crop → (512,) L2-normalised float32."""
        feat = self._rec.get_feat(aligned_bgr)          # insightface handles mean/std internally
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        if vec.shape[0] != EMBED_DIM:  # pragma: no cover
            raise ValueError(f"expected {EMBED_DIM}-d ArcFace embedding, got {vec.shape[0]}")
        return _l2norm(vec)

    def embed_crop(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Alias — embed a caller-supplied aligned 112×112 crop (used by the /embed sidecar and the
        benchmark when crops are pre-cut)."""
        return self.embed_aligned(aligned_bgr)

    # -- one-shot convenience ----------------------------------------------------------------------
    def observe(self, bgr: np.ndarray, *, align: AlignMethod = "landmark",
                largest_only: bool = True, score_quality: bool = True) -> list[FaceObservation]:
        """Full pipeline on a frame: detect → align → embed (→ quality). Returns one
        FaceObservation per face (or just the largest, the single-subject default). Empty list if no
        face is detected — a `no_face` regime the caller degrades on, it is not an error."""
        dets = self.detect(bgr)
        if not dets:
            return []
        if largest_only:
            dets = dets[:1]
        obs: list[FaceObservation] = []
        for d in dets:
            aligned = self.align(bgr, d, method=align)
            emb = self.embed_aligned(aligned)
            o = FaceObservation(embedding=emb, bbox=d["bbox"], kps=d.get("kps"),
                                det_score=d["det_score"], align=align,
                                embed_version=self.embed_version)
            if score_quality:
                from .quality import face_quality
                o.quality, o.quality_parts = face_quality(bgr, d, aligned)
            obs.append(o)
        return obs
