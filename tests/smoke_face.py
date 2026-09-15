"""Functional smoke for the face pipeline — no dataset, no DB.

Loads the light pack (buffalo_s), detects on insightface's bundled sample image, aligns+embeds, and
asserts the pipeline is sane: 512-d L2-normalised vectors, self-similarity ~1, two different faces
clearly lower, and that landmark alignment vs a naive box-crop actually produce different embeddings
(so the Gate-0a ablation is measuring something real).

Needs the `face` extra (insightface, onnxruntime, opencv) and a one-time model download (buffalo_s
from the insightface model zoo). Run:
    .venv/bin/python tests/smoke_face.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmem_identity.perception.face import EMBED_DIM, FACE_MODEL_ID, FaceEmbedder  # noqa: E402


def main() -> int:
    ok = True

    def chk(c, m):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + m)
        ok = ok and c

    emb = FaceEmbedder()                      # buffalo_s (light pack)
    print(f"loaded pack={emb.model_pack} model_id={emb.model_id} providers-auto")
    chk(emb.model_id == FACE_MODEL_ID, f"embedder model_id == store's active face model_id ({FACE_MODEL_ID})")

    import insightface
    img = insightface.data.get_image("t1")    # bundled group photo (BGR)
    dets = emb.detect(img)
    chk(len(dets) >= 2, f"SCRFD detected {len(dets)} faces in the sample image")

    obs = emb.observe(img, largest_only=False)
    chk(len(obs) >= 2, f"observe() returned {len(obs)} embedded faces")
    v0 = obs[0].embedding
    chk(v0.shape[0] == EMBED_DIM, f"embedding is {EMBED_DIM}-d (got {v0.shape[0]})")
    chk(abs(float(np.linalg.norm(v0)) - 1.0) < 1e-3, "embedding is L2-normalised")

    aligned = emb.align(img, dets[0], method="landmark")
    s = float(np.dot(emb.embed_aligned(aligned), emb.embed_aligned(aligned)))
    chk(s > 0.999, f"self-similarity ~1 on the same crop (got {s:.4f})")

    if len(obs) >= 2:
        d = float(np.dot(obs[0].embedding, obs[1].embedding))
        chk(d < 0.6, f"two different faces score clearly lower (got {d:.3f})")

    lm = emb.embed_aligned(emb.align(img, dets[0], "landmark"))
    bc = emb.embed_aligned(emb.align(img, dets[0], "boxcrop"))
    chk(float(np.dot(lm, bc)) < 0.999, "landmark vs box-crop give different vectors (ablation is real)")

    print("\n" + ("FACE FUNCTIONAL PASS" if ok else "FACE FUNCTIONAL FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
