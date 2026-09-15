"""Face pipeline public-benchmark validation + the alignment ablation (Phase 0a, Gate 0a).

Runs the SAME `perception.face` pipeline the live system uses against a public verification benchmark
(LFW out of the box; any pair list works) and answers two questions with NO captured data:

  1. **Does the pipeline reproduce known numbers?** If ArcFace `buffalo_l` doesn't land near its
     published LFW accuracy/EER here, the bug is in *our* detection/alignment — and this catches it
     before household data can mask it.
  2. **How much does 5-point-landmark alignment actually buy?** The ablation runs verification with
     `align=landmark` vs `align=boxcrop` on identical pairs and reports the gap. The design predicts a
     large gap (~2× EER); proving it here is what makes "alignment is non-negotiable" evidence, not
     dogma.

This is deliberately the FIRST face deliverable because it is unblocked by hardware and de-risks the
highest-risk component. It needs the `face` + `eval` extras (insightface, onnxruntime, opencv,
scikit-learn) and a downloaded dataset; it never touches the identity DB.

Usage:
    python -m nmem_identity.eval.face_benchmark --dataset lfw \\
        --root /data/lfw/lfw-deepfunneled --pairs /data/lfw/pairs.txt \\
        --align landmark,boxcrop [--limit 1000]

    # or a generic pair list: JSON [["a.jpg","b.jpg",true], ...] with paths relative to --root
    python -m nmem_identity.eval.face_benchmark --dataset generic --root /data --pairs pairs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# --- dataset pair loaders --------------------------------------------------------------------------

def load_lfw_pairs(pairs_file: str, root: str) -> list[tuple[str, str, bool]]:
    """Parse the canonical LFW `pairs.txt`. Genuine line: `name i j`; impostor line: `n1 i n2 j`.
    Image path convention: <root>/<name>/<name>_<idx:04d>.jpg."""
    root_p = Path(root)

    def img(name: str, idx: str) -> str:
        return str(root_p / name / f"{name}_{int(idx):04d}.jpg")

    pairs: list[tuple[str, str, bool]] = []
    lines = [ln.strip() for ln in Path(pairs_file).read_text().splitlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) == 3:                              # genuine: same person
            name, i, j = parts
            pairs.append((img(name, i), img(name, j), True))
        elif len(parts) == 4:                            # impostor: two people
            n1, i, n2, j = parts
            pairs.append((img(n1, i), img(n2, j), False))
        # header lines (e.g. "10 300") have 1-2 tokens → skipped
    return pairs


def load_generic_pairs(pairs_file: str, root: str) -> list[tuple[str, str, bool]]:
    """A JSON list of [path_a, path_b, same_bool], paths relative to (or under) --root."""
    root_p = Path(root)
    data = json.loads(Path(pairs_file).read_text())
    out = []
    for a, b, same in data:
        out.append((str(root_p / a), str(root_p / b), bool(same)))
    return out


# --- embedding + scoring ---------------------------------------------------------------------------

def _read_bgr(path: str):
    import cv2  # type: ignore
    img = cv2.imread(path)
    return img  # None if unreadable


def embed_images(embedder, image_paths: list[str], method: str,
                 verbose: bool = True) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Embed each unique image once under one alignment method. Returns (embeddings, failures) where
    `failures[path]` is a reason string for images with no usable face (detection miss / unreadable).
    A detection miss under `landmark` is a real pipeline outcome (`no_face`), reported not hidden."""
    embs: dict[str, np.ndarray] = {}
    failures: dict[str, str] = {}
    total = len(image_paths)
    for n, p in enumerate(image_paths):
        if verbose and n % 500 == 0:
            print(f"  [{method}] embedding {n}/{total} ...", file=sys.stderr)
        bgr = _read_bgr(p)
        if bgr is None:
            failures[p] = "unreadable"
            continue
        try:
            obs = embedder.observe(bgr, align=method, largest_only=True, score_quality=False)
        except Exception as e:                            # alignment/embarrassing edge → treat as miss
            failures[p] = f"error:{type(e).__name__}"
            continue
        if not obs:
            failures[p] = "no_face"
            continue
        embs[p] = obs[0].embedding
    return embs, failures


def score_pairs(pairs: list[tuple[str, str, bool]],
                embs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Cosine similarity per pair (embeddings are L2-normalised → dot). Pairs where either image
    failed to embed are skipped and counted. Returns (sims, labels, n_skipped)."""
    sims, labels, skipped = [], [], 0
    for a, b, same in pairs:
        va, vb = embs.get(a), embs.get(b)
        if va is None or vb is None:
            skipped += 1
            continue
        sims.append(float(np.dot(va, vb)))
        labels.append(1 if same else 0)
    return np.asarray(sims, dtype=np.float64), np.asarray(labels, dtype=np.int32), skipped


def verification_metrics(sims: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """EER, ROC-AUC, and best-threshold accuracy for a similarity score vs binary same/different."""
    if sims.size == 0:
        return {"n": 0, "eer": float("nan"), "auc": float("nan"),
                "acc": float("nan"), "thr": float("nan")}
    try:
        from sklearn.metrics import roc_auc_score, roc_curve  # type: ignore
        fpr, tpr, thr = roc_curve(labels, sims)
        auc = float(roc_auc_score(labels, sims))
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except Exception:                                    # numpy-only fallback (no sklearn extra)
        order = np.argsort(-sims)
        s_sorted, l_sorted = sims[order], labels[order]
        P, N = int(labels.sum()), int((labels == 0).sum())
        thr_cands = np.unique(sims)
        best_eer, eer = 1.0, 1.0
        for t in thr_cands:
            pred = sims >= t
            tp = int(((pred == 1) & (labels == 1)).sum()); fp = int(((pred == 1) & (labels == 0)).sum())
            far = fp / N if N else 0.0
            frr = 1.0 - (tp / P if P else 0.0)
            if abs(far - frr) < best_eer:
                best_eer = abs(far - frr); eer = (far + frr) / 2.0
        auc = float("nan")

    # Best-threshold accuracy (balanced verification accuracy at the optimal operating point).
    thr_cands = np.unique(sims)
    accs = [(float(((sims >= t).astype(int) == labels).mean()), float(t)) for t in thr_cands]
    acc, best_thr = max(accs, key=lambda x: x[0])
    return {"n": int(sims.size), "eer": eer, "auc": auc, "acc": acc, "thr": best_thr}


# --- driver ----------------------------------------------------------------------------------------

def _embed_rgb(embedder, img, method):
    """Embed one in-memory RGB image (as sklearn's LFW loader returns them). Returns the 512-d vector
    or None on a detection miss. Handles float[0,1]/[0,255] → uint8 and RGB→BGR for insightface."""
    import cv2  # type: ignore
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (a * 255.0 if float(a.max()) <= 1.0 else a).astype(np.uint8)
    bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    obs = embedder.observe(bgr, align=method, largest_only=True, score_quality=False)
    return obs[0].embedding if obs else None


def run_sklearn_lfw(methods: list[str], subset: str = "test",
                    limit: Optional[int] = None, pack: Optional[str] = None) -> dict:
    """Gate-0a via sklearn's built-in LFW (downloaded from figshare, cached) — no umass files needed.
    `subset`: 'test' (1000 pairs), 'train' (2200), '10_folds' (6000, the full standard protocol)."""
    from nmem_identity import models
    from nmem_identity.perception.face import FaceEmbedder
    from sklearn.datasets import fetch_lfw_pairs  # type: ignore

    pack = pack or models.DEFAULT_FACE_PACK
    print(f"[face-benchmark] source=sklearn-lfw subset={subset} pack={pack} methods={methods}")
    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0, slice_=None, funneled=True)
    pairs, labels = data.pairs, data.target       # pairs: (n,2,H,W,3); target: 1=same,0=different
    if limit:
        pairs, labels = pairs[:limit], labels[:limit]
    print(f"  loaded {len(pairs)} pairs ({int((labels == 1).sum())} genuine / {int((labels == 0).sum())} impostor)")

    embedder = FaceEmbedder(model_pack=pack)
    results: dict[str, dict] = {}
    for method in methods:
        sims, ys, det_fail = [], [], 0
        for i in range(len(pairs)):
            va = _embed_rgb(embedder, pairs[i][0], method)
            vb = _embed_rgb(embedder, pairs[i][1], method)
            if va is None or vb is None:
                det_fail += 1
                continue
            sims.append(float(np.dot(va, vb)))
            ys.append(int(labels[i]))
            if i % 200 == 0:
                print(f"  [{method}] {i}/{len(pairs)} ...", file=sys.stderr)
        m = verification_metrics(np.asarray(sims), np.asarray(ys))
        m.update({"detection_failures": det_fail, "embed_failures": det_fail, "pairs_skipped": det_fail})
        results[method] = m
    _print_report(results, methods, pack)
    return results


def run(dataset: str, root: str, pairs_file: str, methods: list[str],
        limit: Optional[int] = None, pack: Optional[str] = None) -> dict:
    try:
        from nmem_identity import models
        from nmem_identity.perception.face import FaceEmbedder
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"face pipeline unavailable: {e}\nInstall: pip install 'nmem-identity[face,eval]'")
    pack = pack or models.DEFAULT_FACE_PACK          # default = the LIGHT pack the live path uses

    if dataset == "lfw":
        pairs = load_lfw_pairs(pairs_file, root)
    elif dataset == "generic":
        pairs = load_generic_pairs(pairs_file, root)
    else:
        raise SystemExit(f"unknown dataset {dataset!r} (lfw|generic)")
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise SystemExit("no pairs loaded")

    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"[face-benchmark] dataset={dataset} pack={pack} pairs={len(pairs)} "
          f"unique_images={len(uniq)} methods={methods}")

    embedder = FaceEmbedder(model_pack=pack)
    results: dict[str, dict] = {}
    for method in methods:
        embs, failures = embed_images(embedder, uniq, method)
        sims, labels, skipped = score_pairs(pairs, embs)
        m = verification_metrics(sims, labels)
        det_fail = sum(1 for r in failures.values() if r == "no_face")
        m.update({"detection_failures": det_fail, "embed_failures": len(failures),
                  "pairs_skipped": skipped})
        results[method] = m

    _print_report(results, methods, pack)
    return results


# Published LFW verification accuracy per pack, for the Gate-0a sanity check.
_PACK_REF_ACC = {"buffalo_s": 0.995, "buffalo_l": 0.998}


def _print_report(results: dict[str, dict], methods: list[str], pack: str = "") -> None:
    print("\n=== Verification (public benchmark) ===")
    print(f"{'align':10s} {'n':>7s} {'EER':>8s} {'AUC':>7s} {'acc':>7s} {'thr':>7s} "
          f"{'no_face':>8s} {'skipped':>8s}")
    for method in methods:
        r = results[method]
        print(f"{method:10s} {r['n']:>7d} {r['eer']:>8.4f} {r['auc']:>7.4f} {r['acc']:>7.4f} "
              f"{r['thr']:>7.3f} {r['detection_failures']:>8d} {r['pairs_skipped']:>8d}")

    if "landmark" in results and "boxcrop" in results:
        el, eb = results["landmark"]["eer"], results["boxcrop"]["eer"]
        print("\n=== Alignment ablation ===")
        print(f"landmark EER {el:.4f}  vs  boxcrop EER {eb:.4f}")
        if el > 0:
            print(f"box-crop is {eb / el:.2f}× the landmark EER "
                  f"({'confirms alignment matters' if eb > 1.3 * el else 'gap smaller than expected — investigate'})")
    # Sanity note against the published reference for THIS pack (light buffalo_s ~0.995 LFW acc;
    # heavy buffalo_l ~0.998). Being ~1% under the reference points at our detection/alignment, not
    # the model.
    if "landmark" in results:
        acc = results["landmark"]["acc"]
        ref = _PACK_REF_ACC.get(pack, 0.99)
        floor = ref - 0.01
        verdict = (f"OK — near published {pack} numbers (~{ref:.3f})" if acc >= floor else
                   f"BELOW published {pack} numbers (~{ref:.3f}) — suspect detection/alignment, fix here")
        print(f"\n[Gate 0a] pack={pack} landmark accuracy {acc:.4f} → {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sklearn-lfw", choices=["sklearn-lfw", "lfw", "generic"],
                    help="sklearn-lfw (default; figshare download, no umass files) | lfw (pairs.txt) | generic")
    ap.add_argument("--root", help="image root directory (lfw/generic only)")
    ap.add_argument("--pairs", help="LFW pairs.txt or a generic pairs JSON (lfw/generic only)")
    ap.add_argument("--subset", default="test", choices=["test", "train", "10_folds"],
                    help="sklearn-lfw subset: test=1000 pairs, train=2200, 10_folds=6000 (full protocol)")
    ap.add_argument("--align", default="landmark,boxcrop",
                    help="comma list of alignment methods to score (default runs the ablation)")
    ap.add_argument("--pack", default=None,
                    help="face model pack (default: the live light pack; pass buffalo_l for the ceiling)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of pairs (quick runs)")
    args = ap.parse_args()
    methods = [m.strip() for m in args.align.split(",") if m.strip()]
    if args.dataset == "sklearn-lfw":
        run_sklearn_lfw(methods, subset=args.subset, limit=args.limit, pack=args.pack)
    else:
        if not (args.root and args.pairs):
            raise SystemExit(f"--root and --pairs are required for --dataset {args.dataset}")
        run(args.dataset, args.root, args.pairs, methods, args.limit, args.pack)


if __name__ == "__main__":
    main()
