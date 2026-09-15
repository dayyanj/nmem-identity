"""Face pipeline latency + VRAM footprint benchmark (Tier 4 — shared-GPU sizing).

Measures single-frame detect+align+embed latency for the light pack (buffalo_s) under a chosen ONNX
provider + detector input size, and — on CUDA — the GPU memory this process holds (so we know the
sidecar's footprint on a shared GPU). Run on the deployment host:

    python tests/bench_face_latency.py --provider cpu  --det 320
    python tests/bench_face_latency.py --provider cpu  --det 640
    CUDA_VISIBLE_DEVICES=<3090-index> python tests/bench_face_latency.py --provider cuda --det 320
    CUDA_VISIBLE_DEVICES=<3090-index> python tests/bench_face_latency.py --provider cuda --det 640

Latency is per full frame (detect the whole image + align + embed the largest face) — the real
per-capture cost. Recognition runs at a low FPS (~2), so compare against a ~500 ms budget, not 30 fps.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmem_identity.perception.face import FaceEmbedder  # noqa: E402


def gpu_mem_for_pid(pid: int):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True)
        for line in out.splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                return int(m)
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--det", type=int, default=640)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()
    providers = (["CPUExecutionProvider"] if args.provider == "cpu"
                 else ["CUDAExecutionProvider", "CPUExecutionProvider"])

    import insightface
    img = insightface.data.get_image("t1")        # bundled multi-face photo

    mem_before = gpu_mem_for_pid(os.getpid())
    emb = FaceEmbedder(providers=providers, det_size=(args.det, args.det))
    try:
        active = emb._rec.session.get_providers()
    except Exception:
        active = ["?"]

    for _ in range(5):                             # warmup (CUDA kernels / graph build)
        emb.observe(img, largest_only=True, score_quality=False)
    mem_after = gpu_mem_for_pid(os.getpid())

    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        emb.observe(img, largest_only=True, score_quality=False)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    mean = statistics.mean(ts)
    p50 = ts[len(ts) // 2]
    p95 = ts[min(len(ts) - 1, int(len(ts) * 0.95))]

    print(f"provider={args.provider} det={args.det} iters={args.iters} active={active}")
    print(f"  latency ms: mean={mean:.1f}  p50={p50:.1f}  p95={p95:.1f}  -> ~{1000.0/mean:.1f} FPS (single face)")
    if args.provider == "cuda":
        held = (mem_after - mem_before) if (mem_before is not None and mem_after is not None) else mem_after
        print(f"  GPU mem held by this PID: {held} MiB (before={mem_before} after={mem_after})")


if __name__ == "__main__":
    main()
