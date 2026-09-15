"""Start matcher_server, test /match create+match over HTTP, clean up, stop.

Needs a Python that can reach the identity DB (configure via NMEM_IDENTITY_DSN). Run with the repo's
`src` on PYTHONPATH; the server is launched with the same interpreter running this test."""
import json, math, os, subprocess, sys, time, urllib.request, asyncio
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
from nmem_identity.store import db

# Port-configurable so the smoke never collides with a live matcher (default 9404). The launched
# server reads NMEM_IDENTITY_MATCHER_PORT via config, and we point the client at the same port.
PORT = int(os.environ.get("SMOKE_MATCHER_PORT", os.environ.get("NMEM_IDENTITY_MATCHER_PORT", "9404")))
_BASE = f"http://127.0.0.1:{PORT}"

def post(body):
    r = urllib.request.Request(f"{_BASE}/match", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=10).read())

def health():
    return json.loads(urllib.request.urlopen(f"{_BASE}/health", timeout=3).read())

async def cleanup(cids):
    pool = await db.get_pool()
    for c in {x for x in cids if x is not None}:
        await pool.execute("DELETE FROM identity_cluster WHERE cluster_id=$1", c)
    await db.close_pool()

def main():
    env = dict(os.environ, PYTHONPATH=str(_REPO / "src"), NMEM_IDENTITY_MATCHER_PORT=str(PORT))
    srv = subprocess.Popen([sys.executable, str(_REPO / "services" / "matcher_server.py")],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cids, ok = [], True
    try:
        for _ in range(30):
            try:
                if health().get("ok"): break
            except Exception: pass
            time.sleep(1)
        A = [math.sin(1*0.7 + i*0.013) for i in range(192)]
        B = [math.sin(999*0.7 + i*0.013) for i in range(192)]
        r1 = post({"embedding": A, "session_id": "t1", "quality": 2.0})
        r2 = post({"embedding": A, "session_id": "t2", "quality": 2.0})
        r3 = post({"embedding": B, "session_id": "t3", "quality": 2.0})
        cids += [r1["cluster_id"], r2["cluster_id"], r3["cluster_id"]]
        def chk(c, m):
            nonlocal ok; print(("PASS " if c else "FAIL ")+m); ok = ok and c
        chk(r1["action"] == "created" and r1["person_name"] is None, f"A created (cid={r1['cluster_id']}, unnamed)")
        chk(r2["action"] == "matched" and r2["cluster_id"] == r1["cluster_id"], f"A re-matched same cluster (sim={r2['sim']:.3f})")
        chk(r3["action"] == "created" and r3["cluster_id"] != r1["cluster_id"], f"B distinct cluster (sim vs A={r3['sim']:.3f})")

        # face channel: 512-d ArcFace-scale vectors, quality is a 0..1 score (NOT seconds). This only
        # forms/learns clusters if the matcher applies FACE thresholds, not the voice ones — the
        # regression guard for that fix.
        F1 = [math.sin(2 * 0.7 + i * 0.011) for i in range(512)]
        F2 = [math.sin(500 * 0.7 + i * 0.031) for i in range(512)]
        f1 = post({"embedding": F1, "embedder": "face", "session_id": "tf1", "quality": 0.8})
        f2 = post({"embedding": F1, "embedder": "face", "session_id": "tf2", "quality": 0.8})
        f3 = post({"embedding": F2, "embedder": "face", "session_id": "tf3", "quality": 0.8})
        cids += [f1.get("cluster_id"), f2.get("cluster_id"), f3.get("cluster_id")]
        chk(f1["action"] == "created", f"FACE F1 created a cluster (cid={f1['cluster_id']}) — face thresholds applied")
        chk(f2["action"] == "matched" and f2["cluster_id"] == f1["cluster_id"], f"FACE F1 re-matched (sim={f2['sim']:.3f})")
        chk(f3["action"] == "created" and f3["cluster_id"] != f1["cluster_id"], f"FACE F2 distinct cluster (sim vs F1={f3['sim']:.3f})")

        # dim guard
        try:
            post({"embedding": A[:100]}); chk(False, "short embedding should 400")
        except urllib.error.HTTPError as e:
            chk(e.code == 400, "short embedding rejected 400")
    finally:
        asyncio.run(cleanup(cids))
        srv.terminate(); srv.wait(timeout=10)
    print("\n" + ("MATCHER E2E PASS" if ok else "MATCHER E2E FAILED"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
