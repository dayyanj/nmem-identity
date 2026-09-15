"""End-to-end: synthetic captures -> ingest -> store -> KNN match -> cleanup (live identity DB)."""
import asyncio, json, os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmem_identity import ingest
from nmem_identity.store import db, templates as T

async def main():
    rng = np.random.default_rng(1)
    centers = {"itest_a": rng.normal(size=192), "itest_b": rng.normal(size=192)}
    caps, labels = [], {}
    ts = 1.0
    for sp in centers:
        for sess in ("s1","s2"):
            for i in range(8):
                v = centers[sp] + rng.normal(scale=0.3, size=192)
                ts += 1
                key = f"{sp}_{sess}_{i}.wav"
                caps.append({"ts":ts,"session_id":sess,"sr":48000,"duration_s":2.0,
                             "campp_emb":v.tolist(),"wav":f"/x/{key}"})
                labels[key] = {"speaker":sp,"regime":"clean"}
    d = tempfile.mkdtemp()
    capf, labf = os.path.join(d,"c.jsonl"), os.path.join(d,"l.json")
    open(capf,"w").write("\n".join(json.dumps(c) for c in caps)); open(labf,"w").write(json.dumps(labels))

    await ingest.run(capf, labf, reset=True)

    pool = await db.get_pool()
    ok = True
    for sp in centers:
        pid = await pool.fetchval("SELECT person_id FROM identity_person WHERE display_name=$1", sp)
        row = await pool.fetchrow("SELECT cluster_id, member_count, centroid_campp IS NOT NULL AS hasc, "
                                  "campp_model_id FROM identity_cluster WHERE person_id=$1", pid)
        print(f"{sp}: person={pid} cluster={row['cluster_id']} members={row['member_count']} "
              f"centroid={row['hasc']} model={row['campp_model_id']}")
        ok = ok and row["member_count"]==16 and row["hasc"] and row["campp_model_id"]=="campplus_onnx_v1"
    # KNN: a fresh A-like vector should match itest_a's cluster, not B's
    q = (centers["itest_a"] + rng.normal(scale=0.3, size=192)).tolist()
    hits = await T.knn_clusters("campp", q, k=2)
    pid_a = await pool.fetchval("SELECT person_id FROM identity_person WHERE display_name='itest_a'")
    top_person = hits[0]["person_id"]
    print(f"KNN top: cluster={hits[0]['cluster_id']} person={top_person} sim={hits[0]['sim']:.3f} (a_person={pid_a})")
    ok = ok and top_person == pid_a

    # cleanup
    for sp in centers:
        pid = await pool.fetchval("SELECT person_id FROM identity_person WHERE display_name=$1", sp)
        if pid: await T.delete_person(pid)
    stray = await pool.fetchval("SELECT count(*) FROM identity_person WHERE display_name LIKE 'itest_%'")
    ok = ok and stray==0
    print(f"cleanup: stray itest persons={stray}")
    await db.close_pool()
    print("\n" + ("INGEST E2E PASS" if ok else "INGEST E2E FAILED"))
    sys.exit(0 if ok else 1)

asyncio.run(main())
