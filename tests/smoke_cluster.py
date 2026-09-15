"""cluster.match_or_create against live DB: match / create / quality-floor. Isolated modality='ctest'."""
import asyncio, sys, math
sys.path.insert(0, "/mnt/nas_projects/apps/nmem-identity/src")
from nmem_identity import cluster
from nmem_identity.store import db, templates as T
MOD="ctest"
def vec(seed,n=192): return [math.sin(seed*0.7+i*0.013) for i in range(n)]
def jitter(v,s=0.02):
    import random; random.seed(1); return [x+random.uniform(-s,s) for x in v]
async def main():
    pool=await db.get_pool(); ok=True
    def chk(c,m):
        nonlocal ok; print(("PASS " if c else "FAIL ")+m); ok=ok and c
    A=vec(1); B=vec(999)
    # 1. first sighting -> created + learned
    r1=await cluster.match_or_create("campp",A,session_id="s1",modality=MOD,quality=2.0)
    chk(r1["action"]=="created" and r1["is_new"] and r1["learned"], f"first sighting created+learned (cid={r1['cluster_id']})")
    cidA=r1["cluster_id"]
    # 2. same voice, good quality -> matched + learned, member bumps to 2
    r2=await cluster.match_or_create("campp",jitter(A),session_id="s2",modality=MOD,quality=2.0)
    chk(r2["action"]=="matched" and r2["cluster_id"]==cidA and r2["learned"], f"same-voice matched+learned (sim={r2['sim']:.3f})")
    mc=await pool.fetchval("SELECT member_count FROM identity_cluster WHERE cluster_id=$1",cidA)
    chk(mc==2, f"member_count==2 after 2 learned (got {mc})")
    # 3. distant voice -> created (new cluster)
    r3=await cluster.match_or_create("campp",B,session_id="s2",modality=MOD,quality=2.0)
    chk(r3["action"]=="created" and r3["cluster_id"]!=cidA, f"distant voice -> new cluster (sim vs A={r3['sim']:.3f})")
    # 4. same voice but SHORT (quality<1.5) -> matched_nolearn, member stays 2
    r4=await cluster.match_or_create("campp",jitter(A),session_id="s3",modality=MOD,quality=0.5)
    chk(r4["action"]=="matched_nolearn" and not r4["learned"] and r4["cluster_id"]==cidA, "short utterance recognised but not learned")
    mc2=await pool.fetchval("SELECT member_count FROM identity_cluster WHERE cluster_id=$1",cidA)
    chk(mc2==2, f"member_count unchanged by short utterance (got {mc2})")
    # 5. distant NEW voice but SHORT (quality<1.5) -> unknown_lowquality, NO cluster created (leak fix)
    C=vec(555)
    before=await pool.fetchval("SELECT count(*) FROM identity_cluster WHERE modality=$1", MOD)
    r5=await cluster.match_or_create("campp",C,session_id="s4",modality=MOD,quality=0.4)
    after=await pool.fetchval("SELECT count(*) FROM identity_cluster WHERE modality=$1", MOD)
    chk(r5["action"]=="unknown_lowquality" and r5["cluster_id"] is None and after==before,
        f"low-quality miss -> unknown, NO cluster leaked (before={before} after={after})")
    # cleanup
    await pool.execute("DELETE FROM identity_cluster WHERE modality=$1", MOD)
    stray=await pool.fetchval("SELECT count(*) FROM identity_cluster WHERE modality=$1", MOD)
    chk(stray==0, f"cleanup ok (stray={stray})")
    await db.close_pool()
    print("\n"+("ALL PASS" if ok else "SOME FAILED")); sys.exit(0 if ok else 1)
asyncio.run(main())
