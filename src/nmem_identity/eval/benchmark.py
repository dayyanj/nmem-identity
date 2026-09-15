"""Gate-0 benchmark — is CAM++ good enough to recognise this population by voice?

Reads a capture log (`captures.jsonl`, one CAM++ voiceprint per utterance) plus a hand-label file,
and answers the Gate-0 questions on REAL data, stratified by acoustic regime:

  * How is the real distribution split across regimes (clean / noisy / short / crosstalk)?
  * Verification: same-speaker vs different-speaker cosine separation -> EER, per regime.
  * Identification: leave-one-SESSION-out nearest-centroid accuracy (the live task: a new session's
    utterance matched against centroids built from OTHER sessions) -> cross-session re-ID, per regime.

**Gate 0 (from the plan):** if `clean` dominates the real distribution AND CAM++-only identification
accuracy there is >= ~0.98-0.99 AND degraded regimes are rare, ship CAM++-NN + conversational
enrolment and STOP. Proceed to the heavier apparatus only if noisy/short/crosstalk are materially
frequent and hurt accuracy. This script prints those numbers and a verdict; the operator makes the call.

Usage:
    python -m nmem_identity.eval.benchmark \
        --captures <capture-dir>/captures.jsonl \
        --labels   <capture-dir>/labels.json

`labels.json`: { "<utt_key>": {"speaker": "alex", "regime": "clean"}, ... } where <utt_key> is the wav
basename (preferred, stable) or "<session_id>:<ts>". Utterances with no label are ignored (reported).
Generate a blank label template with `--make-template`.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

REGIMES = ("clean", "noisy", "short", "crosstalk")


# ---- loading ---------------------------------------------------------------------------------

def _utt_key(rec: dict) -> str:
    wav = rec.get("wav") or ""
    if wav:
        return os.path.basename(wav)
    return f"{rec.get('session_id')}:{rec.get('ts')}"


def load(captures: str, labels: Optional[str]) -> list[dict]:
    """Return labelled utterances: [{key, vec(np.float32), speaker, regime, session, dur}]."""
    lab = json.loads(Path(labels).read_text()) if labels and Path(labels).exists() else {}
    out, unlabelled = [], 0
    for line in Path(captures).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        emb = rec.get("campp_emb")
        if not emb:
            continue
        key = _utt_key(rec)
        meta = lab.get(key)
        if not meta:
            unlabelled += 1
            continue
        out.append({
            "key": key, "vec": np.asarray(emb, dtype=np.float32),
            "speaker": meta["speaker"], "regime": meta.get("regime", "clean"),
            "session": rec.get("session_id", ""), "dur": rec.get("duration_s", 0.0),
        })
    if unlabelled:
        print(f"[note] {unlabelled} captured utterances have no label (ignored)")
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(_unit(a) @ _unit(b))


# ---- verification (EER) ----------------------------------------------------------------------

def eer(genuine: list[float], impostor: list[float]) -> tuple[float, float]:
    """Equal error rate + its threshold from genuine/impostor cosine scores. Returns (eer, thr).

    This is the *discrete* EER — the observed operating point where |FAR-FRR| is smallest — not an
    interpolated ROC crossing. On small datasets it can differ slightly from interpolated EER; for a
    household go/no-go gate the discrete value is sufficient (and never optimistic — it reports a real
    achievable operating point)."""
    if not genuine or not impostor:
        return float("nan"), float("nan")
    g = np.asarray(genuine); im = np.asarray(impostor)
    thrs = np.unique(np.concatenate([g, im]))
    # EER = the threshold where FAR (impostors accepted) and FRR (genuine rejected) are closest.
    best_t, best_gap, best_val = float(thrs[0]), 1e9, 1.0
    for t in thrs:
        far = float((im >= t).mean())
        frr = float((g < t).mean())
        if abs(far - frr) < best_gap:
            best_gap, best_val, best_t = abs(far - frr), (far + frr) / 2, float(t)
    return best_val, best_t


def verification(utts: list[dict], regime: Optional[str] = None) -> dict:
    us = [u for u in utts if regime is None or u["regime"] == regime]
    gen, imp = [], []
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            s = _cos(us[i]["vec"], us[j]["vec"])
            (gen if us[i]["speaker"] == us[j]["speaker"] else imp).append(s)
    e, t = eer(gen, imp)
    return {"n": len(us), "pairs_gen": len(gen), "pairs_imp": len(imp),
            "eer": e, "thr": t,
            "mean_gen": float(np.mean(gen)) if gen else float("nan"),
            "mean_imp": float(np.mean(imp)) if imp else float("nan")}


# ---- identification (leave-one-session-out nearest centroid) ----------------------------------

def identify_loso(utts: list[dict], regime: Optional[str] = None) -> dict:
    """For each utterance, build per-speaker centroids from all OTHER sessions and classify to the
    nearest. Regime filter applies to the TEST utterance; centroids use all regimes (as live)."""
    test = [u for u in utts if regime is None or u["regime"] == regime]
    speakers = sorted({u["speaker"] for u in utts})
    if len(speakers) < 2 or not test:
        return {"n": len(test), "acc": float("nan"), "speakers": len(speakers)}
    correct = 0
    scored = 0
    for u in test:
        cents = {}
        for sp in speakers:
            vecs = [x["vec"] for x in utts
                    if x["speaker"] == sp and x["session"] != u["session"]]
            if vecs:
                cents[sp] = _unit(np.mean([_unit(v) for v in vecs], axis=0))
        if len(cents) < 2 or u["speaker"] not in cents:
            continue                                  # can't score (speaker only in this session)
        pred = max(cents, key=lambda sp: _unit(u["vec"]) @ cents[sp])
        correct += int(pred == u["speaker"])
        scored += 1
    return {"n": len(test), "scored": scored,
            "acc": (correct / scored) if scored else float("nan"),
            "speakers": len(speakers)}


# ---- report ----------------------------------------------------------------------------------

def report(utts: list[dict]) -> dict:
    reg_counts = Counter(u["regime"] for u in utts)
    spk_counts = Counter(u["speaker"] for u in utts)
    sess = len({u["session"] for u in utts})
    total = len(utts)
    print(f"\n=== nmem-identity Gate-0 benchmark (CAM++-only) ===")
    print(f"utterances={total}  speakers={len(spk_counts)}  sessions={sess}")
    print(f"per speaker : {dict(spk_counts)}")
    print(f"per regime  : {dict(reg_counts)}")
    if total:
        clean_frac = reg_counts.get("clean", 0) / total
        print(f"clean fraction = {clean_frac:.3f}")

    print("\n-- verification (EER; lower better) --")
    ov = verification(utts)
    print(f"  overall     EER={ov['eer']:.4f}  mean_gen={ov['mean_gen']:.3f} "
          f"mean_imp={ov['mean_imp']:.3f}  (gen={ov['pairs_gen']} imp={ov['pairs_imp']})")
    per_reg_ver = {}
    for r in REGIMES:
        if reg_counts.get(r):
            v = verification(utts, r)
            per_reg_ver[r] = v
            print(f"  {r:9s}   EER={v['eer']:.4f}  n={v['n']}")

    print("\n-- identification (leave-one-session-out nearest centroid; higher better) --")
    ov_id = identify_loso(utts)
    print(f"  overall     acc={ov_id['acc']:.4f}  (scored={ov_id.get('scored')}/{ov_id['n']})")
    per_reg_id = {}
    for r in REGIMES:
        if reg_counts.get(r):
            d = identify_loso(utts, r)
            per_reg_id[r] = d
            print(f"  {r:9s}   acc={d['acc']:.4f}  (scored={d.get('scored')}/{d['n']})")

    # Gate-0 verdict. `clean_acc` alone can be met by a single scoreable utterance, so the SHIP path
    # also requires enough SCORED clean utterances (identify_loso skips unscoreable ones).
    print("\n-- Gate 0 verdict --")
    MIN_CLEAN_SCORED = 30
    clean_frac = reg_counts.get("clean", 0) / total if total else 0.0
    clean_id = per_reg_id.get("clean", {})
    clean_acc = clean_id.get("acc", float("nan"))
    clean_scored = clean_id.get("scored", 0) or 0
    degraded = sum(reg_counts.get(r, 0) for r in ("noisy", "short", "crosstalk"))
    degraded_frac = degraded / total if total else 0.0
    verdict = "INSUFFICIENT DATA"
    if total >= 50 and len(spk_counts) >= 2 and sess >= 2:
        if (clean_frac >= 0.8 and not np.isnan(clean_acc) and clean_acc >= 0.98
                and clean_scored >= MIN_CLEAN_SCORED and degraded_frac <= 0.2):
            verdict = "SHIP CAM++-ONLY (clean dominates + high accuracy; heavier apparatus unjustified)"
        elif clean_scored < MIN_CLEAN_SCORED:
            verdict = f"INSUFFICIENT DATA (only {clean_scored} scoreable clean utterances; need >= {MIN_CLEAN_SCORED})"
        elif degraded_frac > 0.2:
            verdict = f"PROCEED to Phase 1 (degraded regimes {degraded_frac:.0%} are material)"
        else:
            verdict = "MARGINAL — inspect per-regime accuracy before deciding"
    print(f"  clean_frac={clean_frac:.2f}  clean_id_acc={clean_acc:.3f}  "
          f"clean_scored={clean_scored}  degraded_frac={degraded_frac:.2f}")
    print(f"  => {verdict}")
    return {"total": total, "regimes": dict(reg_counts), "speakers": dict(spk_counts),
            "sessions": sess, "verification": ov, "identification": ov_id, "verdict": verdict}


def make_template(captures: str, out: str) -> None:
    """Write a blank labels template (one entry per captured utterance) for the operator to fill."""
    tmpl = {}
    for line in Path(captures).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not rec.get("campp_emb"):
            continue
        tmpl[_utt_key(rec)] = {"speaker": "", "regime": "clean",
                               "_dur": rec.get("duration_s"), "_wav": rec.get("wav")}
    Path(out).write_text(json.dumps(tmpl, indent=2))
    print(f"wrote {len(tmpl)} blank label rows -> {out}  (fill 'speaker' + 'regime' per utterance)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--make-template", default=None, metavar="OUT.json",
                    help="write a blank labels template from the captures and exit")
    args = ap.parse_args()
    if args.make_template:
        make_template(args.captures, args.make_template)
        return
    utts = load(args.captures, args.labels)
    if not utts:
        print("no labelled utterances found; run with --make-template to create a label file first")
        return
    report(utts)


if __name__ == "__main__":
    main()
