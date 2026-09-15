"""Text-style (LUAR authorship) public-benchmark validation — the authorship analogue of
face_benchmark, run BEFORE trusting the channel on real interlocutors.

Runs the SAME `perception.text_style` embedder the live system uses over a verification pair list and
answers the one question that gates every downstream use of this channel:

  **How reliable is authorship at chat length?** Text is the weakest modality and its reliability
  collapses as the authored text shrinks. So EER is reported OVERALL and, crucially, STRATIFIED BY
  AUTHORED LENGTH — the confidence-vs-length curve is exactly what Phase 5 fusion calibrates its
  weight from, and what stops us shipping trust ahead of evidence. If EER at short (chat-turn) lengths
  is poor — likely — the soft-prior framing still holds; we simply weight it lower.

Pair-list format (generic JSON): `[[text_a, text_b, same_author_bool], ...]`. Bring any authorship
verification set (PAN, Reddit cross-author, ...) reshaped to this. It never touches the identity DB.

Needs the `text` + `eval` extras (torch, transformers, scikit-learn is optional — EER here reuses the
pure `eval.benchmark.eer`). Usage:

    python -m nmem_identity.eval.text_benchmark --pairs pairs.json [--limit 2000] \\
        [--model rrivera1849/LUAR-MUD] [--bands 40,120,280]
"""
from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np

from . import benchmark
from .. import config
from ..perception import text_style as text_mod


def load_pairs(path: str) -> list[tuple[str, str, bool]]:
    """Parse a generic JSON pair list: [[text_a, text_b, same_author], ...]."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: list[tuple[str, str, bool]] = []
    for row in raw:
        if len(row) < 3:
            continue
        out.append((str(row[0]), str(row[1]), bool(row[2])))
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))            # both L2-normalised by the embedder → dot == cosine


def _band(n: int, edges: list[int]) -> str:
    """Label an authored-char count by length band, e.g. edges [40,120,280] → '<=40','41-120',...."""
    lo = 0
    for hi in edges:
        if n <= hi:
            return f"{lo+1}-{hi}" if lo else f"<={hi}"
        lo = hi
    return f">{edges[-1]}"


def score_pairs(emb, pairs: list[tuple[str, str, bool]]) -> list[tuple[float, bool, int]]:
    """Embed every verification pair once → scored rows ``(cosine, same_author, retained_len)``.

    The single expensive pass (LUAR inference over both texts of every pair). Both the EER summary and
    the calibration fit consume THESE rows, so a caller that needs both must embed only once (codex).
    Bucketing/curve length is the SHORTER side's RETAINED (post-truncation) length — the weaker signal
    governs, and retained-length is what the embedding actually reflects."""
    def _obs(text: str):
        # observe() truncates to the episode/token window and reports authored_chars = the RETAINED
        # length. score_quality=False: we only need the vector + retained length, not the gate score.
        return emb.observe(text_mod.split_documents(text), score_quality=False)

    scored: list[tuple[float, bool, int]] = []
    for a, b, same in pairs:
        oa, ob = _obs(a), _obs(b)
        scored.append((_cos(oa.embedding, ob.embedding), bool(same),
                       min(oa.authored_chars, ob.authored_chars)))
    return scored


def summarise(scored: list[tuple[float, bool, int]], *, model_name: str, episode_len: int,
              doc_max_tokens: int, bands: list[int]) -> dict:
    """EER overall + stratified by retained authored length, computed from pre-scored rows (no
    embedding). Separated from scoring so the calibration command can reuse the same rows it fits on."""
    def _eer(rows: list[tuple[float, bool, int]]) -> dict:
        gen = [s for s, same, _ in rows if same]
        imp = [s for s, same, _ in rows if not same]
        if not gen or not imp:
            return {"n": len(rows), "n_gen": len(gen), "n_imp": len(imp), "eer": float("nan")}
        e, t = benchmark.eer(gen, imp)
        return {"n": len(rows), "n_gen": len(gen), "n_imp": len(imp), "eer": e, "thr": t,
                "mean_gen": float(np.mean(gen)), "mean_imp": float(np.mean(imp))}

    overall = _eer(scored)
    by_len = {}
    for row in scored:
        by_len.setdefault(_band(row[2], bands), []).append(row)
    strat = {band: _eer(rows) for band, rows in sorted(by_len.items())}
    return {"model": model_name, "episode_len": episode_len, "doc_max_tokens": doc_max_tokens,
            "overall": overall, "by_length": strat}


def build_embedder(model_name: str, episode_len: int, doc_max_tokens: int):
    """The live LUAR embedder with the deployment's episode settings — else the benchmark embeds
    differently from production and its EER/thresholds don't transfer (codex)."""
    return text_mod.TextStyleEmbedder(model_name=model_name, episode_len=episode_len,
                                      doc_max_tokens=doc_max_tokens)


def run(pairs: list[tuple[str, str, bool]], *, model_name: str, limit: Optional[int],
        bands: list[int], episode_len: int, doc_max_tokens: int) -> dict:
    if limit:
        pairs = pairs[:limit]
    emb = build_embedder(model_name, episode_len, doc_max_tokens)
    scored = score_pairs(emb, pairs)
    return summarise(scored, model_name=model_name, episode_len=episode_len,
                     doc_max_tokens=doc_max_tokens, bands=bands)


def report(res: dict) -> None:
    ov = res["overall"]
    print(f"\n== text-style verification ({res['model']}) ==")
    print(f"  episode_len={res['episode_len']}  doc_max_tokens={res['doc_max_tokens']}  "
          f"(the effective preprocessing — must match the live sidecar's config)")
    print(f"  overall   EER={ov['eer']:.4f}  n={ov['n']} (gen={ov['n_gen']} imp={ov['n_imp']})")
    print("\n-- by RETAINED authored length (the confidence-vs-length curve; lower EER better) --")
    for band, v in res["by_length"].items():
        if v.get("eer") == v.get("eer"):     # not NaN
            print(f"  {band:>10}   EER={v['eer']:.4f}  n={v['n']}")
        else:
            print(f"  {band:>10}   EER=  n/a   n={v['n']} (need both genuine+impostor)")


def main() -> None:
    ap = argparse.ArgumentParser(description="LUAR authorship verification benchmark (length-stratified).")
    ap.add_argument("--pairs", required=True, help="JSON [[text_a, text_b, same_author], ...]")
    ap.add_argument("--model", default=text_mod.DEFAULT_TEXT_MODEL, help="HF LUAR checkpoint.")
    ap.add_argument("--limit", type=int, default=None, help="Cap pairs (debug).")
    ap.add_argument("--bands", default="40,120,280",
                    help="Retained-char band edges, comma-separated (default 40,120,280).")
    ap.add_argument("--episode-len", type=int, default=int(config.get("text_episode_len")),
                    help="Docs per LUAR episode (default: the deployment's text_episode_len).")
    ap.add_argument("--doc-max-tokens", type=int, default=int(config.get("text_doc_max_tokens")),
                    help="Tokens per document (default: the deployment's text_doc_max_tokens).")
    args = ap.parse_args()
    bands = [int(x) for x in args.bands.split(",") if x.strip()]
    res = run(load_pairs(args.pairs), model_name=args.model, limit=args.limit, bands=bands,
              episode_len=args.episode_len, doc_max_tokens=args.doc_max_tokens)
    report(res)


if __name__ == "__main__":
    main()
