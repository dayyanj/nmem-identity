"""Operator command — fit a channel's calibration curve from a labelled eval set and persist it to
the store, so live fusion has real evidence weights instead of abstaining.

Fusion (`fuse.py`) turns each channel's cosine into an LLR via a stored `identity_calibration` curve;
with no curve the channel abstains (contributes nothing). This command closes that loop for the
**text_style** channel: it scores an authorship pair list with the SAME LUAR embedder + episode
settings the live sidecar uses (reusing `eval.text_benchmark`), fits a length-banded calibration
(`calibrate.fit_banded`), and upserts it under `(luar_v1, embed_version, 'text_style')`.

This is the step the plan gates all downstream text trust on — the length-stratified curve is fit here,
not hand-picked. Run it after `text_benchmark` says the EER is acceptable; the persisted curve is what
lets a deployment turn fusion on.

    python -m nmem_identity.eval.fit_calibration --pairs pairs.json \\
        [--model rrivera1849/LUAR-MUD] [--bands 40,120,280] [--embed-version 1] [--dry-run]

Needs the `text` + `store` extras (torch, transformers, asyncpg). `--dry-run` fits and prints without
writing (no DB needed for the fit itself). Voice/face calibration is not wired here yet — their score
distributions come from the DB-capture benchmarks, a separate fit path behind the same DAL.
"""
from __future__ import annotations

import argparse
import asyncio

from .. import calibrate, config, models
from ..perception import text_style as text_mod
from . import text_benchmark


def _report(cal: calibrate.Calibration) -> None:
    p = cal.pooled
    print(f"\n== text_style calibration (channel model_id={models.model_id('text_style')}) ==")
    print(f"  pooled:  a={p.a:+.4f}  b={p.b:+.4f}  prior_logodds={p.prior_logodds:+.4f}  "
          f"(n_gen={p.n_genuine} n_imp={p.n_impostor})")
    if cal.bands:
        print("  per-length-band curves (retained authored chars):")
        for label in sorted(cal.bands):
            c = cal.bands[label]
            print(f"    {label:>10}:  a={c.a:+.4f}  b={c.b:+.4f}  "
                  f"(n_gen={c.n_genuine} n_imp={c.n_impostor})")
    else:
        print("  (no band had enough both-class pairs — pooled curve only)")
    # A quick sanity read: LLR at a few cosines, short vs long.
    print("  sample LLRs (cosine → LLR):")
    for cos in (0.1, 0.3, 0.5, 0.7):
        short = cal.llr(cos, length=30)
        long = cal.llr(cos, length=1000)
        print(f"    cos={cos:.2f}  short(30ch)={short:+.3f}  long(1000ch)={long:+.3f}")


async def _persist(cal: calibrate.Calibration, model_id: str, embed_version: int) -> None:
    from ..store import calibration as calib_store   # store extra only needed when actually writing
    # Persist under the FITTED checkpoint's model_id (emb.model_id), NOT the store's active one: fitting
    # a non-default checkpoint (e.g. LUAR-CRUD) for an offline comparison must save under ITS key
    # (luar_crud_v1), never overwrite the live luar_v1 curve. The live matcher only loads the curve
    # whose model_id matches its active generation, so an off-generation fit is simply inert (codex).
    await calib_store.save_calibration(model_id, embed_version, "text_style", cal)
    note = "" if model_id == models.model_id("text_style") else "  [non-active checkpoint — inert until live]"
    print(f"\npersisted text_style calibration → identity_calibration "
          f"(model_id={model_id}, embed_version={embed_version}).{note}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit + persist the text_style calibration curve.")
    ap.add_argument("--pairs", required=True, help="JSON [[text_a, text_b, same_author], ...]")
    ap.add_argument("--model", default=text_mod.DEFAULT_TEXT_MODEL, help="HF LUAR checkpoint.")
    ap.add_argument("--limit", type=int, default=None, help="Cap pairs (debug).")
    ap.add_argument("--bands", default="40,120,280",
                    help="Retained-char band edges, comma-separated (default 40,120,280).")
    ap.add_argument("--min-per-band", type=int, default=20,
                    help="Min both-class pairs to fit a band-specific curve (else pooled fallback).")
    ap.add_argument("--embed-version", type=int, default=1,
                    help="Store key: the text_style embed_version this curve calibrates.")
    ap.add_argument("--episode-len", type=int, default=int(config.get("text_episode_len")))
    ap.add_argument("--doc-max-tokens", type=int, default=int(config.get("text_doc_max_tokens")))
    ap.add_argument("--dry-run", action="store_true", help="Fit + print but do not write to the store.")
    args = ap.parse_args()
    bands = [int(x) for x in args.bands.split(",") if x.strip()]

    pairs = text_benchmark.load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    # Embed the pairs ONCE (the expensive LUAR pass) and reuse those scored rows for BOTH the EER
    # report and the calibration fit — text_benchmark.run would re-embed a second time (codex). The
    # rows are (cosine, same, retained_length), exactly what fit_banded consumes, so the curve is fit
    # on the same scores the EER summarises.
    emb = text_benchmark.build_embedder(args.model, args.episode_len, args.doc_max_tokens)
    scored = text_benchmark.score_pairs(emb, pairs)
    res = text_benchmark.summarise(scored, model_name=args.model, episode_len=args.episode_len,
                                   doc_max_tokens=args.doc_max_tokens, bands=bands)
    text_benchmark.report(res)                        # EER context first — do not calibrate blind

    cal = calibrate.fit_banded(scored, band_edges=bands, channel="text_style",
                               min_per_band=args.min_per_band)
    _report(cal)
    if args.dry_run:
        print("\n[dry-run] not persisted.")
        return
    asyncio.run(_persist(cal, emb.model_id, args.embed_version))


if __name__ == "__main__":
    main()
