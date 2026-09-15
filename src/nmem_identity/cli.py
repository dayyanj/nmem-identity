"""nmem-identity command line — thin wrappers over the store/eval verbs.

    nmem-identity config                       # show resolved config (defaults < TOML < env)
    nmem-identity schema-apply                 # create the identity_* tables + indexes
    nmem-identity health                       # DB connectivity + which identity tables exist
    nmem-identity delete-person <person_id>    # canonical delete (returns raw-blob keys to unlink)
    nmem-identity seed-text-calibration        # seed a conservative DEFAULT text_style curve
    nmem-identity face-benchmark ...           # public-benchmark validation + alignment ablation
    nmem-identity voice-benchmark ...          # Gate-0 voice report

Deliberately dependency-light: `config`/`schema-apply`/`health`/`delete-person` need only the store
deps; the benchmark verbs pull their own extras.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys


def _cmd_config(_: argparse.Namespace) -> int:
    from . import config
    print(json.dumps(config.all_values(), indent=2, default=str))
    return 0


def _cmd_schema_apply(_: argparse.Namespace) -> int:
    from .store import db
    asyncio.run(db.apply_schema())
    print("schema applied")
    return 0


def _cmd_health(_: argparse.Namespace) -> int:
    from .store import db
    print(json.dumps(asyncio.run(db.healthcheck()), indent=2, default=str))
    return 0


def _cmd_delete_person(args: argparse.Namespace) -> int:
    from .store import db, templates as T

    async def _go() -> list[str]:
        refs = await T.delete_person(args.person_id)
        await db.close_pool()
        return refs

    refs = asyncio.run(_go())
    print(f"deleted person {args.person_id}; unlink {len(refs)} raw blob(s):")
    for r in refs:
        print(f"  {r}")
    return 0


def _cmd_seed_text_calibration(args: argparse.Namespace) -> int:
    """Seed a conservative DEFAULT text_style calibration so writing-style identity isn't inert on a
    fresh DB (an uncalibrated channel abstains → the feature silently does nothing). This is a generic
    analytic prior, NOT a fit — refit with `python -m nmem_identity.eval.fit_calibration` for accuracy.

    SAFE by construction: fusion caps text_style to a soft signal and never lets it authorize alone
    (it's a WEAK channel with no `strong` corroboration), so a rough default can only nudge a *guess*,
    never gate anything consequential. The curve is clipped to the same soft cap as belt-and-braces."""
    from . import models
    from .calibrate import Calibration, Curve
    from .store import calibration as C, db

    channel = "text_style"
    model_id = models.model_id(channel)          # "luar_v1" (schema-bound)
    embed_version = 1                            # _ACTIVE_EMBED_VERSION (services/matcher_server.py)

    async def _go() -> bool:
        existing = await C.load_calibration(model_id, embed_version, channel)
        if existing is not None and not args.force:
            await db.close_pool()
            return False
        # LLR(cos) = clip(a·cos + b): a=5, b=-1.5 crosses 0 near the match threshold (cos≈0.30) and
        # reaches the soft cap (~1.5) near cos≈0.6. n=0 provenance marks it a DEFAULT, not a fit.
        cal = Calibration(channel=channel,
                          pooled=Curve(a=5.0, b=-1.5, prior_logodds=0.0,
                                       n_genuine=0, n_impostor=0, clip=1.5))
        await C.save_calibration(model_id, embed_version, channel, cal)
        await db.close_pool()
        return True

    seeded = asyncio.run(_go())
    print(f"text_style default calibration {'seeded' if seeded else 'already present (use --force to overwrite)'}"
          f" for ({model_id}, v{embed_version})")
    return 0


def _cmd_face_benchmark(args: argparse.Namespace) -> int:
    from .eval import face_benchmark as FB
    methods = [m.strip() for m in args.align.split(",") if m.strip()]
    if args.dataset == "sklearn-lfw":
        FB.run_sklearn_lfw(methods, subset=args.subset, limit=args.limit, pack=args.pack)
    else:
        if not (args.root and args.pairs):
            raise SystemExit(f"--root and --pairs required for --dataset {args.dataset}")
        FB.run(args.dataset, args.root, args.pairs, methods, args.limit, args.pack)
    return 0


def _cmd_voice_benchmark(args: argparse.Namespace) -> int:
    from .eval import benchmark as VB
    sys.argv = ["voice-benchmark", "--captures", args.captures, "--labels", args.labels]
    VB.main()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="nmem-identity", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="show resolved config").set_defaults(fn=_cmd_config)
    sub.add_parser("schema-apply", help="create the identity_* tables").set_defaults(fn=_cmd_schema_apply)
    sub.add_parser("health", help="DB connectivity check").set_defaults(fn=_cmd_health)

    dp = sub.add_parser("delete-person", help="canonical delete of one person")
    dp.add_argument("person_id", type=int)
    dp.set_defaults(fn=_cmd_delete_person)

    sc = sub.add_parser("seed-text-calibration",
                        help="seed a conservative DEFAULT text_style calibration so writing-style ID "
                             "isn't inert on a fresh DB (refit for accuracy)")
    sc.add_argument("--force", action="store_true", help="overwrite an existing calibration")
    sc.set_defaults(fn=_cmd_seed_text_calibration)

    fb = sub.add_parser("face-benchmark", help="LFW verification + alignment ablation (Gate 0a)")
    fb.add_argument("--dataset", default="sklearn-lfw", choices=["sklearn-lfw", "lfw", "generic"])
    fb.add_argument("--root", default=None)
    fb.add_argument("--pairs", default=None)
    fb.add_argument("--subset", default="test", choices=["test", "train", "10_folds"])
    fb.add_argument("--align", default="landmark,boxcrop")
    fb.add_argument("--pack", default=None, help="face model pack (default: live light pack)")
    fb.add_argument("--limit", type=int, default=None)
    fb.set_defaults(fn=_cmd_face_benchmark)

    vb = sub.add_parser("voice-benchmark", help="Gate-0 voice report")
    vb.add_argument("--captures", required=True)
    vb.add_argument("--labels", required=True)
    vb.set_defaults(fn=_cmd_voice_benchmark)

    args = ap.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
