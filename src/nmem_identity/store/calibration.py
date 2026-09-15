"""Persistence for per-channel calibration curves (the ``identity_calibration`` table).

A calibration curve (``calibrate.Calibration``) maps a channel's raw cosine to a prior-free LLR; the
store keeps it as JSONB keyed by ``(model_id, embed_version, channel)`` so a preprocessing / weights
change re-fits under a new key rather than silently reusing a stale curve. This module is the thin DAL:
upsert a fitted curve, load one, or load them all for the matcher to hold in memory.

The fit itself is `calibrate.py`; the operator command that runs an eval and calls `save_calibration`
is `eval/fit_calibration.py`. JSONB is sent/received as text (``$n::jsonb`` / ``curve::text``) so we
don't depend on an asyncpg JSON codec being registered — the same tactic templates.py uses for vectors.
"""
from __future__ import annotations

import json
from typing import Optional

from ..calibrate import Calibration
from .db import get_pool


async def save_calibration(model_id: str, embed_version: int, channel: str,
                           calibration: Calibration) -> None:
    """Upsert a channel's fitted calibration under ``(model_id, embed_version, channel)``. Overwrites
    an existing curve for the same key (a re-fit) and refreshes ``fitted_at``. ``n_genuine`` /
    ``n_impostor`` are taken from the pooled curve's provenance."""
    curve = calibration.as_curve_json()
    pooled = calibration.pooled
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO identity_calibration "
        "  (model_id, embed_version, channel, curve, n_genuine, n_impostor, fitted_at) "
        "VALUES ($1, $2, $3, $4::jsonb, $5, $6, now()) "
        "ON CONFLICT (model_id, embed_version, channel) DO UPDATE SET "
        "  curve = EXCLUDED.curve, n_genuine = EXCLUDED.n_genuine, "
        "  n_impostor = EXCLUDED.n_impostor, fitted_at = now()",
        model_id, int(embed_version), channel, json.dumps(curve),
        int(pooled.n_genuine), int(pooled.n_impostor),
    )


async def load_calibration(model_id: str, embed_version: int,
                           channel: str) -> Optional[Calibration]:
    """The calibration for one ``(model_id, embed_version, channel)``, or None if none is stored."""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT curve::text AS curve FROM identity_calibration "
        "WHERE model_id = $1 AND embed_version = $2 AND channel = $3",
        model_id, int(embed_version), channel,
    )
    if row is None or row["curve"] is None:
        return None
    return Calibration.from_curve_json(channel, json.loads(row["curve"]))


async def load_active_calibrations(
        active: dict[str, tuple[str, int]]) -> dict[str, Calibration]:
    """Load the calibration curve for each channel's ACTIVE model generation, keyed by channel.

    ``active`` maps ``channel -> (model_id, embed_version)`` — the generation the LIVE embeddings are
    produced under. A curve is loaded only when its ``(model_id, embed_version)`` matches that
    generation exactly; a channel with no matching curve is simply absent (fusion abstains). This is
    deliberately NOT "the most recently fitted curve": fitting time is not evidence of compatibility,
    and applying a curve fit on one model/version to another generation's scores would silently
    miscalibrate the whole channel (codex)."""
    out: dict[str, Calibration] = {}
    for channel, (model_id, embed_version) in active.items():
        cal = await load_calibration(model_id, embed_version, channel)
        if cal is not None:
            out[channel] = cal
    return out
