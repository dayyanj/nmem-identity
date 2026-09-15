"""Fire-and-forget viz event emitter for nmem-identity.

Cross-process by design: the matcher runs as its own service (:9404), separate from the agent's
process, so it can't use nmem-sym's in-process ``viz_on()`` registry. Instead it POSTs events over
the network to the nmem-viz hub's ``/ingest`` endpoint, where they're merged into the live /ws stream
alongside the nmem-sym + nmem events.

Entirely env-gated and best-effort:
  * ``NMEM_IDENTITY_VIZ_URL``  unset  ⇒ ``emit()`` is a zero-cost no-op (default; behaviour unchanged).
  * ``NMEM_IDENTITY_VIZ_URL``  set    ⇒ e.g. ``http://localhost:5174/ingest``.
  * ``NMEM_IDENTITY_VIZ_TOKEN`` optional shared secret (sent as ``X-Ingest-Token``).

Every failure is swallowed — recognition must never block or break on the visualization.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

_log = logging.getLogger("identity.viz")

VIZ_URL = os.environ.get("NMEM_IDENTITY_VIZ_URL", "")
VIZ_TOKEN = os.environ.get("NMEM_IDENTITY_VIZ_TOKEN", "")

_session: Optional[Any] = None  # lazily-created aiohttp.ClientSession


def viz_enabled() -> bool:
    return bool(VIZ_URL)


async def _post(payload: dict) -> None:
    global _session
    try:
        import aiohttp
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession()
        headers = {"X-Ingest-Token": VIZ_TOKEN} if VIZ_TOKEN else None
        async with _session.post(
            VIZ_URL, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=2),
        ) as resp:
            await resp.read()
    except Exception as e:  # noqa: BLE001 - best-effort; viz never breaks recognition
        _log.debug("viz post failed: %s", e)


def emit(event_type: str, data: dict[str, Any]) -> None:
    """Emit a viz event (fire-and-forget). No-op unless NMEM_IDENTITY_VIZ_URL is set.

    Schedules a background POST on the running loop; if there's no running loop (e.g. called from
    sync/offline code), it silently skips — the matcher endpoints are async so a loop is present."""
    if not VIZ_URL:
        return
    payload = {"type": event_type, "source": "nmem-identity", "data": data}
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(_post(payload))
