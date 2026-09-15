"""asyncpg pool + schema application for the nmem-identity store.

The store is a set of self-contained `identity_*` tables in a pgvector-enabled Postgres. Colocating
it in the same database as an nmem deployment makes the matcher's id-join to `nmem_entity_memory` a
plain local join, but that is a deployment choice, not a requirement — the connection is entirely
configured (see config.py / NMEM_IDENTITY_DSN).

This module NEVER touches nmem's SQLAlchemy models (they are 384-locked and would reject
vector(192)/vector(512)); it owns only the `identity_*` tables.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from .. import config

# The DSN is resolved from config (defaults < TOML < NMEM_IDENTITY_DSN). asyncpg uses the plain
# postgres:// scheme, NOT postgresql+asyncpg://.
DSN = config.dsn()

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide asyncpg pool, creating it on first use. Guarded by a lock so two
    concurrent first-callers can't each create (and leak) a pool."""
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:                    # re-check under the lock
                _pool = await asyncpg.create_pool(DSN, min_size=1, max_size=8)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def apply_schema() -> None:
    """Idempotently create the identity_* tables + indexes (schema.sql is all IF NOT EXISTS).

    Writes DDL to the configured database; run it deliberately against the intended target. HNSW
    index creation needs pgvector >= 0.5."""
    pool = await get_pool()
    sql = _SCHEMA_PATH.read_text()
    async with pool.acquire() as conn:
        await conn.execute(sql)


async def healthcheck() -> dict:
    """Confirm connectivity + that the identity tables exist."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'identity_%' ORDER BY tablename"
        )
    return {"connected": True, "identity_tables": [r["tablename"] for r in tables]}
