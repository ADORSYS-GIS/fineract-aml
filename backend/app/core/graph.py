"""Graph database (Memgraph/Bolt) connection management — ADR 0007.

A lazily-created async Bolt driver, mirroring `core/database.py`. The graph layer is
optional and **fail-open**: when `graph_enabled` is false, or the driver/`neo4j` package
is unavailable, every helper degrades to a no-op (`None` / `False`) so importing this
module and calling it can never crash the scoring path.

Memgraph speaks the Bolt protocol, so the official `neo4j` async driver works unchanged
and keeps us portable to Neo4j (same driver, same Cypher) — only the URL changes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from neo4j import AsyncDriver, AsyncGraphDatabase

    HAS_NEO4J = True
except ImportError:  # pragma: no cover - exercised only when the driver is absent
    AsyncDriver = object  # type: ignore[assignment,misc]
    AsyncGraphDatabase = None  # type: ignore[assignment]
    HAS_NEO4J = False
    logger.info("neo4j driver not installed — graph fraud layer disabled")


_driver: AsyncDriver | None = None


def graph_available() -> bool:
    """True only when the feature flag is on AND the driver package is importable."""
    return settings.graph_enabled and HAS_NEO4J


def _auth():
    """Bolt auth tuple from settings, or None for anonymous (Memgraph's default)."""
    user = settings.graph_database_user
    if user:
        return (user, settings.graph_database_password)
    return None


def _new_driver() -> AsyncDriver | None:
    """Build a driver bound to the *current* event loop. Returns None on failure."""
    try:
        return AsyncGraphDatabase.driver(settings.graph_database_url, auth=_auth())
    except Exception:
        logger.exception("Failed to initialize graph driver — degrading to no-op")
        return None


def get_driver() -> AsyncDriver | None:
    """Return the shared async Bolt driver, creating it on first use.

    Returns None (never raises) when the graph layer is disabled/unavailable, so callers
    can treat "no driver" as a clean fail-open path.

    NOTE: the neo4j async driver binds to the event loop that creates it. This singleton is
    therefore only safe under a long-lived loop (the uvicorn server). Celery tasks spin a
    fresh loop per invocation via `_run_async` — they MUST use `ephemeral_graph_session()`
    instead, or they would reuse a driver tied to a since-closed loop.
    """
    global _driver
    if not graph_available():
        return None
    if _driver is None:
        _driver = _new_driver()
        if _driver is not None:
            logger.info("Graph driver initialized → %s", settings.graph_database_url)
    return _driver


@asynccontextmanager
async def graph_session() -> AsyncGenerator[object | None, None]:
    """Yield an async graph session from the shared (server-loop) driver, or None.

    Use on the long-lived FastAPI loop (live scoring). For Celery tasks use
    `ephemeral_graph_session()`. Guard on `session is None`; the session is closed on exit.
    """
    driver = get_driver()
    if driver is None:
        yield None
        return
    session = driver.session()
    try:
        yield session
    finally:
        await session.close()


@asynccontextmanager
async def ephemeral_graph_session() -> AsyncGenerator[object | None, None]:
    """Yield a session from a fresh, single-use driver bound to the current loop.

    For Celery `_run_async` tasks: each runs in its own short-lived event loop, so a cached
    cross-loop driver would fail on the second task. Both the driver and session are closed
    on exit, avoiding the connection leak a per-task throwaway driver would otherwise cause.
    """
    if not graph_available():
        yield None
        return
    driver = _new_driver()
    if driver is None:
        yield None
        return
    session = driver.session()
    try:
        yield session
    finally:
        await session.close()
        await driver.close()


async def graph_healthcheck() -> bool:
    """Return True if a trivial Cypher round-trip succeeds. Never raises."""
    if not graph_available():
        return False
    try:
        async with graph_session() as session:
            if session is None:
                return False
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return bool(record and record["ok"] == 1)
    except Exception:
        logger.warning("Graph healthcheck failed", exc_info=True)
        return False


async def close_graph() -> None:
    """Close the shared driver on application shutdown."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
