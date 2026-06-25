"""Graph backfill & ring-detection tasks (ADR 0007).

- `backfill_graph`: rebuild/repair the profile-asset graph from Postgres (the authoritative
  store) — cold start + drift correction. Also syncs `known_bad` risk anchors from
  confirmed-fraud alerts so guilt-by-association has something to anchor on.
- `detect_rings`: off-critical-path scan for assets shared by many accounts (the
  collector-smurfing / agent-collusion signal) → raises GRAPH_ANALYSIS alerts.

Both follow the fresh-engine-per-task pattern used by `tasks/training.py` to avoid asyncpg
fork-safety issues, and both no-op when `graph_enabled` is false.
"""

import asyncio
import logging
from datetime import UTC

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Heuristic floor/scale for ring-alert risk scores.
_RING_BASE_SCORE = 0.5
_RING_PER_ACCOUNT = 0.05
_RING_MAX_SCORE = 0.95


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="app.tasks.graph_backfill.backfill_graph")
def backfill_graph(self):
    """Daily: rebuild the graph from recent Postgres transactions."""
    return _run_async(_backfill_async())


@celery_app.task(bind=True, name="app.tasks.graph_backfill.detect_rings")
def detect_rings(self):
    """Every 6h: surface shared-asset rings and raise alerts."""
    return _run_async(_detect_rings_async())


async def _backfill_async() -> dict:
    from datetime import datetime, timedelta

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.models.alert import Alert, AlertStatus
    from app.models.transaction import Transaction

    if not settings.graph_enabled:
        return {"skipped": "graph_disabled"}

    from app.core.graph import ephemeral_graph_session
    from app.graph.client import GraphClient, tx_to_graph_payload
    from app.graph.schema import ensure_schema

    cutoff = datetime.now(UTC) - timedelta(days=settings.graph_backfill_lookback_days)

    engine = create_async_engine(settings.database_url, pool_size=5, max_overflow=2)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    upserted = 0
    anchors = 0
    try:
        async with session_maker() as db:
            tx_rows = await db.execute(
                select(Transaction).where(Transaction.transaction_date >= cutoff)
            )
            transactions = list(tx_rows.scalars().all())

            # Accounts implicated in a confirmed-fraud alert become known_bad anchors.
            bad_rows = await db.execute(
                select(Transaction.fineract_account_id)
                .join(Alert, Alert.transaction_id == Transaction.id)
                .where(Alert.status == AlertStatus.CONFIRMED_FRAUD)
                .distinct()
            )
            bad_accounts = {row[0] for row in bad_rows.all()}

        async with ephemeral_graph_session() as gsession:
            if gsession is None:
                return {"skipped": "graph_unavailable"}
            await ensure_schema(gsession)
            client = GraphClient(gsession)
            for tx in transactions:
                await client.upsert_transaction(tx_to_graph_payload(tx))
                upserted += 1
            for account_id in bad_accounts:
                await client.set_account_flag(account_id, "known_bad", True)
                anchors += 1

            # Activate the (previously dormant) NetworkX analyzer to compute centrality /
            # community metrics and write them onto Account nodes for the ML feature reads.
            await _populate_centrality(client, transactions)
    finally:
        await engine.dispose()

    logger.info("Graph backfill: upserted %d tx, %d known_bad anchors", upserted, anchors)
    return {"upserted": upserted, "anchors": anchors}


async def _populate_centrality(client, transactions) -> None:
    """Run TransactionGraphAnalyzer over money-flow transfers; write pagerank / community /
    is_in_cycle back to Account nodes. Fail-soft — metrics are best-effort enrichment.
    """
    try:
        from app.ml.graph_analyzer import TransactionGraphAnalyzer

        analyzer = TransactionGraphAnalyzer()
        if not analyzer.is_available:
            return
        analyzer.build_graph(transactions)
        graph = analyzer.graph

        import networkx as nx

        # Connected-component → community size lookup.
        community_size: dict[str, int] = {}
        for component in nx.weakly_connected_components(graph):
            size = len(component)
            for node in component:
                community_size[node] = size

        for account_id in graph.nodes():
            feats = analyzer.get_network_features(account_id)
            await client.set_account_metrics(
                account_id,
                pagerank=feats.get("pagerank", 0.0),
                community_size=community_size.get(account_id, 1),
                is_in_cycle=int(feats.get("is_in_cycle", 0.0)),
            )
    except Exception:
        logger.warning("Centrality population skipped (non-fatal)", exc_info=True)


async def _detect_rings_async() -> dict:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.models.alert import Alert, AlertSource, AlertStatus
    from app.models.transaction import Transaction

    if not settings.graph_enabled:
        return {"skipped": "graph_disabled"}

    from app.core.graph import ephemeral_graph_session
    from app.graph.client import GraphClient

    async with ephemeral_graph_session() as gsession:
        if gsession is None:
            return {"skipped": "graph_unavailable"}
        clusters = await GraphClient(gsession).shared_asset_clusters(min_size=3)

    if not clusters:
        return {"clusters": 0, "alerts_created": 0}

    engine = create_async_engine(settings.database_url, pool_size=5, max_overflow=2)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    created = 0
    try:
        async with session_maker() as db:
            for cluster in clusters:
                accounts = cluster["accounts"]
                fan = cluster["fan"]
                title = (
                    f"Shared-{cluster['asset_type']} ring: {fan} accounts "
                    f"share {cluster['asset_type'].lower()} {cluster['asset_id'][:12]}"
                )

                # Dedupe: skip if an open ring alert with this title already exists.
                dupe = await db.execute(
                    select(Alert.id).where(
                        Alert.title == title,
                        Alert.status.in_([AlertStatus.PENDING, AlertStatus.UNDER_REVIEW]),
                    )
                )
                if dupe.first() is not None:
                    continue

                # Anchor the alert to the most recent transaction among the ring's accounts.
                tx_row = await db.execute(
                    select(Transaction)
                    .where(Transaction.fineract_account_id.in_(accounts))
                    .order_by(Transaction.transaction_date.desc())
                    .limit(1)
                )
                transaction = tx_row.scalar_one_or_none()
                if transaction is None:
                    # Fail-loud: a detected ring with no anchorable transaction must be visible,
                    # not silently dropped (Alert.transaction_id is non-nullable).
                    logger.warning(
                        "Ring cluster on %s %s (%d accounts) has no anchorable transaction — "
                        "alert skipped",
                        cluster["asset_type"], cluster["asset_id"], fan,
                    )
                    continue

                risk = min(_RING_BASE_SCORE + _RING_PER_ACCOUNT * fan, _RING_MAX_SCORE)
                db.add(Alert(
                    transaction_id=transaction.id,
                    status=AlertStatus.PENDING,
                    source=AlertSource.GRAPH_ANALYSIS,
                    risk_score=risk,
                    title=title,
                    description=(
                        f"{fan} accounts share {cluster['asset_type']} "
                        f"{cluster['asset_id']}: {', '.join(accounts[:10])}"
                        + (" …" if fan > 10 else "")
                    ),
                ))
                created += 1
            await db.commit()
    finally:
        await engine.dispose()

    logger.info("Ring detection: %d clusters, %d alerts created", len(clusters), created)
    return {"clusters": len(clusters), "alerts_created": created}
