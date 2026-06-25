"""Nightly Celery task to compute and update AgentProfile baselines.

Runs every night to maintain per-agent behavioral baselines used by
agent-specific AML rules (agent_structuring, agent_float_anomaly, etc.).
"""

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 30
_MIN_TRANSACTIONS = 10  # Skip agents with too little history


@celery_app.task(name="tasks.update_agent_profiles", bind=True, max_retries=2)
def update_agent_profiles(self):
    """Compute 30-day behavioral baselines for all active agents."""
    import asyncio

    try:
        asyncio.run(_run_update())
    except Exception as exc:
        logger.error("Agent profile update failed: %s", exc)
        raise self.retry(exc=exc, countdown=300)


async def _run_update():
    from app.core.database import AsyncSessionLocal
    from app.models.agent_profile import AgentProfile
    from app.models.transaction import Transaction

    async with AsyncSessionLocal() as db:
        cutoff = datetime.now(UTC) - timedelta(days=_LOOKBACK_DAYS)

        # Fetch all agent transactions in the lookback window
        result = await db.execute(
            select(Transaction)
            .where(Transaction.agent_id.isnot(None))
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.agent_id, Transaction.transaction_date)
        )
        all_txns = list(result.scalars().all())

        if not all_txns:
            logger.info("No agent transactions found in last %d days", _LOOKBACK_DAYS)
            return

        # Group by agent_id
        by_agent: dict[str, list] = defaultdict(list)
        for txn in all_txns:
            by_agent[txn.agent_id].append(txn)

        updated = 0
        for agent_id, txns in by_agent.items():
            if len(txns) < _MIN_TRANSACTIONS:
                continue

            profile = _compute_profile(agent_id, txns)

            # Upsert
            existing = await db.execute(
                select(AgentProfile).where(AgentProfile.agent_id == agent_id)
            )
            agent_profile = existing.scalar_one_or_none()
            if agent_profile is None:
                agent_profile = AgentProfile(agent_id=agent_id)
                db.add(agent_profile)

            agent_profile.branch_id = txns[-1].branch_id
            agent_profile.avg_daily_tx_count_30d = profile["avg_daily_tx_count"]
            agent_profile.avg_daily_volume_30d = profile["avg_daily_volume"]
            agent_profile.std_daily_volume_30d = profile["std_daily_volume"]
            agent_profile.typical_float_ratio = profile["float_ratio"]
            agent_profile.served_customer_count_30d = profile["unique_customers"]
            agent_profile.avg_new_customers_per_day_30d = profile["avg_new_customers_per_day"]
            agent_profile.avg_tx_amount_30d = profile["avg_tx_amount"]
            agent_profile.p95_tx_amount_30d = profile["p95_tx_amount"]
            agent_profile.peak_hour_distribution = json.dumps(profile["hour_distribution"])
            agent_profile.computed_from_days = _LOOKBACK_DAYS
            agent_profile.last_updated = datetime.now(UTC)

            updated += 1

        await db.commit()
        logger.info("Updated %d agent profiles", updated)


def _compute_profile(agent_id: str, txns: list) -> dict:
    """Compute behavioral metrics from a list of transactions for one agent."""
    import statistics
    from collections import Counter

    amounts = [t.amount for t in txns]
    deposit_amounts = [t.amount for t in txns if t.transaction_type.value == "deposit"]
    withdrawal_amounts = [t.amount for t in txns if t.transaction_type.value == "withdrawal"]

    total_deposits = sum(deposit_amounts)
    total_withdrawals = sum(withdrawal_amounts)
    total_volume = total_deposits + total_withdrawals
    float_ratio = total_deposits / total_volume if total_volume > 0 else 0.5

    # Daily aggregates
    daily_counts: dict = defaultdict(int)
    daily_volumes: dict = defaultdict(float)
    daily_new_customers: dict = defaultdict(set)

    for t in txns:
        day = t.transaction_date.date()
        daily_counts[day] += 1
        daily_volumes[day] += t.amount
        if getattr(t, "kyc_level", None) == 1:
            daily_new_customers[day].add(t.fineract_client_id)

    daily_count_vals = list(daily_counts.values())
    daily_volume_vals = list(daily_volumes.values())

    avg_daily_tx = statistics.mean(daily_count_vals) if daily_count_vals else 0
    avg_daily_vol = statistics.mean(daily_volume_vals) if daily_volume_vals else 0
    std_daily_vol = statistics.stdev(daily_volume_vals) if len(daily_volume_vals) > 1 else 0

    avg_new_per_day = (
        statistics.mean([len(s) for s in daily_new_customers.values()])
        if daily_new_customers
        else 0
    )

    sorted_amounts = sorted(amounts)
    p95_idx = int(len(sorted_amounts) * 0.95)
    p95_amount = sorted_amounts[min(p95_idx, len(sorted_amounts) - 1)]

    hour_counts = Counter(t.transaction_date.hour for t in txns)

    unique_customers = len({t.fineract_client_id for t in txns})

    return {
        "avg_daily_tx_count": avg_daily_tx,
        "avg_daily_volume": avg_daily_vol,
        "std_daily_volume": std_daily_vol,
        "float_ratio": float_ratio,
        "unique_customers": unique_customers,
        "avg_new_customers_per_day": avg_new_per_day,
        "avg_tx_amount": statistics.mean(amounts) if amounts else 0,
        "p95_tx_amount": p95_amount,
        "hour_distribution": dict(hour_counts),
    }
