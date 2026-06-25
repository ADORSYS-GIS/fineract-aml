"""Data retention tasks — archival and purging per regulatory requirements.

AML regulations (CEMAC/COBAC) require:
- Minimum 5-year retention of transaction records and SARs
- Purging of personal data after the retention period (data protection)
- Audit trail of all purging actions
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Default retention periods (configurable via env)
TRANSACTION_RETENTION_YEARS = 7
AUDIT_LOG_RETENTION_YEARS = 10
SCREENING_RETENTION_YEARS = 5


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _enforce_retention():
    from sqlalchemy import delete, func, select

    from app.core.database import async_session
    from app.models.sanctions import ScreeningResult, ScreeningStatus
    from app.models.transaction import Transaction
    from app.services.audit_service import AuditService

    now = datetime.now(UTC)
    tx_cutoff = now - timedelta(days=TRANSACTION_RETENTION_YEARS * 365)
    screening_cutoff = now - timedelta(days=SCREENING_RETENTION_YEARS * 365)

    async with async_session() as db:
        audit = AuditService(db)

        # Count eligible records (don't delete yet — just report)
        tx_count = (
            await db.execute(
                select(func.count(Transaction.id)).where(
                    Transaction.created_at < tx_cutoff
                )
            )
        ).scalar_one()

        screening_count = (
            await db.execute(
                select(func.count(ScreeningResult.id)).where(
                    ScreeningResult.created_at < screening_cutoff,
                    ScreeningResult.status == ScreeningStatus.CLEAR,
                )
            )
        ).scalar_one()

        if tx_count > 0:
            logger.info(
                "Data retention: %d transactions older than %d years eligible for archival",
                tx_count,
                TRANSACTION_RETENTION_YEARS,
            )
            # In production, this would archive to cold storage before deleting.
            # For now, log the count for compliance visibility.
            await audit.log(
                action="retention_check",
                resource_type="transaction",
                details={
                    "eligible_count": tx_count,
                    "cutoff_date": tx_cutoff.isoformat(),
                    "retention_years": TRANSACTION_RETENTION_YEARS,
                    "action": "flagged_for_archival",
                },
            )

        if screening_count > 0:
            # Clear screenings are safe to purge after retention period
            result = await db.execute(
                delete(ScreeningResult).where(
                    ScreeningResult.created_at < screening_cutoff,
                    ScreeningResult.status == ScreeningStatus.CLEAR,
                )
            )
            logger.info(
                "Data retention: purged %d clear screening results older than %d years",
                result.rowcount,
                SCREENING_RETENTION_YEARS,
            )
            await audit.log(
                action="retention_purge",
                resource_type="screening_result",
                details={
                    "purged_count": result.rowcount,
                    "cutoff_date": screening_cutoff.isoformat(),
                    "retention_years": SCREENING_RETENTION_YEARS,
                },
            )

        await db.commit()


@celery_app.task(name="app.tasks.retention.enforce_data_retention")
def enforce_data_retention():
    """Check and enforce data retention policies. Runs monthly."""
    _run_async(_enforce_retention())
