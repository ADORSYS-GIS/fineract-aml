"""Post-disbursement loan monitoring Celery tasks.

After a loan is disbursed, two checks are scheduled:
- 24h check: detect immediate loan-and-run patterns
- 48h check: detect structuring and dispersal patterns

These tasks create alerts via the standard alert pipeline when fraud is detected.
"""

import json
import logging
from datetime import timedelta
from uuid import UUID

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.check_loan_behavior", bind=True, max_retries=2)
def check_loan_behavior(self, loan_watch_id: str, check_window: str):
    """Check post-disbursement behavior at 24h or 48h mark.

    Args:
        loan_watch_id: UUID string of the LoanDisbursementWatch record.
        check_window: "24h" or "48h" to indicate which checkpoint this is.
    """
    import asyncio

    try:
        asyncio.run(_run_check(UUID(loan_watch_id), check_window))
    except Exception as exc:
        logger.error("Loan behavior check failed for %s: %s", loan_watch_id, exc)
        raise self.retry(exc=exc, countdown=300)


async def _run_check(loan_watch_id: UUID, check_window: str):
    from sqlalchemy import select

    from app.core.config import settings
    from app.core.database import async_session as AsyncSessionLocal
    from app.models.alert import Alert, AlertSource, AlertStatus
    from app.models.loan_watch import LoanDisbursementWatch, LoanWatchStatus
    from app.models.transaction import Transaction

    async with AsyncSessionLocal() as db:
        # Load the watch record
        result = await db.execute(
            select(LoanDisbursementWatch).where(LoanDisbursementWatch.id == loan_watch_id)
        )
        watch = result.scalar_one_or_none()
        if not watch:
            logger.warning("LoanDisbursementWatch %s not found", loan_watch_id)
            return
        if watch.status == LoanWatchStatus.FLAGGED:
            logger.info("Watch %s already flagged, skipping %s check", loan_watch_id, check_window)
            return

        # Fetch post-disbursement transactions
        result = await db.execute(
            select(Transaction)
            .where(Transaction.fineract_account_id == watch.fineract_account_id)
            .where(Transaction.transaction_date > watch.disbursed_at)
            .order_by(Transaction.transaction_date)
        )
        post_txns = list(result.scalars().all())

        findings = _analyze_post_disbursement(watch, post_txns, settings)

        # Load existing findings and merge
        existing_findings: dict = {}
        if watch.findings_json:
            try:
                existing_findings = json.loads(watch.findings_json)
            except json.JSONDecodeError:
                pass
        existing_findings[check_window] = findings
        watch.findings_json = json.dumps(existing_findings)

        # Create alert if any finding is high risk
        if findings.get("flagged"):
            watch.status = LoanWatchStatus.FLAGGED

            alert = Alert(
                transaction_id=watch.loan_transaction_id,
                status=AlertStatus.PENDING,
                source=AlertSource.LOAN_MONITORING,
                risk_score=findings["severity"],
                title=f"Post-disbursement fraud detected ({check_window}): "
                f"{watch.currency} {watch.disbursed_amount:,.2f} loan",
                description=findings["description"],
                triggered_rules=json.dumps(findings["triggered_patterns"]),
            )
            db.add(alert)
            await db.flush()
            watch.alert_id = alert.id
            logger.warning(
                "Loan fraud alert created for watch %s (%s check): %s",
                loan_watch_id,
                check_window,
                findings["triggered_patterns"],
            )
        elif check_window == "48h" and watch.status == LoanWatchStatus.ACTIVE:
            watch.status = LoanWatchStatus.CLEARED
            logger.info("Loan watch %s cleared at 48h", loan_watch_id)

        await db.commit()


def _analyze_post_disbursement(watch, post_txns: list, settings) -> dict:
    """Analyze post-disbursement transactions for fraud patterns."""
    from app.models.transaction import TransactionType

    disbursed = watch.disbursed_amount
    triggered_patterns: list[str] = []
    max_severity = 0.0
    descriptions: list[str] = []

    transfers = [t for t in post_txns if t.transaction_type == TransactionType.TRANSFER]
    withdrawals = [t for t in post_txns if t.transaction_type == TransactionType.WITHDRAWAL]

    # 1. Loan-and-run: >80% of disbursed amount transferred out
    total_transferred = sum(t.amount for t in transfers)
    run_ratio = total_transferred / disbursed if disbursed > 0 else 0
    if run_ratio > settings.loan_run_threshold:
        triggered_patterns.append("loan_and_run")
        max_severity = max(max_severity, 0.9)
        descriptions.append(
            f"Loan-and-run: {run_ratio:.0%} of disbursed amount "
            f"({watch.currency} {total_transferred:,.2f}) transferred out"
        )

    # 2. Immediate cash-out: large withdrawal within N minutes
    cashout_window = timedelta(minutes=settings.loan_immediate_cashout_minutes)
    immediate_cashouts = [
        t for t in withdrawals
        if (t.transaction_date - watch.disbursed_at) <= cashout_window
        and t.amount >= disbursed * 0.8
    ]
    if immediate_cashouts:
        triggered_patterns.append("immediate_cash_out")
        max_severity = max(max_severity, 0.95)
        descriptions.append(
            f"Immediate cash-out: {len(immediate_cashouts)} withdrawal(s) within "
            f"{settings.loan_immediate_cashout_minutes} min totalling "
            f"{watch.currency} {sum(t.amount for t in immediate_cashouts):,.2f}"
        )

    # 3. Structuring after disbursement: multiple transfers each below threshold summing to ~loan
    structuring_transfers = [
        t for t in transfers
        if t.amount < settings.structuring_threshold
    ]
    if (
        len(structuring_transfers) >= 3
        and sum(t.amount for t in structuring_transfers) >= disbursed * 0.7
    ):
        triggered_patterns.append("post_disbursement_structuring")
        max_severity = max(max_severity, 0.75)
        descriptions.append(
            f"Post-disbursement structuring: {len(structuring_transfers)} transfers "
            f"each below {settings.structuring_threshold:,.2f} summing to "
            f"{watch.currency} {sum(t.amount for t in structuring_transfers):,.2f}"
        )

    # 4. Cross-agent dispersal: funds sent to many unique counterparties
    unique_counterparties = len({
        t.counterparty_account_id
        for t in transfers
        if t.counterparty_account_id
    })
    if unique_counterparties >= settings.loan_dispersal_counterparty_min:
        triggered_patterns.append("cross_agent_dispersal")
        max_severity = max(max_severity, 0.7)
        descriptions.append(
            f"Cross-agent dispersal: funds sent to {unique_counterparties} unique "
            f"counterparties (threshold: {settings.loan_dispersal_counterparty_min})"
        )

    return {
        "flagged": len(triggered_patterns) > 0,
        "triggered_patterns": triggered_patterns,
        "severity": max_severity,
        "description": "; ".join(descriptions) if descriptions else "No suspicious patterns found",
        "total_transferred": sum(t.amount for t in transfers),
        "total_withdrawn": sum(t.amount for t in withdrawals),
        "unique_counterparties": unique_counterparties,
        "transfer_count": len(transfers),
    }
