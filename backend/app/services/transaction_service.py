"""Service layer for transaction processing."""

import json
import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertSource, AlertStatus
from app.models.transaction import RiskLevel, Transaction
from app.schemas.transaction import TransactionStats, WebhookPayload

logger = logging.getLogger(__name__)


class TransactionService:
    """Handles transaction ingestion, scoring, and querying."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def ingest_transaction(
        self,
        payload: WebhookPayload,
        data_quality_warnings: list[str] | None = None,
    ) -> tuple[Transaction, bool]:
        """Ingest a new transaction from a Fineract webhook.

        Returns:
            (transaction, is_new) — is_new is False if the transaction already existed (idempotent).
        """
        from app.core.config import settings

        # Idempotency: check if this Fineract transaction was already ingested
        existing = await self.db.execute(
            select(Transaction).where(
                Transaction.fineract_transaction_id == payload.transaction_id
            )
        )
        existing_tx = existing.scalar_one_or_none()
        if existing_tx:
            logger.info(
                "Duplicate webhook for transaction %s — skipping",
                payload.transaction_id,
            )
            return existing_tx, False

        transaction = Transaction(
            fineract_transaction_id=payload.transaction_id,
            fineract_account_id=payload.account_id,
            fineract_client_id=payload.client_id,
            transaction_type=payload.transaction_type,
            amount=payload.amount,
            currency=payload.currency or settings.default_currency,
            transaction_date=payload.transaction_date,
            counterparty_account_id=payload.counterparty_account_id,
            counterparty_name=payload.counterparty_name,
            description=payload.description,
            ip_address=payload.ip_address,
            user_agent=payload.user_agent,
            country_code=payload.country_code,
            geo_location=payload.geo_location,
            # WeBank actor context
            actor_type=payload.actor_type,
            agent_id=payload.agent_id,
            branch_id=payload.branch_id,
            merchant_id=payload.merchant_id,
            device_id=payload.device_id,
            kyc_level=payload.kyc_level,
            # Data quality
            data_quality_warnings=json.dumps(data_quality_warnings) if data_quality_warnings else None,
            raw_payload=json.dumps(payload.model_dump(), default=str),
        )
        self.db.add(transaction)
        await self.db.flush()
        logger.info(
            "Ingested transaction %s (amount=%.2f, type=%s, actor=%s)",
            transaction.fineract_transaction_id,
            transaction.amount,
            transaction.transaction_type.value,
            transaction.actor_type or "unknown",
        )
        return transaction, True

    async def update_risk_score(
        self,
        transaction_id: UUID,
        risk_score: float,
        anomaly_score: float | None = None,
        model_version: str | None = None,
        score_explanation: str | None = None,
    ) -> Transaction:
        """Update the risk score of a transaction and set risk level."""
        from app.core.config import settings

        result = await self.db.execute(
            select(Transaction).where(Transaction.id == transaction_id)
        )
        transaction = result.scalar_one()

        transaction.risk_score = risk_score
        transaction.anomaly_score = anomaly_score
        transaction.model_version = model_version
        transaction.score_explanation = score_explanation

        if risk_score >= 0.9:
            transaction.risk_level = RiskLevel.CRITICAL
        elif risk_score >= settings.risk_score_high:
            transaction.risk_level = RiskLevel.HIGH
        elif risk_score >= settings.risk_score_medium:
            transaction.risk_level = RiskLevel.MEDIUM
        else:
            transaction.risk_level = RiskLevel.LOW

        await self.db.flush()
        return transaction

    async def create_alert_if_needed(
        self,
        transaction: Transaction,
        risk_score: float,
        source: AlertSource,
        triggered_rules: list[str] | None = None,
    ) -> Alert | None:
        """Create an alert if the risk score exceeds the threshold."""
        from app.core.config import settings

        if risk_score < settings.risk_score_medium:
            return None

        alert = Alert(
            transaction_id=transaction.id,
            status=AlertStatus.PENDING,
            source=source,
            risk_score=risk_score,
            title=f"Suspicious {transaction.transaction_type.value}: "
            f"{transaction.currency} {transaction.amount:,.2f}",
            description=f"Transaction {transaction.fineract_transaction_id} "
            f"flagged by {source.value} with risk score {risk_score:.2f}",
            triggered_rules=json.dumps(triggered_rules) if triggered_rules else None,
        )
        self.db.add(alert)
        await self.db.flush()
        logger.warning(
            "Alert created for transaction %s (score=%.2f, source=%s)",
            transaction.fineract_transaction_id,
            risk_score,
            source.value,
        )
        return alert

    async def get_transaction(self, transaction_id: UUID) -> Transaction | None:
        result = await self.db.execute(
            select(Transaction).where(Transaction.id == transaction_id)
        )
        return result.scalar_one_or_none()

    async def list_transactions(
        self, page: int = 1, page_size: int = 50, risk_level: RiskLevel | None = None
    ) -> tuple[list[Transaction], int]:
        """List transactions with optional filtering and pagination."""
        query = select(Transaction)
        count_query = select(func.count(Transaction.id))

        if risk_level:
            query = query.where(Transaction.risk_level == risk_level)
            count_query = count_query.where(Transaction.risk_level == risk_level)

        query = query.order_by(Transaction.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        transactions = list(result.scalars().all())

        count_result = await self.db.execute(count_query)
        total = count_result.scalar_one()

        return transactions, total

    async def get_stats(self) -> TransactionStats:
        """Get dashboard statistics."""
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        total = (await self.db.execute(select(func.count(Transaction.id)))).scalar_one()

        flagged = (
            await self.db.execute(
                select(func.count(Transaction.id)).where(
                    Transaction.risk_level.in_([RiskLevel.HIGH, RiskLevel.CRITICAL])
                )
            )
        ).scalar_one()

        confirmed = (
            await self.db.execute(
                select(func.count(Alert.id)).where(
                    Alert.status == AlertStatus.CONFIRMED_FRAUD
                )
            )
        ).scalar_one()

        false_pos = (
            await self.db.execute(
                select(func.count(Alert.id)).where(
                    Alert.status == AlertStatus.FALSE_POSITIVE
                )
            )
        ).scalar_one()

        avg_score = (
            await self.db.execute(
                select(func.avg(Transaction.risk_score)).where(
                    Transaction.risk_score.isnot(None)
                )
            )
        ).scalar_one()

        today_count = (
            await self.db.execute(
                select(func.count(Transaction.id)).where(
                    Transaction.created_at >= today
                )
            )
        ).scalar_one()

        pending_alerts = (
            await self.db.execute(
                select(func.count(Alert.id)).where(Alert.status == AlertStatus.PENDING)
            )
        ).scalar_one()

        return TransactionStats(
            total_transactions=total,
            total_flagged=flagged,
            total_confirmed_fraud=confirmed,
            total_false_positives=false_pos,
            average_risk_score=float(avg_score) if avg_score else None,
            transactions_today=today_count,
            alerts_pending=pending_alerts,
        )

    async def get_account_history(
        self, account_id: str, window_minutes: int = 60
    ) -> list[Transaction]:
        """Get recent transactions for an account within a time window."""
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.fineract_account_id == account_id)
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.transaction_date.desc())
        )
        return list(result.scalars().all())

    async def get_agent_recent_transactions(
        self, agent_id: str, window_minutes: int = 60
    ) -> list[Transaction]:
        """Get all transactions processed by a specific agent within a time window.

        Used by agent-specific AML rules to detect structuring and float anomalies.
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.agent_id == agent_id)
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.transaction_date.desc())
        )
        return list(result.scalars().all())

    async def get_recent_withdrawals_by_client(
        self, client_id: str, window_minutes: int = 60
    ) -> list[Transaction]:
        """Get recent withdrawal transactions for a client across ALL accounts.

        Used for agent collusion detection: checks if a client deposited via one
        agent then immediately withdrew via a different agent.
        """
        from app.models.transaction import TransactionType

        cutoff = datetime.now(UTC) - timedelta(minutes=window_minutes)
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.fineract_client_id == client_id)
            .where(Transaction.transaction_type == TransactionType.WITHDRAWAL)
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.transaction_date.desc())
        )
        return list(result.scalars().all())

    async def get_recent_transactions_to_counterparty(
        self, counterparty_account_id: str, window_hours: int = 24
    ) -> list[Transaction]:
        """Get all transactions to a specific counterparty across all source accounts.

        Used for cross-account structuring detection: multiple accounts each sending
        sub-threshold amounts to the same ultimate beneficiary.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=window_hours)
        result = await self.db.execute(
            select(Transaction)
            .where(Transaction.counterparty_account_id == counterparty_account_id)
            .where(Transaction.transaction_date >= cutoff)
            .order_by(Transaction.transaction_date.desc())
        )
        return list(result.scalars().all())
