"""Case escalation service — auto-escalates stale high-risk cases.

Implements Jube-inspired workflow automation:
- Cases open > 24h with risk score > 0.9 auto-escalate
- ESCALATED cases open > 30 days get a SAR deadline flag (COBAC requirement)
"""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertStatus
from app.models.case import Case, CaseStatus
from app.models.transaction import Transaction

logger = logging.getLogger(__name__)

# How long a case can sit OPEN/INVESTIGATING before auto-escalation
_ESCALATION_HOURS = 24
# COBAC SAR filing deadline for ESCALATED cases (30 days)
_SAR_DEADLINE_DAYS = 30
# Risk score threshold for auto-escalation
_ESCALATION_SCORE_THRESHOLD = 0.9
# Number of HIGH alerts needed to trigger escalation even below score threshold
_HIGH_ALERT_COUNT_THRESHOLD = 3


class EscalationService:
    """Manages automatic case escalation and SLA deadline enforcement."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def auto_escalate_pending_cases(self) -> int:
        """Escalate cases that have been open too long with high-risk activity.

        Returns:
            Number of cases escalated.
        """
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=_ESCALATION_HOURS)

        result = await self.db.execute(
            select(Case).where(
                Case.status.in_([CaseStatus.OPEN, CaseStatus.INVESTIGATING]),
                Case.created_at <= cutoff,
            )
        )
        cases = list(result.scalars().all())

        escalated = 0
        for case in cases:
            reason = await self._escalation_reason(case, now)
            if reason:
                case.status = CaseStatus.ESCALATED
                case.escalation_reason = reason
                case.sla_deadline = now + timedelta(days=_SAR_DEADLINE_DAYS)
                escalated += 1
                logger.info(
                    "Case %s auto-escalated: %s (SLA deadline: %s)",
                    case.case_number,
                    reason,
                    case.sla_deadline.date(),
                )

        if escalated:
            await self.db.flush()
        return escalated

    async def check_sar_deadlines(self) -> int:
        """Flag ESCALATED cases that are approaching or past the SAR filing deadline.

        Returns:
            Number of cases past their SLA deadline.
        """
        now = datetime.now(UTC)

        result = await self.db.execute(
            select(Case).where(
                Case.status == CaseStatus.ESCALATED,
                Case.sla_deadline <= now,
                Case.sar_document_path == None,  # noqa: E711 — SAR not yet filed
            )
        )
        overdue = list(result.scalars().all())

        for case in overdue:
            logger.warning(
                "Case %s is past SAR SLA deadline %s — immediate COBAC filing required",
                case.case_number,
                case.sla_deadline.date() if case.sla_deadline else "N/A",
            )

        return len(overdue)

    async def _escalation_reason(self, case: Case, now: datetime) -> str | None:
        """Determine if a case should be escalated and why."""
        client_id = case.fineract_client_id
        if not client_id:
            return None

        # Check for very high risk alerts
        result = await self.db.execute(
            select(Alert)
            .join(Transaction, Alert.transaction_id == Transaction.id)
            .where(
                Transaction.fineract_client_id == client_id,
                Alert.status.notin_([AlertStatus.FALSE_POSITIVE, AlertStatus.DISMISSED]),
            )
            .order_by(Alert.risk_score.desc())
            .limit(10)
        )
        alerts = list(result.scalars().all())

        if not alerts:
            return None

        max_score = max(a.risk_score for a in alerts)
        high_count = sum(1 for a in alerts if a.risk_score >= 0.6)

        if max_score >= _ESCALATION_SCORE_THRESHOLD:
            open_hours = (now - case.created_at.replace(tzinfo=UTC)).total_seconds() / 3600
            return (
                f"Score critique ({max_score:.2f}) — dossier ouvert depuis {open_hours:.0f}h "
                f"sans résolution"
            )

        if high_count >= _HIGH_ALERT_COUNT_THRESHOLD:
            return (
                f"{high_count} alertes à haut risque sans résolution après "
                f"{_ESCALATION_HOURS}h"
            )

        return None
