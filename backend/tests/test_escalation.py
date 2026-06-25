"""Tests for EscalationService — auto-escalation and SAR deadline enforcement.

Uses AsyncMock + SQLAlchemy stub objects to avoid needing a live database.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.alert import AlertStatus
from app.models.case import CaseStatus
from app.services.escalation_service import (
    _ESCALATION_HOURS,
    _ESCALATION_SCORE_THRESHOLD,
    _HIGH_ALERT_COUNT_THRESHOLD,
    _SAR_DEADLINE_DAYS,
    EscalationService,
)


def _make_case(
    hours_old: float = 30,
    status: CaseStatus = CaseStatus.OPEN,
    fineract_client_id: str = "CLI-001",
    sla_deadline=None,
    sar_document_path=None,
):
    created = datetime.now(UTC) - timedelta(hours=hours_old)
    return SimpleNamespace(
        id="CASE-001",
        case_number="AML-2025-001",
        status=status,
        fineract_client_id=fineract_client_id,
        created_at=created,
        sla_deadline=sla_deadline,
        sar_document_path=sar_document_path,
        escalation_reason=None,
    )


def _make_alert(risk_score: float, status: AlertStatus = AlertStatus.PENDING):
    return SimpleNamespace(risk_score=risk_score, status=status)


def _make_db_stub(cases=None, alerts=None):
    """Create a mock AsyncSession that returns preset query results."""
    db = AsyncMock()

    case_scalars = MagicMock()
    case_scalars.all.return_value = cases or []

    alert_scalars = MagicMock()
    alert_scalars.all.return_value = alerts or []

    case_result = MagicMock()
    case_result.scalars.return_value = case_scalars

    alert_result = MagicMock()
    alert_result.scalars.return_value = alert_scalars

    # First execute call returns cases, subsequent calls return alerts
    db.execute = AsyncMock(side_effect=[case_result, alert_result])
    db.flush = AsyncMock()
    return db


class TestAutoEscalatePendingCases:
    @pytest.mark.asyncio
    async def test_high_risk_stale_case_is_escalated(self):
        case = _make_case(hours_old=_ESCALATION_HOURS + 5)
        high_risk_alert = _make_alert(risk_score=_ESCALATION_SCORE_THRESHOLD + 0.01)
        db = _make_db_stub(cases=[case], alerts=[high_risk_alert])

        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()

        assert count == 1
        assert case.status == CaseStatus.ESCALATED
        assert case.sla_deadline is not None
        # SLA deadline should be ~30 days from now
        delta = case.sla_deadline - datetime.now(UTC)
        assert abs(delta.total_seconds() - _SAR_DEADLINE_DAYS * 86400) < 60

    @pytest.mark.asyncio
    async def test_fresh_case_not_escalated_even_with_high_score(self):
        """Cases that are not stale should not appear in the query results."""
        # The query already filters out fresh cases (Case.created_at <= cutoff),
        # so simulate that by returning an empty cases list.
        db = _make_db_stub(cases=[], alerts=[])
        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()
        assert count == 0

    @pytest.mark.asyncio
    async def test_many_high_alerts_triggers_escalation_below_score_threshold(self):
        case = _make_case(hours_old=_ESCALATION_HOURS + 5)
        # Score below threshold (0.89) but 3+ HIGH alerts
        alerts = [_make_alert(risk_score=0.65) for _ in range(_HIGH_ALERT_COUNT_THRESHOLD)]
        db = _make_db_stub(cases=[case], alerts=alerts)

        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()

        assert count == 1
        assert case.status == CaseStatus.ESCALATED
        assert case.escalation_reason is not None

    @pytest.mark.asyncio
    async def test_case_without_client_id_skipped(self):
        case = _make_case(hours_old=50, fineract_client_id=None)
        # Even with good DB results, a None client_id should skip escalation
        db = _make_db_stub(cases=[case], alerts=[])
        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()
        assert count == 0

    @pytest.mark.asyncio
    async def test_false_positive_alerts_not_counted(self):
        """The DB query filters FALSE_POSITIVE alerts via WHERE clause.
        When all alerts are dismissed the query returns nothing → no escalation."""
        case = _make_case(hours_old=50)
        # Simulate the SQL WHERE filter: DB returns [] because all were FALSE_POSITIVE
        db = _make_db_stub(cases=[case], alerts=[])
        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()
        # No qualifying alerts after DB filter → not escalated
        assert count == 0

    @pytest.mark.asyncio
    async def test_no_alerts_for_client_means_no_escalation(self):
        case = _make_case(hours_old=50)
        db = _make_db_stub(cases=[case], alerts=[])
        service = EscalationService(db)
        count = await service.auto_escalate_pending_cases()
        assert count == 0


class TestCheckSarDeadlines:
    @pytest.mark.asyncio
    async def test_past_deadline_without_sar_is_flagged(self):
        past_deadline = datetime.now(UTC) - timedelta(days=1)
        case = _make_case(
            status=CaseStatus.ESCALATED,
            sla_deadline=past_deadline,
            sar_document_path=None,
        )

        db = AsyncMock()
        result_stub = MagicMock()
        result_stub.scalars.return_value.all.return_value = [case]
        db.execute = AsyncMock(return_value=result_stub)

        service = EscalationService(db)
        count = await service.check_sar_deadlines()

        assert count == 1

    @pytest.mark.asyncio
    async def test_case_with_sar_filed_not_counted(self):
        """Cases that already have a SAR document don't appear in the query results."""
        # Simulate DB filtering out cases with sar_document_path IS NOT NULL
        db = AsyncMock()
        result_stub = MagicMock()
        result_stub.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result_stub)

        service = EscalationService(db)
        count = await service.check_sar_deadlines()

        assert count == 0

    @pytest.mark.asyncio
    async def test_no_overdue_cases_returns_zero(self):
        db = AsyncMock()
        result_stub = MagicMock()
        result_stub.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result_stub)

        service = EscalationService(db)
        count = await service.check_sar_deadlines()

        assert count == 0
