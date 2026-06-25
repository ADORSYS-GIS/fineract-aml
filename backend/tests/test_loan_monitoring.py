"""Tests for post-disbursement loan behavior analysis (pure logic, no DB)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.models.transaction import TransactionType
from app.tasks.loan_monitoring import _analyze_post_disbursement
from tests.conftest import FakeTransaction


def _make_watch(disbursed_amount: float = 100_000.0, currency: str = "XAF"):
    return SimpleNamespace(
        disbursed_amount=disbursed_amount,
        disbursed_at=datetime.now(UTC) - timedelta(hours=2),
        fineract_account_id="ACC-001",
        currency=currency,
        findings_json=None,
    )


def _make_settings(**overrides):
    defaults = dict(
        loan_run_threshold=0.8,
        loan_immediate_cashout_minutes=30,
        structuring_threshold=9000.0,
        loan_dispersal_counterparty_min=5,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestLoanAndRun:
    def test_loan_and_run_triggers_above_threshold(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings()
        # Transfer 85% of disbursed
        post_txns = [
            FakeTransaction(
                amount=85_000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id="ACC-OTHER",
                transaction_date=watch.disbursed_at + timedelta(hours=1),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["flagged"] is True
        assert "loan_and_run" in result["triggered_patterns"]
        assert result["severity"] >= 0.9

    def test_loan_and_run_not_triggered_below_threshold(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings()
        # Transfer only 50%
        post_txns = [
            FakeTransaction(
                amount=50_000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id="ACC-OTHER",
                transaction_date=watch.disbursed_at + timedelta(hours=1),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "loan_and_run" not in result["triggered_patterns"]

    def test_no_post_disbursement_transactions(self):
        watch = _make_watch()
        settings = _make_settings()
        result = _analyze_post_disbursement(watch, [], settings)
        assert result["flagged"] is False
        assert result["triggered_patterns"] == []


class TestImmediateCashOut:
    def test_immediate_cashout_triggers_within_window(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(loan_immediate_cashout_minutes=30)
        # Withdrawal of 82% within 30 min
        post_txns = [
            FakeTransaction(
                amount=82_000.0,
                transaction_type=TransactionType.WITHDRAWAL,
                transaction_date=watch.disbursed_at + timedelta(minutes=15),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["flagged"] is True
        assert "immediate_cash_out" in result["triggered_patterns"]
        assert result["severity"] >= 0.95

    def test_immediate_cashout_not_triggered_after_window(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(loan_immediate_cashout_minutes=30)
        # Withdrawal 2 hours later — outside window
        post_txns = [
            FakeTransaction(
                amount=90_000.0,
                transaction_type=TransactionType.WITHDRAWAL,
                transaction_date=watch.disbursed_at + timedelta(hours=2),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "immediate_cash_out" not in result["triggered_patterns"]

    def test_immediate_cashout_not_triggered_below_80_percent(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(loan_immediate_cashout_minutes=30)
        # Only 50% withdrawal within window
        post_txns = [
            FakeTransaction(
                amount=50_000.0,
                transaction_type=TransactionType.WITHDRAWAL,
                transaction_date=watch.disbursed_at + timedelta(minutes=10),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "immediate_cash_out" not in result["triggered_patterns"]


class TestPostDisbursementStructuring:
    def test_structuring_triggers_with_3_small_transfers(self):
        watch = _make_watch(disbursed_amount=30_000.0)
        settings = _make_settings(structuring_threshold=9000.0)
        # 3 transfers each below threshold summing to >70% of loan
        post_txns = [
            FakeTransaction(
                amount=8000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id=f"ACC-{i}",
                transaction_date=watch.disbursed_at + timedelta(hours=i + 1),
            )
            for i in range(3)
        ]
        # 3 × 8000 = 24000 > 70% of 30000 (21000)
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["flagged"] is True
        assert "post_disbursement_structuring" in result["triggered_patterns"]

    def test_structuring_not_triggered_below_sum_threshold(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(structuring_threshold=9000.0)
        # 3 small transfers but sum is only 24k < 70% of 100k
        post_txns = [
            FakeTransaction(
                amount=8000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=watch.disbursed_at + timedelta(hours=i + 1),
            )
            for i in range(3)
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "post_disbursement_structuring" not in result["triggered_patterns"]

    def test_structuring_not_triggered_with_only_2_transfers(self):
        watch = _make_watch(disbursed_amount=20_000.0)
        settings = _make_settings(structuring_threshold=9000.0)
        post_txns = [
            FakeTransaction(
                amount=8000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=watch.disbursed_at + timedelta(hours=i + 1),
            )
            for i in range(2)
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "post_disbursement_structuring" not in result["triggered_patterns"]


class TestCrossAgentDispersal:
    def test_dispersal_triggers_with_many_counterparties(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(loan_dispersal_counterparty_min=5)
        # 5 transfers to unique counterparties
        post_txns = [
            FakeTransaction(
                amount=20_000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id=f"DEST-{i}",
                transaction_date=watch.disbursed_at + timedelta(hours=1),
            )
            for i in range(5)
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["flagged"] is True
        assert "cross_agent_dispersal" in result["triggered_patterns"]

    def test_dispersal_not_triggered_below_threshold(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(loan_dispersal_counterparty_min=5)
        # Only 3 unique counterparties
        post_txns = [
            FakeTransaction(
                amount=30_000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id=f"DEST-{i}",
                transaction_date=watch.disbursed_at + timedelta(hours=1),
            )
            for i in range(3)
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert "cross_agent_dispersal" not in result["triggered_patterns"]


class TestSeverityAndMultiplePatterns:
    def test_multiple_patterns_detected_simultaneously(self):
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings(
            loan_run_threshold=0.8,
            loan_immediate_cashout_minutes=30,
            loan_dispersal_counterparty_min=5,
            structuring_threshold=9000.0,
        )
        # Large transfers to 5 unique counterparties (triggers dispersal + loan-and-run)
        post_txns = [
            FakeTransaction(
                amount=17_000.0,
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id=f"DEST-{i}",
                transaction_date=watch.disbursed_at + timedelta(hours=1),
            )
            for i in range(5)
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["flagged"] is True
        # Both loan_and_run (85k > 80%) and cross_agent_dispersal should trigger
        assert "cross_agent_dispersal" in result["triggered_patterns"]

    def test_severity_is_max_of_triggered_patterns(self):
        """Immediate cash-out (0.95) should dominate over loan-and-run (0.9)."""
        watch = _make_watch(disbursed_amount=100_000.0)
        settings = _make_settings()
        post_txns = [
            FakeTransaction(
                amount=85_000.0,
                transaction_type=TransactionType.WITHDRAWAL,
                transaction_date=watch.disbursed_at + timedelta(minutes=10),
            )
        ]
        result = _analyze_post_disbursement(watch, post_txns, settings)
        assert result["severity"] == pytest.approx(0.95)
