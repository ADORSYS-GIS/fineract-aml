"""Tests for the AML rule engine."""

from datetime import UTC, datetime

from app.rules.engine import RuleEngine
from tests.conftest import FakeTransaction


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_normal_transaction_no_triggers(self):
        tx = FakeTransaction(amount=500.0)
        result = self.engine.evaluate(tx, [])
        assert len(result.triggered_rules) == 0
        assert result.combined_score == 0.0

    def test_large_amount_triggers(self):
        tx = FakeTransaction(amount=15000.0)
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "large_amount" in triggered

    def test_structuring_triggers(self):
        tx = FakeTransaction(amount=9700.0)
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "structuring" in triggered

    def test_round_number_triggers(self):
        tx = FakeTransaction(amount=5000.0)
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "round_number" in triggered

    def test_unusual_hours_triggers(self):
        tx = FakeTransaction(
            amount=500.0,
            transaction_date=datetime(2025, 6, 15, 3, 0, tzinfo=UTC),
        )
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "unusual_hours" in triggered

    def test_rapid_transactions_triggers(self):
        tx = FakeTransaction(amount=500.0)
        history = [FakeTransaction(amount=100.0) for _ in range(12)]
        result = self.engine.evaluate(tx, history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "rapid_transactions" in triggered

    def test_multiple_rules_increase_score(self):
        # Large + round + unusual hours
        tx = FakeTransaction(
            amount=10000.0,
            transaction_date=datetime(2025, 6, 15, 3, 0, tzinfo=UTC),
        )
        result = self.engine.evaluate(tx, [])
        assert len(result.triggered_rules) >= 2
        assert result.combined_score > 0.3

    def test_combined_score_capped_at_1(self):
        tx = FakeTransaction(amount=50000.0)
        history = [FakeTransaction(amount=20000.0) for _ in range(20)]
        result = self.engine.evaluate(tx, history)
        assert result.combined_score <= 1.0

    def test_new_ip_triggers(self):
        tx = FakeTransaction(amount=500.0, ip_address="10.0.0.99")
        history = [FakeTransaction(amount=100.0, ip_address="192.168.1.1") for _ in range(3)]
        result = self.engine.evaluate(tx, history, account_history_24h=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "new_ip_address" in triggered

    def test_known_ip_does_not_trigger(self):
        tx = FakeTransaction(amount=500.0, ip_address="192.168.1.1")
        history = [FakeTransaction(amount=100.0, ip_address="192.168.1.1") for _ in range(3)]
        result = self.engine.evaluate(tx, history, account_history_24h=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "new_ip_address" not in triggered

    def test_no_ip_does_not_trigger(self):
        tx = FakeTransaction(amount=500.0)
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "new_ip_address" not in triggered
