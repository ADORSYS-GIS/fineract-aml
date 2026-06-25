"""Tests for agent, merchant, and network typology rules."""

from datetime import UTC, datetime, timedelta

from app.models.transaction import TransactionType
from app.rules.engine import RuleEngine
from tests.conftest import FakeTransaction


def _agent_tx(**kwargs):
    defaults = dict(
        actor_type="agent",
        agent_id="AGENT-001",
        fineract_account_id="AGT-ACC-001",
        fineract_client_id="AGT-CLI-001",
        amount=500.0,
        transaction_type=TransactionType.DEPOSIT,
    )
    defaults.update(kwargs)
    return FakeTransaction(**defaults)


def _merchant_tx(**kwargs):
    defaults = dict(
        actor_type="merchant",
        merchant_id="MERCH-001",
        fineract_account_id="MRC-ACC-001",
        amount=500.0,
        transaction_type=TransactionType.DEPOSIT,
    )
    defaults.update(kwargs)
    return FakeTransaction(**defaults)


class TestAgentStructuringRule:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_no_structuring_below_count_threshold(self):
        tx = _agent_tx(amount=9500.0)
        # 4 prior structuring-range deposits (below the 5-count threshold)
        history = [_agent_tx(amount=9200.0, transaction_type=TransactionType.DEPOSIT) for _ in range(4)]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_structuring" not in triggered

    def test_structuring_triggers_at_threshold(self):
        """5 structuring-range deposits via same agent in 1h should trigger."""
        from app.core.config import settings

        # Use amounts in the (structuring_threshold, max_transaction_amount) range
        in_range = settings.structuring_threshold + 50  # e.g. 9550 with defaults
        tx = _agent_tx(amount=in_range)
        # 4 prior + current = 5
        history = [_agent_tx(amount=in_range, transaction_type=TransactionType.DEPOSIT) for _ in range(4)]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_structuring" in triggered

    def test_structuring_not_triggered_for_non_agent(self):
        """Rule must not run for customer transactions."""
        tx = FakeTransaction(amount=9500.0, transaction_type=TransactionType.DEPOSIT)
        history = [FakeTransaction(amount=9300.0, transaction_type=TransactionType.DEPOSIT) for _ in range(6)]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_structuring" not in triggered

    def test_large_amounts_outside_structuring_band_not_counted(self):
        """Deposits above max_transaction_amount do not count for agent structuring."""
        from app.core.config import settings

        tx = _agent_tx(amount=settings.max_transaction_amount + 100)
        history = [
            _agent_tx(amount=settings.max_transaction_amount + 50, transaction_type=TransactionType.DEPOSIT)
            for _ in range(6)
        ]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_structuring" not in triggered


class TestAgentFloatAnomalyRule:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_balanced_float_does_not_trigger(self):
        tx = _agent_tx(amount=10000.0, transaction_type=TransactionType.DEPOSIT)
        # Equal deposits and withdrawals
        deposits = [_agent_tx(amount=10000.0, transaction_type=TransactionType.DEPOSIT) for _ in range(3)]
        withdrawals = [_agent_tx(amount=10000.0, transaction_type=TransactionType.WITHDRAWAL) for _ in range(3)]
        result = self.engine.evaluate(tx, [], agent_history=deposits + withdrawals)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_float_anomaly" not in triggered

    def test_heavily_deposit_skewed_triggers(self):
        """>95% deposits with sufficient volume should trigger."""
        tx = _agent_tx(amount=100000.0, transaction_type=TransactionType.DEPOSIT)
        deposits = [_agent_tx(amount=100000.0, transaction_type=TransactionType.DEPOSIT) for _ in range(9)]
        withdrawals = [_agent_tx(amount=1000.0, transaction_type=TransactionType.WITHDRAWAL)]
        result = self.engine.evaluate(tx, [], agent_history=deposits + withdrawals)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_float_anomaly" in triggered

    def test_below_volume_minimum_does_not_trigger(self):
        """Even 100% deposit ratio should not trigger if total volume is tiny."""
        tx = _agent_tx(amount=100.0, transaction_type=TransactionType.DEPOSIT)
        deposits = [_agent_tx(amount=100.0, transaction_type=TransactionType.DEPOSIT) for _ in range(3)]
        result = self.engine.evaluate(tx, [], agent_history=deposits)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_float_anomaly" not in triggered


class TestAgentAccountFarmingRule:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_farming_triggers_above_threshold(self):
        """≥8 unique KYC-L1 customers served by same agent in 24h."""
        tx = _agent_tx(kyc_level=1, fineract_client_id="CLI-999")
        # 7 prior unique KYC-L1 clients + current = 8
        history = [
            _agent_tx(kyc_level=1, fineract_client_id=f"CLI-{i}")
            for i in range(7)
        ]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_account_farming" in triggered

    def test_farming_below_threshold_does_not_trigger(self):
        tx = _agent_tx(kyc_level=1, fineract_client_id="CLI-999")
        history = [
            _agent_tx(kyc_level=1, fineract_client_id=f"CLI-{i}")
            for i in range(5)
        ]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_account_farming" not in triggered

    def test_high_kyc_clients_not_counted(self):
        """Only KYC level 1 counts for account farming detection."""
        tx = _agent_tx(kyc_level=3, fineract_client_id="CLI-999")
        history = [
            _agent_tx(kyc_level=3, fineract_client_id=f"CLI-{i}")
            for i in range(10)
        ]
        result = self.engine.evaluate(tx, [], agent_history=history)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_account_farming" not in triggered


class TestAgentCustomerCollusionRule:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_collusion_triggers_when_client_withdraws_via_different_agent(self):
        tx = _agent_tx(
            transaction_type=TransactionType.DEPOSIT,
            counterparty_account_id="CLIENT-X",
        )
        # Client X withdrew via a DIFFERENT agent recently
        withdrawal = FakeTransaction(
            transaction_type=TransactionType.WITHDRAWAL,
            actor_type="agent",
            agent_id="AGENT-002",  # different agent
            fineract_client_id="CLIENT-X",
            counterparty_account_id="CLIENT-X",
        )
        result = self.engine.evaluate(
            tx, [], account_history_24h=[withdrawal]
        )
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_customer_collusion" in triggered

    def test_collusion_not_triggered_when_same_agent(self):
        tx = _agent_tx(
            transaction_type=TransactionType.DEPOSIT,
            counterparty_account_id="CLIENT-X",
        )
        # Withdrawal via SAME agent — not collusion
        withdrawal = FakeTransaction(
            transaction_type=TransactionType.WITHDRAWAL,
            actor_type="agent",
            agent_id="AGENT-001",  # same agent
            fineract_client_id="CLIENT-X",
        )
        result = self.engine.evaluate(
            tx, [], account_history_24h=[withdrawal]
        )
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_customer_collusion" not in triggered

    def test_collusion_not_checked_without_counterparty(self):
        tx = _agent_tx(transaction_type=TransactionType.DEPOSIT, counterparty_account_id=None)
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "agent_customer_collusion" not in triggered


class TestMerchantRules:
    def setup_method(self):
        self.engine = RuleEngine()

    def test_merchant_collection_account_triggers(self):
        from app.core.config import settings

        # Big inflow, no outgoing
        large_amount = settings.max_transaction_amount * 3
        tx = _merchant_tx(amount=large_amount)
        history_7d = [
            _merchant_tx(amount=large_amount, transaction_type=TransactionType.DEPOSIT)
            for _ in range(3)
        ]
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "merchant_collection_account" in triggered

    def test_merchant_collection_not_triggered_with_outgoing(self):
        from app.core.config import settings

        large_amount = settings.max_transaction_amount * 3
        tx = _merchant_tx(amount=large_amount)
        deposits = [_merchant_tx(amount=large_amount) for _ in range(3)]
        outgoing = [
            _merchant_tx(amount=5000.0, transaction_type=TransactionType.TRANSFER)
        ]
        result = self.engine.evaluate(tx, [], account_history_7d=deposits + outgoing)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "merchant_collection_account" not in triggered

    def test_high_value_anonymous_payment_triggers(self):
        from app.core.config import settings

        tx = _merchant_tx(
            amount=settings.anonymous_payment_alert_threshold + 1,
            counterparty_name=None,
            counterparty_account_id=None,
        )
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "high_value_anonymous_payment" in triggered

    def test_high_value_anonymous_not_triggered_when_identified(self):
        from app.core.config import settings

        tx = _merchant_tx(
            amount=settings.anonymous_payment_alert_threshold + 1,
            counterparty_name="Known Payer",
        )
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "high_value_anonymous_payment" not in triggered

    def test_merchant_rules_not_applied_to_agents(self):
        tx = _agent_tx()
        result = self.engine.evaluate(tx, [])
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "merchant_collection_account" not in triggered
        assert "high_value_anonymous_payment" not in triggered


class TestNetworkTypologyRules:
    def setup_method(self):
        self.engine = RuleEngine()

    # ── scatter_gather ─────────────────────────────────────────────────────────

    def test_scatter_gather_triggers_with_many_senders(self):
        from app.core.config import settings

        # 8 unique inbound deposits in 7d history
        history_7d = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                counterparty_account_id=f"SENDER-{i}",
            )
            for i in range(settings.scatter_gather_min_senders)
        ]
        # Current tx is a large outbound transfer (gather phase)
        tx = FakeTransaction(amount=50000.0, transaction_type=TransactionType.TRANSFER)
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "scatter_gather" in triggered

    def test_scatter_gather_not_triggered_on_deposit(self):
        from app.core.config import settings

        history_7d = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                counterparty_account_id=f"SENDER-{i}",
            )
            for i in range(settings.scatter_gather_min_senders)
        ]
        tx = FakeTransaction(amount=50000.0, transaction_type=TransactionType.DEPOSIT)
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "scatter_gather" not in triggered

    def test_scatter_gather_requires_7d_history(self):
        tx = FakeTransaction(amount=50000.0, transaction_type=TransactionType.TRANSFER)
        result = self.engine.evaluate(tx, [], account_history_7d=None)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "scatter_gather" not in triggered

    # ── bipartite_layering ─────────────────────────────────────────────────────

    def test_bipartite_layering_triggers_with_fan_in_and_fan_out(self):
        from app.core.config import settings

        fan = settings.bipartite_fan_threshold
        deposits = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                counterparty_account_id=f"SRC-{i}",
            )
            for i in range(fan)
        ]
        transfers = [
            FakeTransaction(
                transaction_type=TransactionType.TRANSFER,
                counterparty_account_id=f"DST-{i}",
            )
            for i in range(fan)
        ]
        tx = FakeTransaction(amount=1000.0, transaction_type=TransactionType.TRANSFER)
        result = self.engine.evaluate(tx, [], account_history_7d=deposits + transfers)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "bipartite_layering" in triggered

    def test_bipartite_layering_not_triggered_with_only_fan_in(self):
        from app.core.config import settings

        fan = settings.bipartite_fan_threshold
        deposits = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                counterparty_account_id=f"SRC-{i}",
            )
            for i in range(fan)
        ]
        tx = FakeTransaction(amount=1000.0, transaction_type=TransactionType.TRANSFER)
        result = self.engine.evaluate(tx, [], account_history_7d=deposits)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "bipartite_layering" not in triggered

    # ── stacking ───────────────────────────────────────────────────────────────

    def test_stacking_triggers_with_proportional_chain(self):

        now = datetime.now(UTC)
        # A→B (100k) → B→C (95k) → C→D (90k) all within stacking window
        history_7d = [
            FakeTransaction(
                amount=90000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=now - timedelta(minutes=20),
            ),
            FakeTransaction(
                amount=95000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=now - timedelta(minutes=10),
            ),
        ]
        tx = FakeTransaction(
            amount=100000.0,
            transaction_type=TransactionType.TRANSFER,
            transaction_date=now,
        )
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "stacking" in triggered

    def test_stacking_not_triggered_with_disproportionate_amounts(self):
        now = datetime.now(UTC)
        history_7d = [
            FakeTransaction(
                amount=1000.0,  # Very different from current 100k
                transaction_type=TransactionType.TRANSFER,
                transaction_date=now - timedelta(minutes=10),
            ),
        ]
        tx = FakeTransaction(
            amount=100000.0,
            transaction_type=TransactionType.TRANSFER,
            transaction_date=now,
        )
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "stacking" not in triggered

    def test_stacking_not_triggered_on_non_transfer(self):
        now = datetime.now(UTC)
        history_7d = [
            FakeTransaction(
                amount=90000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=now - timedelta(minutes=10),
            ),
            FakeTransaction(
                amount=95000.0,
                transaction_type=TransactionType.TRANSFER,
                transaction_date=now - timedelta(minutes=5),
            ),
        ]
        tx = FakeTransaction(
            amount=100000.0,
            transaction_type=TransactionType.DEPOSIT,  # Not a transfer
            transaction_date=now,
        )
        result = self.engine.evaluate(tx, [], account_history_7d=history_7d)
        triggered = [r.rule_name for r in result.triggered_rules]
        assert "stacking" not in triggered
