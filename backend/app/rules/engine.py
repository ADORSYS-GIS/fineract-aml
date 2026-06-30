"""Rule engine — deterministic AML detection rules.

These rules run on every transaction before ML scoring. They catch
obvious suspicious patterns that don't need machine learning:
- Large transactions
- Rapid successive transactions (structuring)
- Round-number transactions just below reporting thresholds
- New accounts with high activity
- Unusual transaction times
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app.core.config import settings
from app.models.transaction import Transaction

logger = logging.getLogger(__name__)

# WAT = West Africa Time (UTC+1), used for Cameroon local time
WAT = timezone(timedelta(hours=1))


def _normalize_dt(dt: datetime) -> datetime:
    """Strip timezone for comparison to handle mixed aware/naive datetimes."""
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


@dataclass
class RuleResult:
    """Result from a single rule evaluation."""

    rule_name: str
    category: str
    triggered: bool
    severity: float  # 0.0 to 1.0
    details: str


@dataclass
class RuleEngineResult:
    """Aggregated result from all rules."""

    results: list[RuleResult] = field(default_factory=list)

    @property
    def triggered_rules(self) -> list[RuleResult]:
        return [r for r in self.results if r.triggered]

    @property
    def max_severity(self) -> float:
        triggered = self.triggered_rules
        return max(r.severity for r in triggered) if triggered else 0.0

    @property
    def combined_score(self) -> float:
        """Weighted combination of all triggered rule severities."""
        triggered = self.triggered_rules
        if not triggered:
            return 0.0
        # Use a weighted average that increases with more triggered rules
        base_score = sum(r.severity for r in triggered) / len(triggered)
        rule_count_bonus = min(len(triggered) * 0.05, 0.2)
        return min(base_score + rule_count_bonus, 1.0)

    @property
    def rule_names(self) -> list[str]:
        return [r.rule_name for r in self.triggered_rules]


class RuleEngine:
    """Evaluates a transaction against all AML detection rules."""

    def evaluate(
        self,
        transaction: Transaction,
        account_history: list[Transaction],
        account_history_24h: list[Transaction] | None = None,
        account_history_7d: list[Transaction] | None = None,
        counterparty_history: list[Transaction] | None = None,
        agent_history: list[Transaction] | None = None,
    ) -> RuleEngineResult:
        """Run all rules against a transaction.

        Args:
            transaction: The transaction to evaluate.
            account_history: Short-window history (1h) for velocity rules.
            account_history_24h: Longer-window history (24h) for IP-based rules.
                Falls back to account_history if not provided.
            account_history_7d: 7-day history for trend rules. Currently accepted
                for future use; not yet consumed by individual rules.
            counterparty_history: Recent transactions received by the counterparty
                (used for cross-account structuring detection).
            agent_history: Recent transactions served by the same agent (24h window).
                Used for agent-specific fraud rules when actor_type == "agent".
        """
        result = RuleEngineResult()

        result.results.append(self._check_large_amount(transaction))
        result.results.append(self._check_structuring(transaction))
        result.results.append(self._check_rapid_transactions(transaction, account_history))
        result.results.append(self._check_round_number(transaction))
        result.results.append(self._check_unusual_hours(transaction))
        result.results.append(self._check_velocity(transaction, account_history))
        result.results.append(
            self._check_new_ip(transaction, account_history_24h or account_history)
        )

        # Transfer-specific rules (only for transfer transactions)
        history_24h = account_history_24h or account_history
        result.results.append(self._check_circular_transfer(transaction, history_24h))
        result.results.append(self._check_new_counterparty(transaction, history_24h))
        result.results.append(self._check_rapid_pair_transfers(transaction, history_24h))

        # Cross-account structuring (requires counterparty history)
        if counterparty_history is not None:
            cross_acct = self._check_cross_account_structuring(
                transaction, counterparty_history, settings.structuring_threshold
            )
            if cross_acct is not None:
                result.results.append(cross_acct)

        # Agent-specific rules (only when actor_type == "agent" and agent history provided)
        actor = getattr(transaction, "actor_type", None)
        if actor == "agent" and agent_history is not None:
            result.results.append(self._check_agent_rapid_transactions(transaction, agent_history))
            result.results.append(self._check_agent_float_volume(transaction, agent_history))

        if result.triggered_rules:
            logger.info(
                "Transaction %s triggered %d rules: %s",
                transaction.fineract_transaction_id,
                len(result.triggered_rules),
                result.rule_names,
            )

        return result

    def _check_large_amount(self, tx: Transaction) -> RuleResult:
        """Flag transactions above the configured threshold."""
        triggered = tx.amount >= settings.max_transaction_amount
        severity = min(tx.amount / (settings.max_transaction_amount * 2), 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="large_amount",
            category="amount",
            triggered=triggered,
            severity=severity,
            details=f"Amount {tx.currency} {tx.amount:,.2f} exceeds threshold "
            f"{tx.currency} {settings.max_transaction_amount:,.2f}",
        )

    def _check_structuring(self, tx: Transaction) -> RuleResult:
        """Detect structuring — amounts just below the reporting threshold.

        Criminals split large transactions into smaller ones to avoid
        regulatory reporting thresholds (e.g., $10K in the US).
        """
        threshold = settings.structuring_threshold
        upper = settings.max_transaction_amount
        triggered = threshold <= tx.amount < upper
        severity = 0.7 if triggered else 0.0
        return RuleResult(
            rule_name="structuring",
            category="pattern",
            triggered=triggered,
            severity=severity,
            details=f"Amount {tx.currency} {tx.amount:,.2f} is between "
            f"{tx.currency} {threshold:,.2f} and {tx.currency} {upper:,.2f} "
            f"(potential structuring)",
        )

    def _check_rapid_transactions(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Flag accounts with too many transactions in a short window."""
        count = len(history)
        triggered = count >= settings.rapid_transaction_count
        severity = min(count / (settings.rapid_transaction_count * 2), 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="rapid_transactions",
            category="velocity",
            triggered=triggered,
            severity=severity,
            details=f"Account has {count} transactions in the last "
            f"{settings.rapid_transaction_window_minutes} minutes "
            f"(threshold: {settings.rapid_transaction_count})",
        )

    def _check_round_number(self, tx: Transaction) -> RuleResult:
        """Flag round-number transactions (e.g., exactly $5000, $1000).

        Legitimate transactions tend to have odd amounts; round numbers
        can indicate manufactured transactions.
        """
        is_round = tx.amount >= 1000 and tx.amount % 1000 == 0
        return RuleResult(
            rule_name="round_number",
            category="pattern",
            triggered=is_round,
            severity=0.3 if is_round else 0.0,
            details=f"Round-number transaction: {tx.currency} {tx.amount:,.2f}",
        )

    def _check_unusual_hours(self, tx: Transaction) -> RuleResult:
        """Flag transactions outside normal business hours (2-5 AM local WAT)."""
        if isinstance(tx.transaction_date, datetime):
            # Convert UTC to WAT (UTC+1) for Cameroon
            local_hour = tx.transaction_date.astimezone(WAT).hour
        else:
            local_hour = 12
        is_unusual = 2 <= local_hour <= 5
        return RuleResult(
            rule_name="unusual_hours",
            category="timing",
            triggered=is_unusual,
            severity=0.4 if is_unusual else 0.0,
            details=f"Transaction at {local_hour}:00 WAT (unusual hours: 2-5 AM)",
        )

    def _check_velocity(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Flag high cumulative volume in the time window."""
        total_volume = sum(t.amount for t in history) + tx.amount
        threshold = settings.max_transaction_amount * 3
        triggered = total_volume >= threshold
        severity = min(total_volume / (threshold * 2), 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="high_velocity_volume",
            category="velocity",
            triggered=triggered,
            severity=severity,
            details=f"Cumulative volume {tx.currency} {total_volume:,.2f} in "
            f"{settings.rapid_transaction_window_minutes} minutes "
            f"(threshold: {tx.currency} {threshold:,.2f})",
        )

    def _check_new_ip(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Flag transactions from an IP address not previously seen for this account."""
        current_ip = getattr(tx, "ip_address", None)
        if not current_ip:
            return RuleResult(
                rule_name="new_ip_address",
                category="geographic",
                triggered=False,
                severity=0.0,
                details="No IP address available",
            )
        known_ips = {
            getattr(t, "ip_address", None)
            for t in history
            if getattr(t, "ip_address", None)
        }
        is_new = current_ip not in known_ips and len(known_ips) > 0
        return RuleResult(
            rule_name="new_ip_address",
            category="geographic",
            triggered=is_new,
            severity=0.5 if is_new else 0.0,
            details=f"IP {current_ip} {'is new' if is_new else 'is known'} for account "
            f"(known IPs: {len(known_ips)})",
        )

    def _check_circular_transfer(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Detect circular transfer patterns (A->B->A) indicating layering.

        Flags only when this is an INCOMING transfer and we previously sent
        a similar amount to the same counterparty within 24 h — classic
        round-trip layering.  Does NOT fire on legitimate repeat outbound
        payments (e.g. rent) to avoid false positives.
        """
        tx_type = getattr(tx, "transaction_type", None)
        tx_type_val = tx_type.value if hasattr(tx_type, "value") else str(tx_type)
        counterparty = getattr(tx, "counterparty_account_id", None)

        # Circular layering: receive from party you recently sent to (A sends to B, B sends back to A)
        # Only flag if this looks like an inbound that matches a recent outbound
        is_incoming = tx_type_val in ("DEPOSIT", "TRANSFER_IN", "TRANSFER", "deposit", "transfer_in", "transfer")

        if not is_incoming or not counterparty:
            return RuleResult(
                rule_name="circular_transfer",
                category="transfer",
                triggered=False,
                severity=0.0,
                details="Not an incoming transfer or no counterparty",
            )

        _outbound_types = {"WITHDRAWAL", "TRANSFER_OUT", "TRANSFER", "withdrawal", "transfer_out", "transfer"}

        def _tx_type_val(t: Transaction) -> str:
            tt = getattr(t, "transaction_type", "")
            return tt.value if hasattr(tt, "value") else str(tt)

        recent_outbound_to_same = [
            t for t in history
            if (
                getattr(t, "counterparty_account_id", None) == counterparty
                and _tx_type_val(t) in _outbound_types
                and abs(
                    _normalize_dt(tx.transaction_date) - _normalize_dt(t.transaction_date)
                ).total_seconds() < 86400
                and abs(t.amount - tx.amount) / max(tx.amount, 1) < 0.2  # within 20% amount
                and t.id != tx.id
            )
        ]
        is_circular = len(recent_outbound_to_same) > 0
        return RuleResult(
            rule_name="circular_transfer",
            category="transfer",
            triggered=is_circular,
            severity=0.7 if is_circular else 0.0,
            details=(
                f"Circular transfer detected: received from counterparty {counterparty} "
                f"after {len(recent_outbound_to_same)} recent outbound(s) to same party within 24h"
                if is_circular
                else f"No circular pattern with counterparty {counterparty}"
            ),
        )

    def _check_new_counterparty(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Flag first-time transfers to an unknown counterparty account."""
        tx_type = getattr(tx, "transaction_type", None)
        tx_type_val = tx_type.value if hasattr(tx_type, "value") else str(tx_type)
        counterparty = getattr(tx, "counterparty_account_id", None)

        if tx_type_val != "transfer" or not counterparty:
            return RuleResult(
                rule_name="new_counterparty_transfer",
                category="transfer",
                triggered=False,
                severity=0.0,
                details="Not a transfer or no counterparty",
            )

        known_counterparties = {
            getattr(t, "counterparty_account_id", None)
            for t in history
            if getattr(t, "counterparty_account_id", None) and t.id != tx.id
        }
        is_new = counterparty not in known_counterparties and len(known_counterparties) > 0
        return RuleResult(
            rule_name="new_counterparty_transfer",
            category="transfer",
            triggered=is_new,
            severity=0.4 if is_new else 0.0,
            details=f"Transfer to {'new' if is_new else 'known'} counterparty "
            f"{counterparty} (known: {len(known_counterparties)})",
        )

    def _check_rapid_pair_transfers(
        self, tx: Transaction, history: list[Transaction]
    ) -> RuleResult:
        """Flag multiple transfers between the same pair of accounts in 24h."""
        tx_type = getattr(tx, "transaction_type", None)
        tx_type_val = tx_type.value if hasattr(tx_type, "value") else str(tx_type)
        counterparty = getattr(tx, "counterparty_account_id", None)

        if tx_type_val != "transfer" or not counterparty:
            return RuleResult(
                rule_name="rapid_pair_transfers",
                category="transfer",
                triggered=False,
                severity=0.0,
                details="Not a transfer or no counterparty",
            )

        pair_transfers = [
            t for t in history
            if getattr(t, "counterparty_account_id", None) == counterparty
            and t.id != tx.id
        ]
        count = len(pair_transfers) + 1  # include current
        triggered = count >= 3
        severity = min(count / 6.0, 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="rapid_pair_transfers",
            category="transfer",
            triggered=triggered,
            severity=severity,
            details=f"{count} transfers with counterparty {counterparty} in 24h "
            f"(threshold: 3)",
        )

    def _check_cross_account_structuring(
        self, tx: Transaction, counterparty_history: list[Transaction], threshold: float
    ) -> RuleResult | None:
        """
        Detect multiple accounts sending sub-threshold amounts to the same beneficiary.
        Classic CEMAC/mobile money smurfing typology.
        """
        if not counterparty_history or len(counterparty_history) < 3:
            return None

        # Count distinct accounts sending to same counterparty recently
        recent_senders = set(
            t.fineract_account_id for t in counterparty_history
            if t.fineract_account_id != tx.fineract_account_id
        )
        total_inflow = sum(t.amount for t in counterparty_history)

        if len(recent_senders) >= 3 and total_inflow > threshold * 2:
            return RuleResult(
                rule_name="cross_account_structuring",
                category="pattern",
                triggered=True,
                severity=0.75,
                details=(
                    f"{len(recent_senders)} distinct accounts sent "
                    f"{total_inflow:,.0f} XAF to same beneficiary (structuring via multiple accounts)"
                ),
            )
        return None

    def _check_agent_rapid_transactions(
        self, tx: Transaction, agent_history: list[Transaction]
    ) -> RuleResult:
        """Flag agents processing more transactions than the agent threshold in 1h."""
        count = len(agent_history)
        threshold = settings.rapid_transaction_count_agent
        triggered = count >= threshold
        severity = min(count / (threshold * 2), 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="agent_rapid_transactions",
            category="agent",
            triggered=triggered,
            severity=severity,
            details=(
                f"Agent {tx.agent_id} processed {count} transactions in 24h "
                f"(threshold: {threshold})"
            ),
        )

    def _check_agent_float_volume(
        self, tx: Transaction, agent_history: list[Transaction]
    ) -> RuleResult:
        """Flag agents whose 24h float volume suggests imbalance / layering."""
        total = sum(t.amount for t in agent_history) + tx.amount
        threshold = settings.agent_float_volume_minimum
        triggered = total >= threshold
        severity = min(total / (threshold * 3), 1.0) if triggered else 0.0
        return RuleResult(
            rule_name="agent_float_volume",
            category="agent",
            triggered=triggered,
            severity=severity,
            details=(
                f"Agent {tx.agent_id} float volume {total:,.0f} XAF in 24h "
                f"(threshold: {threshold:,.0f} XAF)"
            ),
        )
