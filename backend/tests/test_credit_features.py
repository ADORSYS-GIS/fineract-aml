"""Tests for the credit feature extractor."""

import unittest
from datetime import UTC, datetime, timedelta

from app.features.credit_extractor import CREDIT_FEATURE_NAMES, CreditFeatureExtractor
from app.models.transaction import TransactionType
from tests.conftest import FakeTransaction


class TestCreditFeatureExtractor(unittest.TestCase):
    """Test customer-level credit feature extraction."""

    def test_feature_count_matches_names(self):
        """Feature vector length must match CREDIT_FEATURE_NAMES."""
        tx = FakeTransaction(
            transaction_type=TransactionType.DEPOSIT,
            amount=1000,
            transaction_date=datetime.now(UTC) - timedelta(days=10),
        )
        features = CreditFeatureExtractor.extract([tx], 0, None, 30)
        assert len(features) == len(CREDIT_FEATURE_NAMES)

    def test_empty_transactions_returns_zeros(self):
        """Empty history should return all-zero features."""
        features = CreditFeatureExtractor.extract([], 0, None, 0)
        assert len(features) == len(CREDIT_FEATURE_NAMES)
        assert all(f == 0.0 for f in features)

    def test_deposit_features(self):
        """Deposits should contribute to deposit-related features."""
        now = datetime.now(UTC)
        txs = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                amount=5000,
                transaction_date=now - timedelta(days=i * 10),
            )
            for i in range(6)
        ]
        features = CreditFeatureExtractor.extract(txs, 0, None, 60)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["avg_monthly_deposits"] > 0
        assert feat_dict["max_single_deposit"] == 5000.0

    def test_withdrawal_features(self):
        """Withdrawals should contribute to withdrawal features."""
        now = datetime.now(UTC)
        txs = [
            FakeTransaction(
                transaction_type=TransactionType.WITHDRAWAL,
                amount=2000,
                transaction_date=now - timedelta(days=5),
            ),
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                amount=10000,
                transaction_date=now - timedelta(days=10),
            ),
        ]
        features = CreditFeatureExtractor.extract(txs, 0, None, 30)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["avg_monthly_withdrawals"] > 0
        assert feat_dict["max_single_withdrawal"] == 2000.0
        assert feat_dict["savings_rate"] > 0  # net positive

    def test_loan_repayment_rate(self):
        """Loan repayment rate should be computed from disbursements and repayments."""
        now = datetime.now(UTC)
        txs = [
            FakeTransaction(
                transaction_type=TransactionType.LOAN_DISBURSEMENT,
                amount=100000,
                transaction_date=now - timedelta(days=30),
            ),
            FakeTransaction(
                transaction_type=TransactionType.LOAN_REPAYMENT,
                amount=80000,
                transaction_date=now - timedelta(days=15),
            ),
        ]
        features = CreditFeatureExtractor.extract(txs, 0, None, 60)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert 0.79 < feat_dict["loan_repayment_rate"] < 0.81  # 80000/100000

    def test_no_loans_gives_perfect_rate(self):
        """No loan transactions should give a repayment rate of 1.0."""
        tx = FakeTransaction(
            transaction_type=TransactionType.DEPOSIT,
            amount=1000,
            transaction_date=datetime.now(UTC) - timedelta(days=5),
        )
        features = CreditFeatureExtractor.extract([tx], 0, None, 30)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["loan_repayment_rate"] == 1.0

    def test_fraud_alert_features(self):
        """Fraud alerts should affect risk history features."""
        tx = FakeTransaction(
            transaction_type=TransactionType.DEPOSIT,
            amount=1000,
            transaction_date=datetime.now(UTC) - timedelta(days=5),
        )
        features = CreditFeatureExtractor.extract([tx], 3, 10, 30)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["total_fraud_alerts"] == 3.0
        assert feat_dict["days_since_last_fraud_alert"] == 10.0

    def test_no_fraud_gives_365_days(self):
        """No fraud alerts should give 365 days since last fraud."""
        tx = FakeTransaction(
            transaction_type=TransactionType.DEPOSIT,
            amount=1000,
            transaction_date=datetime.now(UTC) - timedelta(days=5),
        )
        features = CreditFeatureExtractor.extract([tx], 0, None, 30)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["days_since_last_fraud_alert"] == 365.0

    def test_transfer_features(self):
        """Transfer transactions should affect transfer ratio features."""
        now = datetime.now(UTC)
        txs = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                amount=5000,
                transaction_date=now - timedelta(days=5),
            ),
            FakeTransaction(
                transaction_type=TransactionType.TRANSFER,
                amount=2000,
                counterparty_account_id="ACC-OTHER",
                transaction_date=now - timedelta(days=3),
            ),
        ]
        features = CreditFeatureExtractor.extract(txs, 0, None, 30)
        feat_dict = dict(zip(CREDIT_FEATURE_NAMES, features))

        assert feat_dict["unique_counterparties"] >= 1

    def test_feature_names_match(self):
        """get_feature_names() should return the same list as CREDIT_FEATURE_NAMES."""
        assert CreditFeatureExtractor.get_feature_names() == list(CREDIT_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
