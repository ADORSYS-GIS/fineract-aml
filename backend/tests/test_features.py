"""Tests for feature extraction."""

from datetime import UTC, datetime

import numpy as np
import pytest

from app.features.extractor import FEATURE_NAMES, FeatureExtractor
from app.models.transaction import TransactionType
from tests.conftest import FakeTransaction


class TestFeatureExtractor:
    def test_output_shape(self):
        tx = FakeTransaction()
        features = FeatureExtractor.extract(tx, [], [])
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(FEATURE_NAMES),)

    def test_feature_names_match(self):
        names = FeatureExtractor.get_feature_names()
        assert names == FEATURE_NAMES
        assert len(names) == len(FEATURE_NAMES)

    def test_deposit_one_hot(self):
        tx = FakeTransaction(transaction_type=TransactionType.DEPOSIT)
        features = FeatureExtractor.extract(tx, [], [])
        assert features[2] == 1.0  # is_deposit
        assert features[3] == 0.0  # is_withdrawal
        assert features[4] == 0.0  # is_transfer

    def test_withdrawal_one_hot(self):
        tx = FakeTransaction(transaction_type=TransactionType.WITHDRAWAL)
        features = FeatureExtractor.extract(tx, [], [])
        assert features[2] == 0.0
        assert features[3] == 1.0
        assert features[4] == 0.0

    def test_weekend_flag(self):
        # 2025-06-14 is Saturday
        tx = FakeTransaction(
            transaction_date=datetime(2025, 6, 14, 10, 0, tzinfo=UTC)
        )
        features = FeatureExtractor.extract(tx, [], [])
        assert features[7] == 1.0  # is_weekend

    def test_night_flag(self):
        tx = FakeTransaction(
            transaction_date=datetime(2025, 6, 15, 3, 0, tzinfo=UTC)
        )
        features = FeatureExtractor.extract(tx, [], [])
        assert features[8] == 1.0  # is_night

    def test_history_features(self):
        tx = FakeTransaction(amount=1000.0)
        history = [FakeTransaction(amount=200.0) for _ in range(5)]
        features = FeatureExtractor.extract(tx, history, history)
        assert features[11] == 5.0  # tx_count_1h
        assert features[12] == 5.0  # tx_count_24h
        assert features[13] == 1000.0  # total_amount_1h (5*200)

    def test_all_features_are_finite(self):
        tx = FakeTransaction()
        features = FeatureExtractor.extract(tx, [], [])
        assert np.all(np.isfinite(features))

    def test_new_ip_feature(self):
        tx = FakeTransaction(ip_address="10.0.0.99")
        history = [FakeTransaction(ip_address="192.168.1.1") for _ in range(3)]
        features = FeatureExtractor.extract(tx, [], history)
        ip_new_idx = FEATURE_NAMES.index("is_new_ip_for_account")
        ip_count_idx = FEATURE_NAMES.index("unique_ips_24h")
        assert features[ip_new_idx] == 1.0  # new IP
        assert features[ip_count_idx] == 1.0  # 1 unique known IP

    def test_known_ip_feature(self):
        tx = FakeTransaction(ip_address="192.168.1.1")
        history = [FakeTransaction(ip_address="192.168.1.1") for _ in range(3)]
        features = FeatureExtractor.extract(tx, [], history)
        ip_new_idx = FEATURE_NAMES.index("is_new_ip_for_account")
        assert features[ip_new_idx] == 0.0  # known IP

    def test_feature_count_is_36(self):
        assert len(FEATURE_NAMES) == 36

    def test_7d_features_populated(self):
        tx = FakeTransaction(amount=500.0)
        history_7d = [FakeTransaction(amount=200.0) for _ in range(14)]
        features = FeatureExtractor.extract(tx, [], [], account_history_7d=history_7d)
        idx = FEATURE_NAMES.index("tx_count_7d")
        assert features[idx] == 14.0
        idx_total = FEATURE_NAMES.index("total_amount_7d")
        assert features[idx_total] == 14 * 200.0

    def test_7d_features_fallback_to_24h(self):
        """When account_history_7d is None, extractor uses 24h history as fallback."""
        tx = FakeTransaction(amount=100.0)
        history_24h = [FakeTransaction(amount=50.0) for _ in range(3)]
        features = FeatureExtractor.extract(tx, [], history_24h, account_history_7d=None)
        idx = FEATURE_NAMES.index("tx_count_7d")
        # Fallback: 7d features use 24h history
        assert features[idx] == 3.0

    def test_actor_context_agent(self):
        tx = FakeTransaction(actor_type="agent", kyc_level=2)
        features = FeatureExtractor.extract(tx, [], [])
        assert features[FEATURE_NAMES.index("is_agent")] == 1.0
        assert features[FEATURE_NAMES.index("is_merchant")] == 0.0
        assert features[FEATURE_NAMES.index("kyc_level_norm")] == pytest.approx(0.5)
        assert features[FEATURE_NAMES.index("is_new_kyc")] == 0.0

    def test_actor_context_merchant(self):
        tx = FakeTransaction(actor_type="merchant")
        features = FeatureExtractor.extract(tx, [], [])
        assert features[FEATURE_NAMES.index("is_agent")] == 0.0
        assert features[FEATURE_NAMES.index("is_merchant")] == 1.0

    def test_actor_context_new_kyc(self):
        tx = FakeTransaction(actor_type="customer", kyc_level=1)
        features = FeatureExtractor.extract(tx, [], [])
        assert features[FEATURE_NAMES.index("is_new_kyc")] == 1.0
        assert features[FEATURE_NAMES.index("kyc_level_norm")] == pytest.approx(0.25)

    def test_actor_context_unknown_kyc_defaults_to_half(self):
        tx = FakeTransaction(actor_type="customer", kyc_level=None)
        features = FeatureExtractor.extract(tx, [], [])
        assert features[FEATURE_NAMES.index("kyc_level_norm")] == pytest.approx(0.5)

    def test_receiver_diversity_7d(self):
        tx = FakeTransaction()
        history_7d = [
            FakeTransaction(counterparty_account_id=f"ACC-{i}") for i in range(4)
        ]
        features = FeatureExtractor.extract(tx, [], [], account_history_7d=history_7d)
        idx = FEATURE_NAMES.index("receiver_diversity_7d")
        # 4 unique recipients / 4 txns = 1.0
        assert features[idx] == pytest.approx(1.0)

    def test_geo_distance_different_country(self):
        tx = FakeTransaction(country_code="NG")  # Nigeria — different from history
        history_7d = [FakeTransaction(country_code="CM") for _ in range(5)]  # Cameroon
        features = FeatureExtractor.extract(tx, [], [], account_history_7d=history_7d)
        idx = FEATURE_NAMES.index("geo_distance_from_usual")
        assert features[idx] == 1.0

    def test_geo_distance_same_country(self):
        tx = FakeTransaction(country_code="CM")
        history_7d = [FakeTransaction(country_code="CM") for _ in range(5)]
        features = FeatureExtractor.extract(tx, [], [], account_history_7d=history_7d)
        idx = FEATURE_NAMES.index("geo_distance_from_usual")
        assert features[idx] == 0.0

    def test_velocity_trend_normal(self):
        """Equal 24h and 7d rate should give trend ≈ 1.0."""
        tx = FakeTransaction()
        # 1 txn/day over 7d = 7 txns; 1 txn in 24h → trend ≈ 1.0
        history_24h = [FakeTransaction() for _ in range(1)]
        history_7d = [FakeTransaction() for _ in range(7)]
        features = FeatureExtractor.extract(tx, [], history_24h, account_history_7d=history_7d)
        idx = FEATURE_NAMES.index("tx_velocity_trend")
        assert features[idx] == pytest.approx(1.0)
