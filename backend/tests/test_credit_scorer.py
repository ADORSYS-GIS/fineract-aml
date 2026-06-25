"""Tests for the credit scoring engine."""

import unittest
from datetime import UTC, datetime, timedelta

import numpy as np

from app.features.credit_extractor import CreditFeatureExtractor
from app.ml.credit_scorer import CreditScorer
from app.models.credit_profile import CreditSegment
from app.models.transaction import TransactionType
from tests.conftest import FakeTransaction


class TestCreditScorer(unittest.TestCase):
    """Test rule-based credit scoring and segment classification."""

    def setUp(self):
        self.scorer = CreditScorer()

    def test_high_score_gets_tier_a(self):
        """Score >= 0.8 should map to Tier A."""
        assert self.scorer.classify_segment(0.85) == CreditSegment.TIER_A
        assert self.scorer.classify_segment(0.80) == CreditSegment.TIER_A

    def test_medium_score_gets_tier_c(self):
        """Score between 0.5 and 0.65 should map to Tier C."""
        assert self.scorer.classify_segment(0.55) == CreditSegment.TIER_C

    def test_low_score_gets_tier_e(self):
        """Score below 0.35 should map to Tier E."""
        assert self.scorer.classify_segment(0.2) == CreditSegment.TIER_E
        assert self.scorer.classify_segment(0.0) == CreditSegment.TIER_E

    def test_tier_a_max_amount(self):
        """Tier A should have the highest max credit amount."""
        tier_a_amount = self.scorer.compute_max_amount(CreditSegment.TIER_A)
        tier_b_amount = self.scorer.compute_max_amount(CreditSegment.TIER_B)
        tier_e_amount = self.scorer.compute_max_amount(CreditSegment.TIER_E)

        assert tier_a_amount > tier_b_amount
        assert tier_e_amount == 0.0

    def test_score_returns_tuple(self):
        """score() should return (float, dict)."""
        features = np.zeros(19)  # len(CREDIT_FEATURE_NAMES)
        score, components = self.scorer.score(features)
        assert isinstance(score, float)
        assert isinstance(components, dict)
        assert 0.0 <= score <= 1.0

    def test_good_customer_scores_high(self):
        """A customer with consistent deposits should score well."""
        now = datetime.now(UTC)
        txs = []

        # Regular monthly deposits (6 months)
        for i in range(6):
            txs.append(FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                amount=50000,
                transaction_date=now - timedelta(days=30 * i + 1),
            ))

        # Regular withdrawals (lower than deposits)
        for i in range(6):
            txs.append(FakeTransaction(
                transaction_type=TransactionType.WITHDRAWAL,
                amount=20000,
                transaction_date=now - timedelta(days=30 * i + 15),
            ))

        features = CreditFeatureExtractor.extract(txs, 0, None, 180)
        score, components = self.scorer.score(features)

        assert score > 0.4, f"Good customer should score > 0.4, got {score}"

    def test_fraud_customer_scores_lower(self):
        """A customer with fraud alerts should score lower."""
        now = datetime.now(UTC)
        txs = [
            FakeTransaction(
                transaction_type=TransactionType.DEPOSIT,
                amount=50000,
                transaction_date=now - timedelta(days=30 * i + 1),
            )
            for i in range(6)
        ]

        # With fraud
        features_fraud = CreditFeatureExtractor.extract(txs, 3, 5, 180)
        score_fraud, _ = self.scorer.score(features_fraud)

        # Without fraud
        features_clean = CreditFeatureExtractor.extract(txs, 0, None, 180)
        score_clean, _ = self.scorer.score(features_clean)

        assert score_fraud < score_clean, (
            f"Fraud customer ({score_fraud}) should score lower than clean ({score_clean})"
        )

    def test_recommend_approve(self):
        """Good score + low amount should recommend approve."""
        from app.models.credit_request import CreditRecommendation

        rec = self.scorer.recommend(
            score=0.75,
            segment=CreditSegment.TIER_B,
            requested_amount=500000,  # Half of 2M tier B limit
            max_amount=2000000,
        )
        assert rec == CreditRecommendation.APPROVE

    def test_recommend_reject_over_limit(self):
        """Amount over max should recommend reject."""
        from app.models.credit_request import CreditRecommendation

        rec = self.scorer.recommend(
            score=0.75,
            segment=CreditSegment.TIER_B,
            requested_amount=5000000,  # Over 2M tier B limit
            max_amount=2000000,
        )
        assert rec == CreditRecommendation.REJECT

    def test_recommend_reject_tier_e(self):
        """Tier E should always recommend reject."""
        from app.models.credit_request import CreditRecommendation

        rec = self.scorer.recommend(
            score=0.1,
            segment=CreditSegment.TIER_E,
            requested_amount=100,
            max_amount=0,
        )
        assert rec == CreditRecommendation.REJECT

    def test_recommend_review_carefully(self):
        """Marginal cases should recommend review_carefully."""
        from app.models.credit_request import CreditRecommendation

        rec = self.scorer.recommend(
            score=0.55,
            segment=CreditSegment.TIER_C,
            requested_amount=800000,  # 80% of 1M tier C limit
            max_amount=1000000,
        )
        assert rec == CreditRecommendation.REVIEW_CAREFULLY


if __name__ == "__main__":
    unittest.main()
