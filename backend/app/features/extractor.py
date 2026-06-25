"""Feature engineering — transforms raw transactions into ML-ready features.

Features are the numerical representations that ML models use to
make predictions. Good features are the #1 driver of model performance.

Feature categories:
1. Transaction-level: amount, time, type
2. Account-level: history, patterns, age
3. Behavioral: deviations from normal patterns
4. Extended windows (7d): velocity trends, receiver diversity, geo distance
5. Actor context: actor type flags, KYC level

NOTE: Feature count expanded from 22 → 36 in Phase 7.1, then 36 → 38 in Phase 7.2:
  - hour_of_day replaced with hour_sin + hour_cos (cyclical encoding)
  - is_new_device added (SIM-swap / multi-account signal)
Existing trained models will return is_ready=False until retrained.
"""

import logging
import math
from datetime import datetime

import numpy as np

from app.models.transaction import Transaction, TransactionType

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # ── Transaction-level (12) ────────────────────────────────────────────────
    "amount",
    "amount_log",
    "is_deposit",
    "is_withdrawal",
    "is_transfer",
    "hour_sin",              # cyclical encoding of hour (sin component)
    "hour_cos",              # cyclical encoding of hour (cos component)
    "day_of_week",
    "is_weekend",
    "is_night",              # 10PM–6AM
    "is_round_hundred",
    "is_round_thousand",
    # ── Account-level / 24h window (12) ──────────────────────────────────────
    "tx_count_1h",
    "tx_count_24h",
    "total_amount_1h",
    "total_amount_24h",
    "avg_amount_24h",
    "max_amount_24h",
    "amount_vs_avg_ratio",
    "unique_counterparties_24h",
    "same_type_ratio_24h",
    "is_new_ip_for_account",
    "unique_ips_24h",
    "is_new_device",         # 1.0=new/unknown device, 0.0=familiar, 0.5=unknown
    # ── Extended 7d window (10) ───────────────────────────────────────────────
    "tx_count_7d",
    "total_amount_7d",
    "avg_amount_7d",
    "max_amount_7d",
    "unique_counterparties_7d",
    "amount_vs_7d_avg_ratio",
    "tx_velocity_trend",     # 24h / 7d tx rate (normalized)
    "receiver_diversity_7d", # unique recipients / total 7d txns (Medium article #2)
    "geo_distance_from_usual", # 1.0 if country differs from 7d mode (Medium article #2)
    "has_loan_disbursement_7d",
    # ── Actor context (4) ─────────────────────────────────────────────────────
    "is_agent",
    "is_merchant",
    "kyc_level_norm",        # kyc_level / 4.0 (0=unknown → 0.5 default)
    "is_new_kyc",            # kyc_level == 1 (brand-new / unverified account)
]

# ── Graph fraud-layer features (7) — appended only when graph_enabled (ADR 0007) ──────
# Toggling AML_GRAPH_ENABLED changes the vector width, so a model trained under one setting
# returns is_ready=False under the other (guarded by the dimension check in the ML models).
GRAPH_FEATURE_NAMES = [
    "graph_distance_to_known_bad",  # 1/hops to nearest known_bad/frozen/sanctioned (0 if none)
    "graph_shared_device_count",    # # other accounts sharing this device
    "graph_shared_ip_count",        # # other accounts sharing this IP
    "graph_degree",                 # money-flow degree (in+out)
    "graph_pagerank",               # money-flow centrality
    "graph_is_in_cycle",            # participates in a layering cycle
    "graph_community_size",         # size of the account's connected community
]


def _graph_features_enabled() -> bool:
    from app.core.config import settings
    return settings.graph_enabled


class FeatureExtractor:
    """Extracts numerical features from a transaction and its context."""

    @staticmethod
    def extract(
        transaction: Transaction,
        account_history_1h: list[Transaction],
        account_history_24h: list[Transaction],
        account_history_7d: list[Transaction] | None = None,
        graph_features: dict | None = None,
    ) -> np.ndarray:
        """Extract features for a single transaction.

        Args:
            transaction: The transaction to score.
            account_history_1h: Same account's transactions in last 1 hour.
            account_history_24h: Same account's transactions in last 24 hours.
            account_history_7d: Same account's transactions in last 7 days (optional).
                                 When None, 7d features default to 24h values / 7.

        Returns:
            Feature vector as numpy array of shape (38,).
        """
        features = []
        h7d = account_history_7d or account_history_24h  # graceful fallback

        # ── Transaction-level features ────────────────────────────────────────
        features.append(transaction.amount)
        features.append(np.log1p(transaction.amount))

        features.append(1.0 if transaction.transaction_type == TransactionType.DEPOSIT else 0.0)
        features.append(1.0 if transaction.transaction_type == TransactionType.WITHDRAWAL else 0.0)
        features.append(1.0 if transaction.transaction_type == TransactionType.TRANSFER else 0.0)

        tx_dt = transaction.transaction_date
        if isinstance(tx_dt, datetime):
            hour = tx_dt.hour
            dow = tx_dt.weekday()
        else:
            hour = 12
            dow = 0

        # Cyclical encoding: treats hour 23 and hour 0 as adjacent
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        features.append(hour_sin)
        features.append(hour_cos)
        features.append(float(dow))
        features.append(1.0 if dow >= 5 else 0.0)
        features.append(1.0 if hour >= 22 or hour <= 6 else 0.0)

        features.append(1.0 if transaction.amount >= 100 and transaction.amount % 100 == 0 else 0.0)
        features.append(1.0 if transaction.amount >= 1000 and transaction.amount % 1000 == 0 else 0.0)

        # ── Account-level / 24h window features ──────────────────────────────
        features.append(float(len(account_history_1h)))
        features.append(float(len(account_history_24h)))

        total_1h = sum(t.amount for t in account_history_1h)
        total_24h = sum(t.amount for t in account_history_24h)
        features.append(total_1h)
        features.append(total_24h)

        # Avoid raw XAF values for new accounts by using a population floor
        POPULATION_FLOOR_XAF = 10_000.0  # typical minimum meaningful transaction
        if account_history_24h:
            amounts_24h = [t.amount for t in account_history_24h]
            avg_24h = float(np.mean(amounts_24h))
            max_24h = float(np.max(amounts_24h))
            features.append(avg_24h)
            features.append(max_24h)
            amount_ratio = transaction.amount / max(avg_24h, POPULATION_FLOOR_XAF)
            amount_ratio = min(amount_ratio, 50.0)  # cap at 50x to prevent outlier domination
            features.append(amount_ratio)
        else:
            features.append(0.0)
            features.append(0.0)
            # New account: ratio against population floor, capped
            amount_ratio = min(transaction.amount / POPULATION_FLOOR_XAF, 50.0)
            features.append(amount_ratio)

        counterparties_24h = {
            t.counterparty_account_id for t in account_history_24h if t.counterparty_account_id
        }
        features.append(float(len(counterparties_24h)))

        if account_history_24h:
            same_type = sum(
                1 for t in account_history_24h
                if t.transaction_type == transaction.transaction_type
            )
            features.append(same_type / len(account_history_24h))
        else:
            features.append(0.0)

        known_ips = {
            t.ip_address for t in account_history_24h if getattr(t, "ip_address", None)
        }
        current_ip = getattr(transaction, "ip_address", None)
        features.append(1.0 if current_ip and current_ip not in known_ips else 0.0)
        features.append(float(len(known_ips)))

        # Device fingerprint consistency — SIM-swap and multi-account signals
        # 1.0 = new/unknown device, 0.0 = familiar device
        device_id = getattr(transaction, "device_id", None)
        if device_id and account_history_24h:
            known_devices = {getattr(t, "device_id", None) for t in account_history_24h if getattr(t, "device_id", None)}
            is_new_device = 1.0 if device_id not in known_devices else 0.0
        else:
            is_new_device = 0.5  # unknown
        features.append(is_new_device)

        # ── Extended 7d window features ───────────────────────────────────────
        tx_count_7d = float(len(h7d))
        total_7d = sum(t.amount for t in h7d)
        avg_7d = total_7d / max(tx_count_7d, 1.0)
        max_7d = float(max((t.amount for t in h7d), default=0.0))

        features.append(tx_count_7d)
        features.append(total_7d)
        features.append(avg_7d)
        features.append(max_7d)

        counterparties_7d = {t.counterparty_account_id for t in h7d if t.counterparty_account_id}
        features.append(float(len(counterparties_7d)))

        features.append(min(transaction.amount / max(avg_7d, POPULATION_FLOOR_XAF), 50.0))

        # Velocity trend: compare 24h rate vs 7d average daily rate
        daily_rate_24h = len(account_history_24h)          # txns in last 24h
        daily_rate_7d_avg = max(tx_count_7d / 7.0, 0.001) # avg txns/day over 7d
        features.append(daily_rate_24h / daily_rate_7d_avg)

        # Receiver diversity: unique recipients / total 7d txns (0 = no diversity = suspicious)
        if h7d:
            recipients = {t.counterparty_account_id for t in h7d if t.counterparty_account_id}
            features.append(len(recipients) / max(len(h7d), 1))
        else:
            features.append(0.0)

        # Geo distance: 1.0 if current country differs from most common country in 7d history
        countries_7d = [
            getattr(t, "country_code", None) for t in h7d if getattr(t, "country_code", None)
        ]
        current_country = getattr(transaction, "country_code", None)
        if countries_7d and current_country:
            from collections import Counter
            modal_country = Counter(countries_7d).most_common(1)[0][0]
            features.append(0.0 if current_country == modal_country else 1.0)
        else:
            features.append(0.0)

        has_loan = any(
            getattr(t, "transaction_type", None) == TransactionType.LOAN_DISBURSEMENT
            for t in h7d
        )
        features.append(1.0 if has_loan else 0.0)

        # ── Actor context features ────────────────────────────────────────────
        actor_type = getattr(transaction, "actor_type", None)
        features.append(1.0 if actor_type == "agent" else 0.0)
        features.append(1.0 if actor_type == "merchant" else 0.0)

        kyc_level = getattr(transaction, "kyc_level", None)
        features.append(float(kyc_level) / 4.0 if kyc_level else 0.5)
        features.append(1.0 if kyc_level == 1 else 0.0)

        # ── Graph fraud-layer features (only when enabled) ────────────────────
        if _graph_features_enabled():
            gf = graph_features or {}
            for name in GRAPH_FEATURE_NAMES:
                # GRAPH_FEATURE_NAMES are prefixed "graph_"; the dict uses bare keys.
                features.append(float(gf.get(name[len("graph_"):], 0.0)))

        expected = FeatureExtractor.get_feature_names()
        assert len(features) == len(expected), (
            f"Feature count mismatch: got {len(features)}, expected {len(expected)}"
        )
        return np.array(features, dtype=np.float64)

    @staticmethod
    def get_feature_names() -> list[str]:
        if _graph_features_enabled():
            return FEATURE_NAMES + GRAPH_FEATURE_NAMES
        return FEATURE_NAMES
