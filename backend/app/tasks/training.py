"""Model training tasks — scheduled periodic retraining.

Anomaly detector: retrained daily on all recent transactions.
Fraud classifier: retrained weekly once enough labeled data exists.

IMPORTANT: Features are extracted with proper account history windows
to avoid train/serve skew. Each transaction gets its 1h and 24h history
reconstructed from the training batch + database, matching inference behavior.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import numpy as np

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _build_account_history(
    tx: "Transaction",
    all_transactions: list["Transaction"],
    account_index: dict[str, list["Transaction"]],
) -> tuple[list["Transaction"], list["Transaction"], list["Transaction"]]:
    """Reconstruct 1h, 24h, and 7d account history for a transaction from batch data.

    This ensures training features match inference features (no train/serve skew).
    Returns (history_1h, history_24h, history_7d).
    """
    siblings = account_index.get(tx.fineract_account_id, [])
    tx_date = tx.transaction_date
    cutoff_1h = tx_date - timedelta(hours=1)
    cutoff_24h = tx_date - timedelta(hours=24)
    cutoff_7d = tx_date - timedelta(days=7)

    history_1h = [
        t for t in siblings
        if t.id != tx.id and cutoff_1h <= t.transaction_date < tx_date
    ]
    history_24h = [
        t for t in siblings
        if t.id != tx.id and cutoff_24h <= t.transaction_date < tx_date
    ]
    history_7d = [
        t for t in siblings
        if t.id != tx.id and cutoff_7d <= t.transaction_date < tx_date
    ]
    return history_1h, history_24h, history_7d


def _build_account_index(transactions: list["Transaction"]) -> dict[str, list["Transaction"]]:
    """Index transactions by account ID for efficient history lookup."""
    index: dict[str, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        index[tx.fineract_account_id].append(tx)
    return index


def _save_drift_baseline(feature_matrix, detector, feature_names):
    """Save drift baseline after anomaly detector training."""
    try:
        from app.ml.drift_detector import DriftDetector

        scores = np.array([detector.predict(row) for row in feature_matrix[:1000]])
        drift = DriftDetector()
        drift.save_baseline(feature_matrix[:1000], scores, feature_names)
    except Exception as e:
        logger.warning("Failed to save drift baseline: %s", e)


def _check_and_log_drift(feature_matrix, detector, feature_names):
    """Check feature drift against baseline using actual model scores."""
    try:
        from app.ml.drift_detector import DriftDetector

        drift = DriftDetector()
        # Use actual model scores instead of zero placeholder to get meaningful PSI
        scaled = detector.scaler.transform(feature_matrix)
        raw_scores = detector.model.decision_function(scaled)
        scores = -raw_scores  # invert: higher = more anomalous
        result = drift.check_drift(feature_matrix, scores, feature_names)
        if result.get("status") in ("warning", "critical"):
            logger.warning("Drift detected: %s", result)
    except Exception as e:
        logger.warning("Failed to check drift: %s", e)


async def _save_health_snapshot(db, model_name: str, metrics: dict) -> None:
    """Persist a ModelHealthSnapshot after each training run."""
    try:
        import json

        from app.models.model_health import ModelHealthSnapshot

        snapshot = ModelHealthSnapshot(
            model_name=model_name,
            model_version=metrics.get("model_version"),
            trained_at=datetime.now(UTC),
            training_sample_count=metrics.get("n_samples"),
            auc_score=metrics.get("roc_auc") or metrics.get("auc"),
            precision_score=metrics.get("precision"),
            recall_score=metrics.get("recall"),
            drift_status="stable",  # freshly trained = stable by definition
            metrics_json=json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}),
        )
        db.add(snapshot)
        await db.commit()

        # Also push to Redis for fast access
        import redis

        from app.core.config import settings
        r = redis.from_url(settings.redis_url, socket_connect_timeout=1)
        r.setex(
            "model_health:latest",
            86400,
            json.dumps({
                "model_name": model_name,
                "trained_at": snapshot.trained_at.isoformat(),
                "sample_count": snapshot.training_sample_count,
                "auc": snapshot.auc_score,
                "drift_status": "stable",
                "snapshot_at": snapshot.trained_at.isoformat(),
            }),
        )
    except Exception as exc:
        logger.warning("Failed to save model health snapshot: %s", exc)


def _log_to_mlflow(model_name: str, metrics: dict):
    """Log training metrics to MLflow if available."""
    try:
        import socket

        import mlflow

        from app.core.config import settings

        # Quick reachability check (1s timeout) before attempting MLflow calls
        uri = settings.mlflow_tracking_uri.replace("http://", "").replace("https://", "")
        host, _, port_str = uri.partition(":")
        port = int(port_str) if port_str else 5000
        s = socket.create_connection((host, port), timeout=1)
        s.close()

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"{model_name}_retrain"):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                elif isinstance(value, str):
                    mlflow.log_param(key, value[:250])  # MLflow param limit
            mlflow.log_param("model_name", model_name)
    except Exception as e:
        logger.debug("MLflow logging skipped (not available): %s", e)


async def _retrain_anomaly():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.features.extractor import FeatureExtractor
    from app.ml.anomaly_detector import AnomalyDetector
    from app.models.alert import Alert, AlertStatus
    from app.models.transaction import Transaction

    _engine = create_async_engine(settings.database_url)
    async_session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Exclude confirmed-fraud transactions — Isolation Forest assumes mostly-clean data
        confirmed_fraud_tx_ids = select(Alert.transaction_id).where(
            Alert.status == AlertStatus.CONFIRMED_FRAUD,
            Alert.transaction_id.isnot(None),
        )
        result = await db.execute(
            select(Transaction)
            .where(Transaction.id.notin_(confirmed_fraud_tx_ids))
            .order_by(Transaction.created_at.desc())
            .limit(10000)
        )
        transactions = list(result.scalars().all())

        if len(transactions) < 100:
            logger.info("Not enough transactions for anomaly training: %d", len(transactions))
            return

        # Build account index for efficient history reconstruction
        account_index = _build_account_index(transactions)

        # Build feature matrix with proper account history (fixes train/serve skew)
        features_list = []
        for tx in transactions:
            history_1h, history_24h, history_7d = _build_account_history(tx, transactions, account_index)
            features = FeatureExtractor.extract(tx, history_1h, history_24h, history_7d)
            features_list.append(features)

        feature_matrix = np.vstack(features_list)

        detector = AnomalyDetector()
        metrics = detector.train(feature_matrix)

        # Save drift baseline and check drift against previous baseline
        _save_drift_baseline(feature_matrix, detector, FeatureExtractor.get_feature_names())
        _check_and_log_drift(feature_matrix, detector, FeatureExtractor.get_feature_names())
        _log_to_mlflow("anomaly_detector", metrics)

        # Persist ModelHealthSnapshot
        await _save_health_snapshot(db, "anomaly_detector", metrics)

        logger.info("Anomaly detector retrained: %s", metrics)

    await _engine.dispose()


async def _retrain_classifier():
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.orm import selectinload

    from app.core.config import settings
    from app.features.extractor import FeatureExtractor
    from app.ml.fraud_classifier import FraudClassifier
    from app.models.alert import Alert, AlertStatus
    from app.models.transaction import Transaction

    _engine = create_async_engine(settings.database_url)
    async_session = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        # Get labeled data: transactions with confirmed fraud or false positive alerts
        fraud_alerts = await db.execute(
            select(Alert)
            .options(selectinload(Alert.transaction))
            .where(Alert.status == AlertStatus.CONFIRMED_FRAUD)
        )
        legit_alerts = await db.execute(
            select(Alert)
            .options(selectinload(Alert.transaction))
            .where(Alert.status == AlertStatus.FALSE_POSITIVE)
        )

        fraud_txs = [a.transaction for a in fraud_alerts.scalars().all() if a.transaction]
        legit_txs = [a.transaction for a in legit_alerts.scalars().all() if a.transaction]

        # Sample unalerted transactions as clean negatives (the majority class in real life)
        # Use 3x the positive count to balance without extreme imbalance
        n_positives = len(fraud_txs)
        if n_positives > 0:
            clean_sample_result = await db.execute(
                select(Transaction)
                .where(~Transaction.id.in_(
                    select(Alert.transaction_id).where(Alert.transaction_id.isnot(None))
                ))
                .order_by(func.random())
                .limit(min(n_positives * 3, 5000))
            )
            clean_txs = list(clean_sample_result.scalars().all())
        else:
            clean_txs = []

        classifier = FraudClassifier()
        total_samples = len(fraud_txs) + len(legit_txs) + len(clean_txs)
        if not classifier.can_train(len(fraud_txs), total_samples):
            logger.info(
                "Not enough labeled data: %d fraud, %d legitimate, %d clean (need %d fraud, %d total)",
                len(fraud_txs),
                len(legit_txs),
                len(clean_txs),
                classifier.MIN_FRAUD_SAMPLES,
                classifier.MIN_TOTAL_SAMPLES,
            )
            return

        # Combine: fraud=1, false_positive=0, clean=0
        all_transactions = fraud_txs + legit_txs + clean_txs
        all_labels = [1] * len(fraud_txs) + [0] * len(legit_txs) + [0] * len(clean_txs)

        # Fetch surrounding transactions for all accounts involved (24h window)
        account_ids = {tx.fineract_account_id for tx in all_transactions}
        all_related_txs = []
        for account_id in account_ids:
            result = await db.execute(
                select(Transaction)
                .where(Transaction.fineract_account_id == account_id)
                .order_by(Transaction.transaction_date)
            )
            all_related_txs.extend(result.scalars().all())

        account_index = _build_account_index(all_related_txs)

        # Build feature matrix with proper account history (fixes train/serve skew)
        features_list = []
        for tx in all_transactions:
            history_1h, history_24h, history_7d = _build_account_history(tx, all_related_txs, account_index)
            features = FeatureExtractor.extract(tx, history_1h, history_24h, history_7d)
            features_list.append(features)

        feature_matrix = np.vstack(features_list)
        label_array = np.array(all_labels)

        # Sort by transaction date to avoid future data leakage (temporal CV)
        sorted_pairs = sorted(
            zip(all_transactions, features_list, all_labels),
            key=lambda p: p[0].transaction_date,
        )
        sorted_features = np.vstack([p[1] for p in sorted_pairs])
        sorted_labels = np.array([p[2] for p in sorted_pairs])

        # Temporal CV: use first 70% as train, last 30% as validation
        n = len(sorted_pairs)
        split = int(n * 0.7)
        train_features = sorted_features[:split]
        train_labels = sorted_labels[:split]
        val_features = sorted_features[split:]
        val_labels = sorted_labels[split:]

        metrics = classifier.train(
            train_features,
            train_labels,
            val_features=val_features,
            val_labels=val_labels,
            feature_names=FeatureExtractor.get_feature_names(),
        )

        # After a successful production deployment, write a copy to the shadow slot.
        if metrics.get("deployed"):
            try:
                from sklearn.preprocessing import FunctionTransformer

                from app.ml.shadow_scorer import ShadowScorer
                shadow = ShadowScorer()
                shadow.save_as_shadow(classifier.model, FunctionTransformer())
                logger.info("Shadow model updated after classifier training")
            except Exception as shadow_exc:
                logger.warning("Failed to update shadow model: %s", shadow_exc)

        # Check drift and log to MLflow
        _check_and_log_drift(feature_matrix, detector=None, feature_names=FeatureExtractor.get_feature_names())
        _log_to_mlflow("fraud_classifier", metrics)

        # Persist ModelHealthSnapshot
        await _save_health_snapshot(db, "fraud_classifier", metrics)

        logger.info("Fraud classifier retrained: %s", metrics)

    await _engine.dispose()


@celery_app.task(name="app.tasks.training.retrain_anomaly_detector")
def retrain_anomaly_detector():
    """Retrain the anomaly detector on recent transactions."""
    _run_async(_retrain_anomaly())


@celery_app.task(name="app.tasks.training.retrain_fraud_classifier")
def retrain_fraud_classifier():
    """Retrain the fraud classifier on labeled data."""
    _run_async(_retrain_classifier())
