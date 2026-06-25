"""Unsupervised anomaly detection — works WITHOUT labeled data.

This module uses Isolation Forest to detect unusual transactions.
The key insight: fraudulent transactions are rare and different from
normal ones, so they're easy to "isolate" in a decision tree.

No labels needed — the model learns what "normal" looks like and
flags anything that deviates.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from app.core.config import settings

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Isolation Forest-based anomaly detector for transactions.

    Training: fit on historical transactions (all assumed mostly normal).
    Inference: score new transactions — higher score = more anomalous.
    """

    def __init__(self):
        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        # Percentile bounds computed during training for consistent inference normalization
        self._score_p5: float | None = None
        self._score_p95: float | None = None
        self._model_path = Path(settings.model_path) / "anomaly_detector.joblib"
        self._scaler_path = Path(settings.model_path) / "anomaly_scaler.joblib"
        self._norm_path = Path(settings.model_path) / "anomaly_norm_params.joblib"

    def train(self, features: np.ndarray, contamination: float | None = None) -> dict:
        """Train the anomaly detector on historical transaction features.

        Args:
            features: 2D array of shape (n_transactions, n_features).
            contamination: Expected proportion of anomalies. Defaults to config value (~1%).

        Returns:
            Training metrics.
        """
        if contamination is None:
            contamination = settings.anomaly_contamination

        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)

        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(scaled_features)

        # Get training anomaly scores
        raw_scores = self.model.decision_function(scaled_features)
        predictions = self.model.predict(scaled_features)
        n_anomalies = int(np.sum(predictions == -1))

        # Compute and store percentile bounds for consistent inference normalization
        inverted = -raw_scores  # invert: higher = more anomalous
        self._score_p5 = float(np.percentile(inverted, 5))
        self._score_p95 = float(np.percentile(inverted, 95))

        scores = raw_scores  # keep raw for metrics
        self._save()

        metrics = {
            "n_samples": len(features),
            "n_anomalies_detected": n_anomalies,
            "anomaly_rate": n_anomalies / len(features),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }
        logger.info("Anomaly detector trained: %s", metrics)
        return metrics

    def predict(self, features: np.ndarray) -> float:
        """Score a single transaction. Returns anomaly score between 0 and 1.

        Higher score = more anomalous = more suspicious.
        """
        if self.model is None:
            self._load()

        if self.model is None:
            logger.warning("No trained anomaly model available, returning 0.0")
            return 0.0

        # Guard against feature-width drift (e.g. AML_GRAPH_ENABLED toggled after training):
        # a model trained on a different vector width must not crash the scoring path.
        expected_n = getattr(self.scaler, "n_features_in_", None)
        if expected_n is not None and expected_n != features.shape[0]:
            logger.warning(
                "Anomaly model feature mismatch (model=%s, input=%s) — returning 0.0; retrain",
                expected_n, features.shape[0],
            )
            return 0.0

        scaled = self.scaler.transform(features.reshape(1, -1))

        # decision_function returns negative for anomalies
        raw_score = self.model.decision_function(scaled)[0]

        # Normalize IF scores using percentile clipping for better dynamic range.
        # IF decision_function: more negative = more anomalous; invert so higher = more anomalous.
        inverted = -raw_score

        p5 = self._score_p5
        p95 = self._score_p95
        if p5 is not None and p95 is not None and p95 > p5:
            normalized = float(np.clip((inverted - p5) / (p95 - p5), 0.0, 1.0))
        else:
            # Fallback to sigmoid if percentile params are unavailable (old model)
            normalized = float(1.0 / (1.0 + np.exp(raw_score * 5)))

        return float(np.clip(normalized, 0.0, 1.0))

    def _save(self):
        """Persist model, scaler, and normalization params to disk using atomic writes."""
        model_dir = Path(settings.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: dump to temp file, then rename to avoid read corruption
        tmp_model = self._model_path.with_suffix(".joblib.tmp")
        tmp_scaler = self._scaler_path.with_suffix(".joblib.tmp")
        tmp_norm = self._norm_path.with_suffix(".joblib.tmp")
        joblib.dump(self.model, tmp_model)
        joblib.dump(self.scaler, tmp_scaler)
        joblib.dump({"p5": self._score_p5, "p95": self._score_p95}, tmp_norm)
        tmp_model.replace(self._model_path)
        tmp_scaler.replace(self._scaler_path)
        tmp_norm.replace(self._norm_path)
        logger.info("Anomaly detector saved to %s", self._model_path)

    def _load(self):
        """Load model, scaler, and normalization params from disk."""
        if self._model_path.exists() and self._scaler_path.exists():
            self.model = joblib.load(self._model_path)
            self.scaler = joblib.load(self._scaler_path)
            # Load normalization params if available (may be missing for old models)
            if self._norm_path.exists():
                norm = joblib.load(self._norm_path)
                self._score_p5 = norm.get("p5")
                self._score_p95 = norm.get("p95")
            logger.info("Anomaly detector loaded from %s", self._model_path)
        else:
            logger.warning("No saved anomaly detector found at %s", self._model_path)
