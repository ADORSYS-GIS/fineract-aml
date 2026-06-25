"""Tests for the anomaly detector."""

import numpy as np

from app.ml.anomaly_detector import AnomalyDetector


class TestAnomalyDetector:
    def test_train_and_predict(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.model_path", str(tmp_path))

        detector = AnomalyDetector()

        # Generate normal data with a few outliers
        rng = np.random.default_rng(42)
        normal_data = rng.normal(loc=0, scale=1, size=(200, 5))
        outliers = rng.normal(loc=10, scale=0.5, size=(10, 5))
        data = np.vstack([normal_data, outliers])

        metrics = detector.train(data, contamination=0.05)
        assert metrics["n_samples"] == 210
        assert metrics["n_anomalies_detected"] > 0

        # Normal point should have low anomaly score
        normal_score = detector.predict(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        assert 0.0 <= normal_score <= 1.0

        # Outlier should have higher anomaly score
        outlier_score = detector.predict(np.array([10.0, 10.0, 10.0, 10.0, 10.0]))
        assert outlier_score > normal_score

    def test_predict_without_model_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.model_path", str(tmp_path))
        detector = AnomalyDetector()
        score = detector.predict(np.array([1.0, 2.0, 3.0]))
        assert score == 0.0
