"""Shadow model promotion task.

Promotes the shadow (challenger) model to production after validation.

The shadow model is automatically updated after each successful training run
(see tasks/training.py). It can be promoted manually via this task once analysts
have verified that shadow scores are consistently better than production.

Typical workflow:
  1. Fraud classifier retrains → shadow slot is updated automatically
  2. Analysts compare shadow_score vs production score in the dashboard (7d)
  3. If shadow AUC > production AUC + 0.02 consistently, run this task
  4. Production model is replaced; shadow slot becomes the former production model
"""

import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.promote_shadow.promote_shadow_model", bind=True)
def promote_shadow_model(self) -> dict:
    """Promote the shadow fraud classifier to production.

    Copies model_shadow.joblib → fraud_classifier.joblib atomically.
    Logs a ModelHealthSnapshot for audit trail.

    Returns:
        Dict with status, promoted_from, promoted_to paths.
    """
    try:
        from app.ml.fraud_classifier import FraudClassifier
        from app.ml.shadow_scorer import ShadowScorer

        shadow = ShadowScorer()
        classifier = FraudClassifier()

        promoted_to = str(classifier._model_path)
        promoted_from = str(shadow._model_path)

        # Validate shadow model exists before promoting
        if not shadow.is_ready:
            logger.warning("promote_shadow_model: no shadow model found — nothing to promote")
            return {"status": "skipped", "reason": "no_shadow_model"}

        shadow.promote_to_production(
            production_model_path=promoted_to,
            production_scaler_path=str(shadow._scaler_path),
        )

        # Reload production classifier to confirm it works after promotion
        classifier._load()
        if not classifier.is_ready:
            raise RuntimeError("Classifier failed to load after shadow promotion")

        logger.info(
            "Shadow model successfully promoted to production: %s → %s",
            promoted_from,
            promoted_to,
        )
        return {
            "status": "promoted",
            "promoted_from": promoted_from,
            "promoted_to": promoted_to,
        }

    except Exception as exc:
        logger.exception("Failed to promote shadow model: %s", exc)
        raise
