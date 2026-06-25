"""Transaction analysis tasks — triggered for every incoming transaction.

Pipeline:
1. Extract features from the transaction + account history
2. Run rule engine (deterministic checks)
3. Run anomaly detector (unsupervised ML — no labels needed)
4. Run fraud classifier (supervised ML — only if trained model exists)
5. Combine scores and create alert if threshold exceeded
6. Screen counterparty against sanctions/PEP watchlists
7. Generate CTR if amount exceeds regulatory threshold
8. Store per-feature score explanation for model explainability
"""

import asyncio
import json
import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from synchronous Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _upsert_graph_edges(transaction, transaction_id: str) -> None:
    """Write the transaction's profile-asset edges to the graph DB (fail-open)."""
    from app.core.config import settings

    if not settings.graph_enabled:
        return
    try:
        from app.core.graph import ephemeral_graph_session
        from app.graph.client import GraphClient, tx_to_graph_payload

        async with ephemeral_graph_session() as gsession:
            if gsession is not None:
                await GraphClient(gsession).upsert_transaction(tx_to_graph_payload(transaction))
    except Exception:
        logger.warning("Graph upsert failed for tx %s (non-fatal)", transaction_id, exc_info=True)


async def _graph_read(account_id: str) -> dict:
    """Read graph ML features + bounded graph score in one session (fail-open).

    Returns {"features": dict | None, "score": float}. On disabled/unavailable/error the
    features are None (extractor then uses zeros) and the score is 0.0.
    """
    from app.core.config import settings

    empty = {"features": None, "score": 0.0}
    if not settings.graph_enabled or not account_id:
        return empty
    try:
        from app.core.graph import ephemeral_graph_session
        from app.graph.client import GraphClient

        async with ephemeral_graph_session() as gsession:
            if gsession is None:
                return empty
            client = GraphClient(gsession)
            features = await client.network_features(account_id, k_hops=settings.graph_k_hops)
            score = await client.compute_graph_score(account_id, k_hops=settings.graph_k_hops)
            return {"features": features, "score": score.score}
    except Exception:
        logger.warning("Graph read failed for account %s (non-fatal)", account_id, exc_info=True)
        return empty


def _build_score_explanation(
    features, feature_names, rule_result, anomaly_score, ml_score, final_score,
    graph_score: float = 0.0,
) -> dict:
    """Build a human-readable explanation of the risk score for regulators."""
    explanation = {
        "final_score": round(final_score, 4),
        "components": {
            "rule_score": round(rule_result.combined_score, 4),
            "anomaly_score": round(anomaly_score, 4),
            "ml_score": round(ml_score, 4),
            "graph_score": round(graph_score, 4),
        },
        "triggered_rules": [
            {"name": r.rule_name, "category": r.category, "severity": round(r.severity, 3)}
            for r in rule_result.triggered_rules
        ],
        "top_features": {},
    }

    # Include top features by absolute value (most influential)
    if feature_names and features is not None:
        feature_pairs = list(zip(feature_names, features.tolist()))
        feature_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        explanation["top_features"] = {
            name: round(val, 4) for name, val in feature_pairs[:10]
        }

    return explanation


async def _analyze(transaction_id: str):
    """Core analysis logic."""
    from uuid import UUID

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.features.extractor import FeatureExtractor
    from app.ml.anomaly_detector import AnomalyDetector
    from app.ml.fraud_classifier import FraudClassifier
    from app.models.alert import AlertSource
    from app.models.ctr import CTRStatus, CurrencyTransactionReport
    from app.models.rule_match import RuleMatch
    from app.models.sanctions import ScreeningStatus
    from app.models.transaction import Transaction
    from app.rules.engine import RuleEngine
    from app.services.sanctions_service import SanctionsScreeningService
    from app.services.transaction_service import TransactionService

    tx_uuid = UUID(transaction_id)

    # Create a fresh engine per task to avoid fork-safety issues with asyncpg
    task_engine = create_async_engine(settings.database_url, pool_size=5, max_overflow=2)
    task_session = async_sessionmaker(task_engine, class_=AsyncSession, expire_on_commit=False)

    async with task_session() as db:
        # 1. Load the transaction
        result = await db.execute(
            select(Transaction).where(Transaction.id == tx_uuid)
        )
        transaction = result.scalar_one_or_none()
        if not transaction:
            logger.error("Transaction %s not found", transaction_id)
            return

        service = TransactionService(db)

        # 1b. Upsert profile-asset graph edges (ADR 0007) — fail-open, never break analysis.
        await _upsert_graph_edges(transaction, transaction_id)

        # 1c. Read graph features + score once (fail-open). Done after the upsert so the
        # current transaction's edges are reflected.
        graph_data = await _graph_read(transaction.fineract_account_id)

        # 2. Get account history for feature extraction
        history_1h = await service.get_account_history(
            transaction.fineract_account_id, window_minutes=60
        )
        history_24h = await service.get_account_history(
            transaction.fineract_account_id, window_minutes=1440
        )

        # 3. Extract features (graph features appended only when AML_GRAPH_ENABLED)
        features = FeatureExtractor.extract(
            transaction, history_1h, history_24h, graph_features=graph_data["features"]
        )

        # 4. Run rule engine (uses 24h history for IP-based rules)
        rule_engine = RuleEngine()
        rule_result = rule_engine.evaluate(transaction, history_1h, history_24h)

        # Store rule matches
        for match in rule_result.triggered_rules:
            rule_match = RuleMatch(
                transaction_id=transaction.id,
                rule_name=match.rule_name,
                rule_category=match.category,
                severity=match.severity,
                details=match.details,
            )
            db.add(rule_match)

        # 5. Run anomaly detector
        anomaly_detector = AnomalyDetector()
        anomaly_score = anomaly_detector.predict(features)
        has_anomaly_model = anomaly_detector.model is not None

        # 6. Run fraud classifier (only if trained)
        fraud_classifier = FraudClassifier()
        ml_score = 0.0
        model_version = None
        if fraud_classifier.is_ready:
            ml_score, model_version = fraud_classifier.predict(features)

        # 7. Combine scores
        # Rule-heavy weighting until classifier has unbiased training data (Issue #33)
        if fraud_classifier.is_ready:
            final_score = (
                rule_result.combined_score * 0.5
                + anomaly_score * 0.3
                + ml_score * 0.2
            )
        elif has_anomaly_model:
            # No ML model yet — rely on rules + anomaly detection
            final_score = anomaly_score * 0.5 + rule_result.combined_score * 0.5
        else:
            # No trained models at all — rules are the only signal
            final_score = rule_result.combined_score

        # 7b. Graph fraud layer (ADR 0007) — guilt-by-association + shared-asset fan.
        # Uses the score read in step 1c. Fail-open: score is 0.0 when disabled/unavailable.
        graph_score = graph_data["score"]
        if settings.graph_enabled and graph_score:
            # Convex blend preserves the existing rule/anomaly/ml weighting above.
            w = settings.graph_score_weight
            final_score = final_score * (1 - w) + graph_score * w

        # 8. Build score explanation for model explainability
        explanation = _build_score_explanation(
            features,
            FeatureExtractor.get_feature_names(),
            rule_result,
            anomaly_score,
            ml_score,
            final_score,
            graph_score=graph_score,
        )

        # 9. Update transaction risk score + explanation
        await service.update_risk_score(
            transaction.id,
            risk_score=final_score,
            anomaly_score=anomaly_score,
            model_version=model_version,
            score_explanation=json.dumps(explanation),
        )

        # 10. Create alert if needed
        # Determine primary source based on which signal contributed most (Issues #24, #25)
        if rule_result.triggered_rules and rule_result.combined_score >= anomaly_score and rule_result.combined_score >= ml_score:
            source = AlertSource.RULE_ENGINE
        elif ml_score > 0 and ml_score >= anomaly_score and fraud_classifier.is_ready:
            source = AlertSource.ML_MODEL
        elif anomaly_score > 0 and has_anomaly_model:
            source = AlertSource.ANOMALY_DETECTION
        elif rule_result.triggered_rules:
            source = AlertSource.RULE_ENGINE
        else:
            source = AlertSource.RULE_ENGINE  # fallback

        await service.create_alert_if_needed(
            transaction,
            risk_score=final_score,
            source=source,
            triggered_rules=rule_result.rule_names if rule_result.triggered_rules else None,
        )

        # 11. Sanctions/PEP screening — counterparty + originator (FATF Rec 6)
        sanctions_hit = False
        if settings.sanctions_screening_enabled:
            sanctions_service = SanctionsScreeningService(db)

            # Screen counterparty
            if transaction.counterparty_name:
                cp_result = await sanctions_service.screen_name(
                    transaction.counterparty_name, str(transaction.id)
                )
                if cp_result and cp_result.status in (
                    ScreeningStatus.POTENTIAL_MATCH, ScreeningStatus.CONFIRMED_MATCH
                ):
                    sanctions_hit = True

            # Screen originating customer (FATF Rec 6 requires both parties)
            if transaction.fineract_client_id:
                orig_result = await sanctions_service.screen_name(
                    str(transaction.fineract_client_id), str(transaction.id)
                )
                if orig_result and orig_result.status in (
                    ScreeningStatus.POTENTIAL_MATCH, ScreeningStatus.CONFIRMED_MATCH
                ):
                    sanctions_hit = True

            # Auto-escalate on any sanctions match (Issue #23)
            if sanctions_hit:
                final_score = max(final_score, 0.95)
                await service.update_risk_score(
                    transaction.id,
                    risk_score=final_score,
                    anomaly_score=anomaly_score,
                    model_version=model_version,
                    score_explanation=json.dumps(explanation),
                )
                logger.warning(
                    "Sanctions match for transaction %s — escalated to CRITICAL",
                    transaction.fineract_transaction_id,
                )

        # 12. Auto-generate CTR if amount exceeds regulatory threshold
        if transaction.amount >= settings.ctr_threshold:
            ctr = CurrencyTransactionReport(
                transaction_id=transaction.id,
                fineract_client_id=transaction.fineract_client_id,
                fineract_account_id=transaction.fineract_account_id,
                amount=transaction.amount,
                currency=transaction.currency,
                transaction_type=transaction.transaction_type.value,
                status=CTRStatus.PENDING,
            )
            db.add(ctr)
            logger.info(
                "CTR auto-generated for transaction %s (amount=%.2f %s)",
                transaction.fineract_transaction_id,
                transaction.amount,
                transaction.currency,
            )

        await db.commit()

        logger.info(
            "Analysis complete for %s: rule_score=%.2f, anomaly=%.2f, ml=%.2f, final=%.2f",
            transaction.fineract_transaction_id,
            rule_result.combined_score,
            anomaly_score,
            ml_score,
            final_score,
        )

    await task_engine.dispose()


@celery_app.task(name="app.tasks.analysis.analyze_transaction", bind=True, max_retries=3)
def analyze_transaction(self, transaction_id: str):
    """Analyze a transaction for AML/fraud indicators."""
    try:
        _run_async(_analyze(transaction_id))
    except Exception as exc:
        logger.exception("Failed to analyze transaction %s", transaction_id)
        self.retry(exc=exc, countdown=10)
