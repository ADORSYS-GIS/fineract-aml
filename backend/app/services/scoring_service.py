"""Synchronous scoring service — shared logic for webhook pipeline and live scoring API.

Extracted from tasks/analysis.py so both the async Celery pipeline and
the real-time POST /api/v1/score endpoint call the same code path.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of a synchronous risk scoring pass."""

    risk_score: float
    risk_level: str                       # "low" | "medium" | "high" | "critical"
    rule_score: float
    anomaly_score: float
    ml_score: float
    triggered_rules: list[dict] = field(default_factory=list)
    recommendation: str = "monitor"       # "pass" | "monitor" | "review" | "block"
    latency_ms: float = 0.0
    degraded_mode: bool = False           # True when DB reads timed out
    graph_score: float = 0.0              # graph fraud-layer component (0 when disabled) — ADR 0007
    graph_degraded: bool = False          # True when the graph read timed out / was unavailable


def _risk_level_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def _recommendation_from_score(score: float, risk_level: str) -> str:
    if risk_level == "critical":
        return "block"
    if risk_level == "high":
        return "review"
    if risk_level == "medium":
        return "monitor"
    return "pass"


class ScoringService:
    """Performs synchronous risk scoring without writing to the database."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def score_transaction(self, payload: dict) -> ScoringResult:
        """Score a transaction payload in real-time.

        Args:
            payload: Dict matching the WebhookPayload schema fields.

        Returns:
            ScoringResult with risk score, triggered rules, and recommendation.
        """

        from app.features.extractor import FeatureExtractor
        from app.ml.anomaly_detector import AnomalyDetector
        from app.ml.fraud_classifier import FraudClassifier
        from app.models.transaction import Transaction, TransactionType
        from app.rules.engine import RuleEngine
        from app.services.transaction_service import TransactionService

        start = time.monotonic()
        degraded = False

        # Build a lightweight transaction object from the payload (no DB write)
        tx = Transaction(
            fineract_transaction_id=payload.get("transaction_id", "sync"),
            fineract_account_id=payload.get("account_id", ""),
            fineract_client_id=payload.get("client_id", ""),
            transaction_type=TransactionType(payload.get("transaction_type", "transfer")),
            amount=float(payload.get("amount", 0)),
            currency=payload.get("currency", "XAF"),
            transaction_date=datetime.now(UTC),
            counterparty_name=payload.get("counterparty_name"),
            counterparty_account_id=payload.get("counterparty_account_id"),
            actor_type=payload.get("actor_type"),
            agent_id=payload.get("agent_id"),
            merchant_id=payload.get("merchant_id"),
            kyc_level=payload.get("kyc_level"),
        )

        service = TransactionService(self.db)
        timeout = settings.sync_scoring_timeout_ms / 1000.0

        # Fetch histories with timeout to preserve latency SLO
        history_1h: list = []
        history_24h: list = []
        history_7d: list = []
        agent_history: list = []

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    service.get_account_history(tx.fineract_account_id, window_minutes=60),
                    service.get_account_history(tx.fineract_account_id, window_minutes=1440),
                    service.get_account_history(tx.fineract_account_id, window_minutes=10080),
                ),
                timeout=timeout,
            )
            history_1h, history_24h, history_7d = results[0], results[1], results[2]
            if tx.actor_type == "agent" and tx.agent_id:
                agent_history = await asyncio.wait_for(
                    service.get_agent_recent_transactions(tx.agent_id, window_minutes=1440),
                    timeout=timeout,
                )
        except TimeoutError:
            degraded = True
            logger.warning("Sync scoring: DB history fetch timed out — degraded mode")

        # Feature extraction
        features = FeatureExtractor.extract(tx, history_1h, history_24h, history_7d if history_7d else None)

        # Rule engine
        rule_engine = RuleEngine()
        rule_result = rule_engine.evaluate(
            tx,
            history_1h,
            account_history_24h=history_24h,
            account_history_7d=history_7d,
            agent_history=agent_history if agent_history else None,
        )

        # Anomaly detection
        anomaly_detector = AnomalyDetector()
        anomaly_score = anomaly_detector.predict(features)

        # ML classifier (if available)
        fraud_classifier = FraudClassifier()
        ml_score = 0.0
        if fraud_classifier.is_ready:
            ml_score, _ = fraud_classifier.predict(features)

        # Combine scores
        if fraud_classifier.is_ready:
            final_score = ml_score * 0.5 + anomaly_score * 0.3 + rule_result.combined_score * 0.2
        elif anomaly_detector.model is not None:
            final_score = anomaly_score * 0.5 + rule_result.combined_score * 0.5
        else:
            final_score = rule_result.combined_score

        # Graph fraud layer (ADR 0007) — bounded, fail-open. A timeout/error leaves the
        # combined score untouched and only flags `graph_degraded` (never blocks the path).
        graph_score = 0.0
        graph_degraded = False
        if settings.graph_enabled:
            graph_score, graph_degraded = await self._graph_component(tx.fineract_account_id)
            if not graph_degraded:
                w = settings.graph_score_weight
                final_score = final_score * (1 - w) + graph_score * w

        risk_level = _risk_level_from_score(final_score)
        recommendation = _recommendation_from_score(final_score, risk_level)
        latency_ms = (time.monotonic() - start) * 1000

        return ScoringResult(
            risk_score=round(final_score, 4),
            risk_level=risk_level,
            rule_score=round(rule_result.combined_score, 4),
            anomaly_score=round(anomaly_score, 4),
            ml_score=round(ml_score, 4),
            triggered_rules=[
                {
                    "name": r.rule_name,
                    "category": r.category,
                    "severity": round(r.severity, 3),
                }
                for r in rule_result.triggered_rules
            ],
            recommendation=recommendation,
            latency_ms=round(latency_ms, 1),
            degraded_mode=degraded,
            graph_score=round(graph_score, 4),
            graph_degraded=graph_degraded,
        )

    async def _graph_component(self, account_id: str) -> tuple[float, bool]:
        """Compute the bounded graph score for an account on the live path.

        Returns (score, degraded). `degraded` is True (and score 0.0) on timeout, missing
        account id, graph unavailability, or any error — the caller then skips the graph term.
        """
        if not account_id:
            return 0.0, True
        try:
            from app.core.graph import graph_session
            from app.graph.client import GraphClient

            budget = settings.graph_scoring_timeout_ms / 1000.0

            async def _run() -> float:
                async with graph_session() as gsession:
                    if gsession is None:
                        raise RuntimeError("graph unavailable")
                    result = await GraphClient(gsession).compute_graph_score(
                        account_id, k_hops=settings.graph_k_hops
                    )
                    return result.score

            score = await asyncio.wait_for(_run(), timeout=budget)
            return score, False
        except Exception:  # includes asyncio.TimeoutError — fail-open
            logger.warning("Live graph score degraded for account %s", account_id, exc_info=True)
            return 0.0, True
