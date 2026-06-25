"""Synchronous risk scoring endpoint.

Scores a transaction payload in real-time (< 400ms target latency)
without writing to the database. Used by the BFF for pre-transaction
risk assessment and integration testing.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_token
from app.services.scoring_service import ScoringService

router = APIRouter(
    prefix="/score", tags=["Scoring"], dependencies=[Depends(verify_token)]
)


class ScoringRequest(BaseModel):
    transaction_id: str = Field(..., description="Idempotency key / Fineract transaction ID")
    account_id: str
    client_id: str
    transaction_type: str = Field(..., description="deposit | withdrawal | transfer | loan_disbursement")
    amount: float = Field(..., gt=0)
    currency: str = "XAF"
    counterparty_name: str | None = None
    counterparty_account_id: str | None = None
    # Actor context (Phase 1)
    actor_type: str | None = Field(default=None, description="customer | agent | merchant")
    agent_id: str | None = None
    merchant_id: str | None = None
    kyc_level: int | None = Field(default=None, ge=1, le=4)


class ScoringResponse(BaseModel):
    risk_score: float = Field(..., description="0.0–1.0 combined risk score")
    risk_level: str = Field(..., description="low | medium | high | critical")
    rule_score: float
    anomaly_score: float
    ml_score: float
    triggered_rules: list[dict]
    recommendation: str = Field(..., description="pass | monitor | review | block")
    latency_ms: float
    degraded_mode: bool = Field(
        default=False,
        description="True when DB history fetch timed out; score is rules-only",
    )
    graph_score: float = Field(
        default=0.0,
        description="Graph fraud-layer component (0.0 when disabled/degraded) — ADR 0007",
    )
    graph_degraded: bool = Field(
        default=False,
        description="True when the graph read timed out or was unavailable; graph term skipped",
    )


@router.post("", response_model=ScoringResponse)
async def score_transaction(
    request: ScoringRequest,
    db: AsyncSession = Depends(get_db),
):
    """Score a transaction synchronously without persisting it.

    Returns the risk score and triggered rules within the configured
    latency budget (default 400ms). Falls back to rules-only scoring
    if the DB history read times out (`degraded_mode: true`).
    """
    from app.core.config import settings

    if not settings.sync_scoring_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Synchronous scoring is disabled (SYNC_SCORING_ENABLED=false)",
        )

    service = ScoringService(db)
    result = await service.score_transaction(request.model_dump())
    return ScoringResponse(
        risk_score=result.risk_score,
        risk_level=result.risk_level,
        rule_score=result.rule_score,
        anomaly_score=result.anomaly_score,
        ml_score=result.ml_score,
        triggered_rules=result.triggered_rules,
        recommendation=result.recommendation,
        latency_ms=result.latency_ms,
        degraded_mode=result.degraded_mode,
        graph_score=result.graph_score,
        graph_degraded=result.graph_degraded,
    )
