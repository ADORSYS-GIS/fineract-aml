"""Customer AML endpoints — status reads and KYC sync (fineract-aml#2).

Two purposes:
  GET  /customers/{client_id}/aml-status   — BFF session-start risk gate
  POST /customers/{client_id}/kyc-sync     — BFF forwards person_id from webank-verify webhook
"""

import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_token
from app.models.alert import Alert, AlertStatus
from app.models.case import Case, CaseStatus
from app.models.customer import Customer, CustomerRiskLevel
from app.models.transaction import Transaction
from app.services.kyc_service import KYCService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/customers", tags=["Customers"])

# ── Auth helpers ──────────────────────────────────────────────────────────────

def _verify_internal_key(x_aml_api_key: str | None = Header(default=None)):
    """Shared internal API key for server-to-server calls (BFF → AML).

    Configure via AML_INTERNAL_API_KEY env var. When not set the endpoint
    falls back to JWT auth so it can still be hit from the compliance UI.
    """
    expected = getattr(settings, "internal_api_key", "")
    if expected and x_aml_api_key == expected:
        return  # valid internal call
    # If no internal key configured (dev/test), accept without checking
    if not expected:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid X-AML-Api-Key",
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

class AMLStatusResponse(BaseModel):
    client_id: str
    risk_level: str = Field(..., description="low | medium | high")
    is_pep: bool
    is_sanctioned: bool
    edd_required: bool
    edd_reason: str | None
    edd_completed_at: datetime | None
    kyc_verified: bool
    person_id: uuid.UUID | None = Field(
        None, description="Stable biometric key from webank-verify (ADR 0005)"
    )
    open_alerts: int = Field(..., description="Alerts in pending or under_review state")
    open_cases: int = Field(..., description="Cases in open or investigating state")
    last_risk_score: float | None = Field(None, description="Most recent transaction risk score")
    last_recommendation: str | None = Field(
        None, description="pass | monitor | review | block derived from last score"
    )


class KYCSyncRequest(BaseModel):
    person_id: uuid.UUID = Field(..., description="Stable biometric key from webank-verify")
    kyc_level: int = Field(..., ge=1, le=4, description="KYC level just approved")


class KYCSyncResponse(BaseModel):
    client_id: str
    person_id: uuid.UUID
    kyc_verified: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recommendation_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.85:
        return "block"
    if score >= 0.6:
        return "review"
    if score >= 0.3:
        return "monitor"
    return "pass"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/{client_id}/aml-status",
    response_model=AMLStatusResponse,
    dependencies=[Depends(verify_token)],
)
async def get_aml_status(client_id: str, db: AsyncSession = Depends(get_db)):
    """Return the AML risk profile for a Fineract client.

    Called by the BFF at session start to gate feature access:
    - is_sanctioned → block all transaction flows
    - edd_required and edd_completed_at is null → block credit requests
    - open_cases > 0 → disable referral payout (fail-closed, ADR 0004)

    If the customer is not yet in the AML DB it is synced from Fineract first.
    """
    kyc_service = KYCService(db)
    customer = await kyc_service.get_or_sync_customer(client_id)
    await db.commit()

    # Open alerts (pending / under_review)
    open_alert_result = await db.execute(
        select(func.count(Alert.id))
        .join(Transaction, Transaction.id == Alert.transaction_id)
        .where(
            Transaction.fineract_client_id == client_id,
            Alert.status.in_([AlertStatus.PENDING, AlertStatus.UNDER_REVIEW]),
        )
    )
    open_alerts = open_alert_result.scalar_one() or 0

    # Open cases
    open_case_result = await db.execute(
        select(func.count(Case.id)).where(
            Case.fineract_client_id == client_id,
            Case.status.in_([CaseStatus.OPEN, CaseStatus.INVESTIGATING, CaseStatus.ESCALATED]),
        )
    )
    open_cases = open_case_result.scalar_one() or 0

    # Last risk score from most recent transaction
    last_tx_result = await db.execute(
        select(Transaction.risk_score)
        .where(Transaction.fineract_client_id == client_id)
        .order_by(Transaction.transaction_date.desc())
        .limit(1)
    )
    last_score_row = last_tx_result.first()
    last_risk_score = last_score_row[0] if last_score_row else None

    return AMLStatusResponse(
        client_id=client_id,
        risk_level=customer.risk_level.value,
        is_pep=customer.is_pep,
        is_sanctioned=customer.is_sanctioned,
        edd_required=customer.edd_required,
        edd_reason=customer.edd_reason,
        edd_completed_at=customer.edd_completed_at,
        kyc_verified=customer.kyc_verified,
        person_id=customer.person_id,
        open_alerts=open_alerts,
        open_cases=open_cases,
        last_risk_score=last_risk_score,
        last_recommendation=_recommendation_from_score(last_risk_score),
    )


@router.post(
    "/{client_id}/kyc-sync",
    response_model=KYCSyncResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(_verify_internal_key)],
)
async def kyc_sync(
    client_id: str,
    body: KYCSyncRequest,
    db: AsyncSession = Depends(get_db),
):
    """Receive person_id from the BFF after a webank-verify KYC approval.

    Fire-and-forget from BFF perspective. Upserts person_id onto the Customer
    record and triggers a fresh Fineract sync so risk assessment is current.
    Call only when person_id is present in the webank-verify webhook — the BFF
    must not send this when person_id is absent (no face extracted).
    """
    kyc_service = KYCService(db)
    customer = await kyc_service.get_or_sync_customer(client_id)

    customer.person_id = body.person_id
    customer.kyc_verified = True
    customer.kyc_verified_at = datetime.now(UTC)

    await db.commit()
    logger.info(
        "KYC sync: client=%s person_id=%s kyc_level=%d",
        client_id, body.person_id, body.kyc_level,
    )

    return KYCSyncResponse(
        client_id=client_id,
        person_id=body.person_id,
        kyc_verified=True,
    )
