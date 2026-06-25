"""Currency Transaction Report (CTR) management API.

COBAC requires CTRs to be filed for transactions above 5,000,000 XAF.
This API allows compliance officers to manage the CTR filing workflow.
"""
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import UserRole, require_role, verify_token

router = APIRouter(prefix="/ctrs", tags=["CTR Management"])


class CTRResponse(BaseModel):
    id: str
    fineract_client_id: str
    fineract_account_id: str
    amount: float
    currency: str
    transaction_type: str
    status: str
    agent_id: str | None
    branch_id: str | None
    counterparty_name: str | None
    counterparty_account: str | None
    filed_at: str | None
    cobac_reference: str | None
    created_at: str


class CTRFilingUpdate(BaseModel):
    cobac_reference: str | None = None
    notes: str | None = None


@router.get(
    "",
    response_model=list[CTRResponse],
    dependencies=[Depends(require_role(UserRole.ANALYST, UserRole.MLRO, UserRole.ADMIN))],
)
async def list_ctrs(
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List CTRs with optional status filter. Requires ANALYST or MLRO role."""
    from app.models.ctr import CurrencyTransactionReport

    query = select(CurrencyTransactionReport).order_by(
        CurrencyTransactionReport.created_at.desc()
    )
    if status_filter:
        query = query.where(CurrencyTransactionReport.status == status_filter)
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    ctrs = result.scalars().all()

    return [
        CTRResponse(
            id=str(c.id),
            fineract_client_id=c.fineract_client_id,
            fineract_account_id=c.fineract_account_id,
            amount=c.amount,
            currency=c.currency,
            transaction_type=c.transaction_type,
            status=str(c.status.value if hasattr(c.status, "value") else c.status),
            agent_id=getattr(c, "agent_id", None),
            branch_id=getattr(c, "branch_id", None),
            counterparty_name=getattr(c, "counterparty_name", None),
            counterparty_account=getattr(c, "counterparty_account", None),
            filed_at=c.filed_at.isoformat() if getattr(c, "filed_at", None) else None,
            cobac_reference=getattr(c, "cobac_reference", None),
            created_at=c.created_at.isoformat(),
        )
        for c in ctrs
    ]


@router.post(
    "/{ctr_id}/file",
    response_model=CTRResponse,
    dependencies=[Depends(require_role(UserRole.MLRO, UserRole.ADMIN))],
)
async def mark_ctr_filed(
    ctr_id: UUID,
    body: CTRFilingUpdate,
    current_user: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """Mark a CTR as filed with COBAC. Requires MLRO role."""
    from app.models.ctr import CurrencyTransactionReport

    result = await db.execute(
        select(CurrencyTransactionReport).where(CurrencyTransactionReport.id == ctr_id)
    )
    ctr = result.scalar_one_or_none()
    if not ctr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CTR not found")

    ctr.status = "filed"
    ctr.filed_at = datetime.now(UTC)
    if hasattr(ctr, "filed_by_user_id"):
        ctr.filed_by_user_id = current_user.get("user_id")
    if body.cobac_reference:
        ctr.cobac_reference = body.cobac_reference
    if body.notes:
        ctr.notes = body.notes

    await db.commit()
    await db.refresh(ctr)

    return CTRResponse(
        id=str(ctr.id),
        fineract_client_id=ctr.fineract_client_id,
        fineract_account_id=ctr.fineract_account_id,
        amount=ctr.amount,
        currency=ctr.currency,
        transaction_type=ctr.transaction_type,
        status=str(ctr.status.value if hasattr(ctr.status, "value") else ctr.status),
        agent_id=getattr(ctr, "agent_id", None),
        branch_id=getattr(ctr, "branch_id", None),
        counterparty_name=getattr(ctr, "counterparty_name", None),
        counterparty_account=getattr(ctr, "counterparty_account", None),
        filed_at=ctr.filed_at.isoformat() if ctr.filed_at else None,
        cobac_reference=getattr(ctr, "cobac_reference", None),
        created_at=ctr.created_at.isoformat(),
    )


@router.post(
    "/{ctr_id}/acknowledge",
    response_model=CTRResponse,
    dependencies=[Depends(require_role(UserRole.MLRO, UserRole.ADMIN))],
)
async def mark_ctr_acknowledged(
    ctr_id: UUID,
    body: CTRFilingUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Record COBAC acknowledgement of a filed CTR. Requires MLRO role."""
    from app.models.ctr import CurrencyTransactionReport

    result = await db.execute(
        select(CurrencyTransactionReport).where(CurrencyTransactionReport.id == ctr_id)
    )
    ctr = result.scalar_one_or_none()
    if not ctr:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CTR not found")

    ctr.status = "acknowledged"
    if body.cobac_reference:
        ctr.cobac_reference = body.cobac_reference
    if body.notes:
        ctr.notes = body.notes

    await db.commit()
    await db.refresh(ctr)

    return CTRResponse(
        id=str(ctr.id),
        fineract_client_id=ctr.fineract_client_id,
        fineract_account_id=ctr.fineract_account_id,
        amount=ctr.amount,
        currency=ctr.currency,
        transaction_type=ctr.transaction_type,
        status=str(ctr.status.value if hasattr(ctr.status, "value") else ctr.status),
        agent_id=getattr(ctr, "agent_id", None),
        branch_id=getattr(ctr, "branch_id", None),
        counterparty_name=getattr(ctr, "counterparty_name", None),
        counterparty_account=getattr(ctr, "counterparty_account", None),
        filed_at=ctr.filed_at.isoformat() if getattr(ctr, "filed_at", None) else None,
        cobac_reference=getattr(ctr, "cobac_reference", None),
        created_at=ctr.created_at.isoformat(),
    )
