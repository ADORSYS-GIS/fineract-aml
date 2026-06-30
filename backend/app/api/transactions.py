"""Transaction endpoints for the compliance dashboard."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_token
from app.models.transaction import RiskLevel
from app.schemas.transaction import TransactionListResponse, TransactionResponse, TransactionStats
from app.services.transaction_service import TransactionService

router = APIRouter(
    prefix="/transactions", tags=["Transactions"], dependencies=[Depends(verify_token)]
)


@router.get("", response_model=TransactionListResponse)
async def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    risk_level: RiskLevel | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List all transactions with optional risk level filter."""
    service = TransactionService(db)
    transactions, total = await service.list_transactions(page, page_size, risk_level)
    return TransactionListResponse(
        items=[TransactionResponse.model_validate(t) for t in transactions],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=TransactionStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Get dashboard statistics."""
    service = TransactionService(db)
    return await service.get_stats()


@router.get("/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single transaction by AML internal UUID."""
    service = TransactionService(db)
    transaction = await service.get_transaction(transaction_id)
    if not transaction:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found"
        )
    return TransactionResponse.model_validate(transaction)


@router.get("/by-fineract-id/{fineract_tx_id}", response_model=TransactionResponse)
async def get_transaction_by_fineract_id(
    fineract_tx_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Look up a transaction by its Fineract transaction ID.

    Used by the BFF to poll AML score results after a transaction is submitted
    (pull-based post-transaction feedback, webank-mobile#139 item 4).
    """
    from sqlalchemy import select

    from app.models.transaction import Transaction
    from fastapi import HTTPException, status

    result = await db.execute(
        select(Transaction).where(Transaction.fineract_transaction_id == fineract_tx_id)
    )
    transaction = result.scalar_one_or_none()
    if not transaction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found"
        )
    return TransactionResponse.model_validate(transaction)
