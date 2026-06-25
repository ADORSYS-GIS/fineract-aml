"""Webhook endpoint — receives transaction events from Fineract."""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import verify_webhook_signature
from app.schemas.transaction import WebhookPayload
from app.services.data_quality_service import DataQualityService
from app.services.transaction_service import TransactionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook", tags=["Webhook"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/fineract", status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("100/minute")
async def receive_fineract_webhook(
    request: Request,
    payload: WebhookPayload,
    db: AsyncSession = Depends(get_db),
    x_webhook_signature: str | None = Header(default=None),
):
    """Receive a transaction event from Fineract.

    This endpoint:
    1. Verifies the webhook signature (if configured)
    2. Ingests the transaction into the database
    3. Triggers async analysis (rule engine + anomaly detection)
    """
    # Always verify webhook signature unless explicitly in debug mode
    if not settings.debug:
        if not x_webhook_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing webhook signature",
            )
        raw_body = await request.body()
        if not verify_webhook_signature(raw_body, x_webhook_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )

    # Capture client IP from request if not in payload
    if not payload.ip_address and request.client:
        payload.ip_address = request.client.host

    # Data quality validation
    dq = DataQualityService()
    result = dq.validate(payload)
    if not result.is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"data_quality_errors": result.errors},
        )
    if result.warnings:
        logger.warning(
            "Data quality warnings for transaction %s: %s",
            payload.transaction_id,
            result.warnings,
        )

    service = TransactionService(db)
    transaction, is_new = await service.ingest_transaction(
        payload, data_quality_warnings=result.warnings or None
    )

    if is_new:
        # Only trigger analysis for new transactions (idempotency)
        from app.tasks.analysis import analyze_transaction

        analyze_transaction.delay(str(transaction.id))

    return {
        "status": "accepted",
        "transaction_id": str(transaction.id),
        "message": "Transaction queued for AML analysis" if is_new else "Transaction already processed",
    }
