"""Fineract polling fallback — catches transactions missed by webhooks.

Polls the Fineract API for recent transactions and ingests any that are not
yet in the AML database. Runs every 60 seconds as a safety net when webhooks
fail or are delayed.
"""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_fineract_date(parts) -> datetime:
    """Parse Fineract's [year, month, day] array into a UTC datetime."""
    if isinstance(parts, list) and len(parts) >= 3:
        return datetime(parts[0], parts[1], parts[2], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _map_fineract_tx_type(raw: dict) -> str:
    """Map Fineract transactionType.code to our TransactionType enum value."""
    code = (raw.get("transactionType") or {}).get("code", "")
    mapping = {
        "savingsAccountTransactionType.deposit": "deposit",
        "savingsAccountTransactionType.withdrawal": "withdrawal",
        "savingsAccountTransactionType.transfer": "transfer",
    }
    return mapping.get(code, "other")


async def _poll_fineract():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.core.config import settings
    from app.models.transaction import Transaction
    from app.schemas.transaction import WebhookPayload
    from app.services.data_quality_service import DataQualityService
    from app.services.transaction_service import TransactionService

    if not settings.fineract_base_url:
        logger.debug("Fineract base URL not configured, skipping poll")
        return

    import os
    tls_verify = os.getenv("FINERACT_TLS_VERIFY", "true").lower() != "false"

    try:
        async with httpx.AsyncClient(verify=tls_verify, timeout=30) as client:
            response = await client.get(
                f"{settings.fineract_base_url}/savingsaccounts/transactions",
                params={"limit": 100, "orderBy": "id", "sortOrder": "DESC"},
                headers={"Fineract-Platform-TenantId": "default"},
            )
            if response.status_code != 200:
                logger.warning("Fineract poll returned status %d", response.status_code)
                return
            data = response.json()
    except httpx.RequestError as e:
        if isinstance(e, httpx.ConnectError) and "ssl" in str(e).lower():
            logger.warning(
                "Fineract poll failed (SSL/TLS error — set FINERACT_TLS_VERIFY=false for self-signed certs): %s", e
            )
        else:
            logger.debug("Fineract poll failed (expected if Fineract is not running): %s", e)
        return

    if not data or "pageItems" not in data:
        return

    engine = create_async_engine(settings.database_url, pool_size=3, max_overflow=1)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    dq = DataQualityService()

    ingested = 0
    try:
        async with session_maker() as db:
            service = TransactionService(db)
            for item in data["pageItems"]:
                fineract_tx_id = str(item.get("id", ""))
                if not fineract_tx_id:
                    continue

                existing = await db.execute(
                    select(Transaction.id).where(
                        Transaction.fineract_transaction_id == fineract_tx_id
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                # Build a minimal WebhookPayload from Fineract's list response.
                # Fields like counterparty and actor context are unavailable at this level;
                # they default to None and the analysis pipeline handles the gaps gracefully.
                currency_data = item.get("currency") or {}
                raw_amount = item.get("amount", 0)
                if isinstance(raw_amount, float):
                    # Round to nearest integer — XAF has no sub-unit (money invariant)
                    raw_amount = round(raw_amount)

                if raw_amount == 0:
                    logger.debug(
                        "Polling: skipping zero-amount transaction %s (fee reversal or adjustment)",
                        fineract_tx_id,
                    )
                    continue

                try:
                    payload = WebhookPayload(
                        transaction_id=fineract_tx_id,
                        account_id=str(item.get("accountId") or item.get("savingsAccountId") or ""),
                        client_id=str(item.get("clientId") or ""),
                        transaction_type=_map_fineract_tx_type(item),
                        amount=raw_amount,
                        currency=currency_data.get("code", settings.default_currency),
                        transaction_date=_parse_fineract_date(item.get("date")),
                    )
                except Exception as exc:
                    logger.warning("Polling: could not build payload for tx %s: %s", fineract_tx_id, exc)
                    continue

                dq_result = dq.validate(payload)
                if not dq_result.is_valid:
                    logger.warning(
                        "Polling: skipping invalid payload for tx %s: %s",
                        fineract_tx_id,
                        getattr(dq_result, "errors", dq_result.warnings),
                    )
                    continue
                transaction, is_new = await service.ingest_transaction(
                    payload,
                    data_quality_warnings=dq_result.warnings or None,
                )

                if is_new:
                    await db.commit()  # commit before dispatch so the Celery worker finds the row
                    from app.tasks.analysis import analyze_transaction
                    analyze_transaction.delay(str(transaction.id))
                    ingested += 1
                    logger.info("Polling ingested missing transaction %s", fineract_tx_id)
    finally:
        await engine.dispose()

    if ingested:
        logger.info("Polling: ingested %d missing transactions", ingested)


@celery_app.task(name="app.tasks.polling.poll_fineract_transactions")
def poll_fineract_transactions():
    """Poll Fineract API for transactions not received via webhook."""
    _run_async(_poll_fineract())
