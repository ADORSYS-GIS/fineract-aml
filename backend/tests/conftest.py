"""Shared test fixtures and helpers."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FakeTransaction:
    """Lightweight transaction object for unit tests (no SQLAlchemy instrumentation)."""

    id: Any = field(default_factory=uuid.uuid4)
    fineract_transaction_id: str = ""
    fineract_account_id: str = "ACC-001"
    fineract_client_id: str = "CLI-001"
    transaction_type: Any = None
    amount: float = 500.0
    currency: str = "USD"
    transaction_date: datetime = field(
        default_factory=lambda: datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc)
    )
    counterparty_account_id: str | None = None
    counterparty_name: str | None = None
    risk_score: float | None = None
    risk_level: Any = None
    anomaly_score: float | None = None
    model_version: str | None = None
    description: str | None = None
    raw_payload: str | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    country_code: str | None = None
    geo_location: str | None = None
    actor_type: str | None = None      # "customer" | "agent" | "merchant"
    agent_id: str | None = None
    merchant_id: str | None = None
    device_id: str | None = None       # SHA-256 device fingerprint (graph asset node)
    kyc_level: int | None = None
    shadow_score: float | None = None

    def __post_init__(self):
        from app.models.transaction import TransactionType

        if not self.fineract_transaction_id:
            self.fineract_transaction_id = f"TX-{uuid.uuid4().hex[:8]}"
        if self.transaction_type is None:
            self.transaction_type = TransactionType.DEPOSIT
