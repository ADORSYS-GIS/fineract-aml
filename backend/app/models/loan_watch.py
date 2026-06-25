"""Loan disbursement watch — tracks loans under post-disbursement monitoring.

After a LOAN_DISBURSEMENT transaction is processed, a LoanDisbursementWatch
record is created and two Celery tasks are scheduled (24h and 48h later)
to check how the funds were used.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class LoanWatchStatus(str, enum.Enum):
    ACTIVE = "active"       # Watch in progress, no issues found yet
    FLAGGED = "flagged"     # Suspicious behavior detected at 24h or 48h check
    CLEARED = "cleared"     # 48h passed with no suspicious behavior


class LoanDisbursementWatch(Base, TimestampMixin):
    """Tracks a loan disbursement through its first 48 hours for fraud detection."""

    __tablename__ = "loan_disbursement_watches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # The disbursement transaction being watched
    loan_transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    fineract_client_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    fineract_account_id: Mapped[str] = mapped_column(String(100), nullable=False)

    # Loan details at disbursement time
    disbursed_amount: Mapped[float] = mapped_column(Float, nullable=False)
    disbursed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)

    # Scheduled check timestamps
    check_24h_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    check_48h_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Current status
    status: Mapped[LoanWatchStatus] = mapped_column(
        Enum(LoanWatchStatus, values_callable=lambda e: [x.value for x in e]),
        nullable=False,
        default=LoanWatchStatus.ACTIVE,
    )

    # Findings from each check (JSON: {"24h": {...}, "48h": {...}})
    findings_json: Mapped[str | None] = mapped_column(Text)

    # Alert created if flagged
    alert_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
