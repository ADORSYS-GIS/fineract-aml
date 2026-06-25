"""Currency Transaction Report (CTR) model — auto-generated for large transactions."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class CTRStatus(str, enum.Enum):
    PENDING = "pending"
    FILED = "filed"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"


class CurrencyTransactionReport(Base, TimestampMixin):
    """Auto-generated CTR for transactions exceeding the regulatory threshold.

    CEMAC/COBAC regulations require automatic reporting of transactions
    above a configurable threshold (e.g. 5,000,000 XAF).
    """

    __tablename__ = "currency_transaction_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    fineract_client_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    fineract_account_id: Mapped[str] = mapped_column(String(100), nullable=False)

    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)

    status: Mapped[CTRStatus] = mapped_column(
        Enum(CTRStatus, values_callable=lambda e: [x.value for x in e]),
        default=CTRStatus.PENDING,
        nullable=False,
    )
    reference_number: Mapped[str | None] = mapped_column(String(100))
    filed_by: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str | None] = mapped_column(Text)

    # COBAC required fields
    filed_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    agent_id: Mapped[str | None] = mapped_column(String(100), nullable=True)  # processing agent
    branch_id: Mapped[str | None] = mapped_column(String(100), nullable=True)  # processing branch
    counterparty_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    counterparty_account: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cobac_reference: Mapped[str | None] = mapped_column(String(100), nullable=True)  # COBAC ack number

    __table_args__ = (
        Index("ix_ctr_status", "status"),
    )
