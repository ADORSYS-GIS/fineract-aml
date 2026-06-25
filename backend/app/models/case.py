"""Case model — groups related alerts into an investigation case."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    CLOSED_LEGITIMATE = "closed_legitimate"
    CLOSED_FRAUD = "closed_fraud"
    SAR_FILED = "sar_filed"


class Case(Base, TimestampMixin):
    """An investigation case grouping related suspicious transactions."""

    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_number: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus, values_callable=lambda e: [x.value for x in e]), default=CaseStatus.OPEN, nullable=False
    )
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )
    fineract_client_id: Mapped[str | None] = mapped_column(String(100), index=True)

    # Escalation and SLA tracking
    escalation_reason: Mapped[str | None] = mapped_column(Text)
    sla_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # SAR document
    sar_document_path: Mapped[str | None] = mapped_column(String(512))

    # COBAC audit trail — set when case reaches a terminal/closed state
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)  # username who closed

    # Relationships
    assignee: Mapped["User | None"] = relationship()  # noqa: F821
    transactions: Mapped[list["CaseTransaction"]] = relationship(back_populates="case")


class CaseTransaction(Base, TimestampMixin):
    """Links transactions to cases (many-to-many)."""

    __tablename__ = "case_transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    case: Mapped["Case"] = relationship(back_populates="transactions")
    transaction: Mapped["Transaction"] = relationship()  # noqa: F821
