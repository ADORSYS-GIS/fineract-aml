"""Alert model — generated when a transaction is flagged as suspicious."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class AlertStatus(str, enum.Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"


class AlertSource(str, enum.Enum):
    RULE_ENGINE = "rule_engine"
    ANOMALY_DETECTION = "anomaly_detection"
    ML_MODEL = "ml_model"
    MANUAL = "manual"
    LOAN_MONITORING = "loan_monitoring"
    GRAPH_ANALYSIS = "graph_analysis"  # shared-asset ring / guilt-by-association (ADR 0007)


class Alert(Base, TimestampMixin):
    """An alert raised when suspicious activity is detected."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    transaction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=False
    )

    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, values_callable=lambda e: [x.value for x in e]), default=AlertStatus.PENDING, nullable=False
    )
    source: Mapped[AlertSource] = mapped_column(Enum(AlertSource, values_callable=lambda e: [x.value for x in e]), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    triggered_rules: Mapped[str | None] = mapped_column(Text)

    # LLM investigation report (JSON: summary, typology_match, risk_factors, etc.)
    investigation_report: Mapped[str | None] = mapped_column(Text)

    # COBAC audit trail — set when alert reaches a terminal state
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)  # username who closed

    assigned_to: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id")
    )

    # Relationships
    transaction: Mapped["Transaction"] = relationship(back_populates="alerts")  # noqa: F821
    assignee: Mapped["User | None"] = relationship()  # noqa: F821
    reviews: Mapped[list["Review"]] = relationship(back_populates="alert")  # noqa: F821

    __table_args__ = (
        Index("ix_alerts_status", "status"),
        Index("ix_alerts_assigned", "assigned_to", "status"),
    )
