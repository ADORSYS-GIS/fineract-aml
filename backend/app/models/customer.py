"""Customer model — KYC/KYB data cached from Fineract.

Stores customer due diligence information pulled from Fineract's client API.
Enables Enhanced Due Diligence (EDD) triggers based on customer risk profile.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class CustomerRiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CustomerType(str, enum.Enum):
    INDIVIDUAL = "individual"
    ENTITY = "entity"  # Business/organization


class Customer(Base, TimestampMixin):
    """Cached KYC/KYB data for a Fineract client.

    Synced from Fineract client API. Used for:
    - Customer Due Diligence (CDD)
    - Enhanced Due Diligence (EDD) triggers
    - PEP/sanctions cross-referencing
    - Beneficial ownership tracking
    """

    __tablename__ = "customers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    fineract_client_id: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )

    # Basic identity
    full_name: Mapped[str] = mapped_column(String(500), nullable=False)
    customer_type: Mapped[CustomerType] = mapped_column(
        Enum(CustomerType, values_callable=lambda e: [x.value for x in e]),
        default=CustomerType.INDIVIDUAL,
        nullable=False,
    )
    date_of_birth: Mapped[datetime | None] = mapped_column(DateTime)
    nationality: Mapped[str | None] = mapped_column(String(2))  # ISO 3166-1 alpha-2
    country_of_residence: Mapped[str | None] = mapped_column(String(2))

    # Identification
    id_type: Mapped[str | None] = mapped_column(String(50))  # passport, national_id, etc.
    id_number: Mapped[str | None] = mapped_column(String(100))
    id_expiry: Mapped[datetime | None] = mapped_column(DateTime)

    # Contact
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))
    address: Mapped[str | None] = mapped_column(Text)

    # Business info (for entities)
    business_name: Mapped[str | None] = mapped_column(String(500))
    registration_number: Mapped[str | None] = mapped_column(String(100))
    beneficial_owners: Mapped[str | None] = mapped_column(Text)  # JSON list of owners

    # Risk classification
    risk_level: Mapped[CustomerRiskLevel] = mapped_column(
        Enum(CustomerRiskLevel, values_callable=lambda e: [x.value for x in e]),
        default=CustomerRiskLevel.LOW,
        nullable=False,
    )
    is_pep: Mapped[bool] = mapped_column(Boolean, default=False)
    pep_details: Mapped[str | None] = mapped_column(Text)  # JSON
    is_sanctioned: Mapped[bool] = mapped_column(Boolean, default=False)
    sanctions_details: Mapped[str | None] = mapped_column(Text)  # JSON

    # EDD triggers
    edd_required: Mapped[bool] = mapped_column(Boolean, default=False)
    edd_reason: Mapped[str | None] = mapped_column(String(500))
    edd_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Sync metadata
    kyc_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    kyc_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index("ix_customers_risk", "risk_level"),
        Index("ix_customers_pep", "is_pep"),
        Index("ix_customers_sanctioned", "is_sanctioned"),
    )
