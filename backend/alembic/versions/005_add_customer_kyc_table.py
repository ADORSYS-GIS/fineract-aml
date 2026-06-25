"""Add customer KYC/KYB table.

Revision ID: 005
Revises: 004
Create Date: 2026-03-21

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM, UUID

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

customer_risk_enum = ENUM("low", "medium", "high", name="customerrisklevel", create_type=False)
customer_type_enum = ENUM("individual", "entity", name="customertype", create_type=False)


def upgrade() -> None:
    op.execute("CREATE TYPE customerrisklevel AS ENUM ('low', 'medium', 'high')")
    op.execute("CREATE TYPE customertype AS ENUM ('individual', 'entity')")

    op.create_table(
        "customers",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # Identity
        sa.Column("fineract_client_id", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("full_name", sa.String(500), nullable=False),
        sa.Column("customer_type", customer_type_enum, nullable=False, server_default="individual"),
        sa.Column("date_of_birth", sa.DateTime),
        sa.Column("nationality", sa.String(2)),
        sa.Column("country_of_residence", sa.String(2)),
        # Identification
        sa.Column("id_type", sa.String(50)),
        sa.Column("id_number", sa.String(100)),
        sa.Column("id_expiry", sa.DateTime),
        # Contact
        sa.Column("email", sa.String(255)),
        sa.Column("phone", sa.String(50)),
        sa.Column("address", sa.Text),
        # Business info
        sa.Column("business_name", sa.String(500)),
        sa.Column("registration_number", sa.String(100)),
        sa.Column("beneficial_owners", sa.Text),
        # Risk
        sa.Column("risk_level", customer_risk_enum, nullable=False, server_default="low"),
        sa.Column("is_pep", sa.Boolean, server_default="false"),
        sa.Column("pep_details", sa.Text),
        sa.Column("is_sanctioned", sa.Boolean, server_default="false"),
        sa.Column("sanctions_details", sa.Text),
        # EDD
        sa.Column("edd_required", sa.Boolean, server_default="false"),
        sa.Column("edd_reason", sa.String(500)),
        sa.Column("edd_completed_at", sa.DateTime(timezone=True)),
        # KYC/Sync
        sa.Column("kyc_verified", sa.Boolean, server_default="false"),
        sa.Column("kyc_verified_at", sa.DateTime(timezone=True)),
        sa.Column("last_synced_at", sa.DateTime(timezone=True)),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_customers_risk", "customers", ["risk_level"])
    op.create_index("ix_customers_pep", "customers", ["is_pep"])
    op.create_index("ix_customers_sanctioned", "customers", ["is_sanctioned"])


def downgrade() -> None:
    op.drop_table("customers")
    op.execute("DROP TYPE IF EXISTS customertype")
    op.execute("DROP TYPE IF EXISTS customerrisklevel")
