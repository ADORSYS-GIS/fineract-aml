"""Add agent_profiles and loan_disbursement_watches tables; update alert source enum.

Revision ID: 007
Revises: 006
Create Date: 2026-04-02

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- agent_profiles table ---
    op.create_table(
        "agent_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("agent_id", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("branch_id", sa.String(100), nullable=True),
        sa.Column("avg_daily_tx_count_30d", sa.Float(), nullable=True),
        sa.Column("avg_daily_volume_30d", sa.Float(), nullable=True),
        sa.Column("std_daily_volume_30d", sa.Float(), nullable=True),
        sa.Column("typical_float_ratio", sa.Float(), nullable=True),
        sa.Column("served_customer_count_30d", sa.Integer(), nullable=True),
        sa.Column("avg_new_customers_per_day_30d", sa.Float(), nullable=True),
        sa.Column("unique_customers_30d", sa.Integer(), nullable=True),
        sa.Column("avg_tx_amount_30d", sa.Float(), nullable=True),
        sa.Column("p95_tx_amount_30d", sa.Float(), nullable=True),
        sa.Column("peak_hour_distribution", sa.Text(), nullable=True),
        sa.Column("computed_from_days", sa.Integer(), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- loan_disbursement_watches table ---
    op.create_table(
        "loan_disbursement_watches",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("loan_transaction_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("fineract_client_id", sa.String(100), nullable=False, index=True),
        sa.Column("fineract_account_id", sa.String(100), nullable=False),
        sa.Column("disbursed_amount", sa.Float(), nullable=False),
        sa.Column("disbursed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("check_24h_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("check_48h_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.Enum("active", "flagged", "cleared", name="loanwatchstatus"),
            nullable=False,
            server_default="active",
        ),
        sa.Column("findings_json", sa.Text(), nullable=True),
        sa.Column("alert_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # --- Extend alert source enum with loan_monitoring ---
    op.execute("ALTER TYPE alertsource ADD VALUE IF NOT EXISTS 'loan_monitoring'")

    # --- Add investigation_report column to alerts ---
    op.add_column("alerts", sa.Column("investigation_report", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("alerts", "investigation_report")
    op.drop_table("loan_disbursement_watches")
    op.drop_table("agent_profiles")
    op.execute("DROP TYPE IF EXISTS loanwatchstatus")
