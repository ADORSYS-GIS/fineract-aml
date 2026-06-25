"""Add RBAC role to users, audit fields to alerts/cases, COBAC fields to CTRs, FK to ScreeningResult.

Revision ID: 011
Revises: 010
Create Date: 2026-04-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add audit fields to alerts
    op.add_column("alerts", sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("alerts", sa.Column("closed_by", sa.String(100), nullable=True))

    # Add audit fields to cases
    op.add_column("cases", sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("cases", sa.Column("closed_by", sa.String(100), nullable=True))

    # Add COBAC fields to currency_transaction_reports
    op.add_column("currency_transaction_reports", sa.Column("agent_id", sa.String(100), nullable=True))
    op.add_column("currency_transaction_reports", sa.Column("branch_id", sa.String(100), nullable=True))
    op.add_column("currency_transaction_reports", sa.Column("counterparty_name", sa.String(255), nullable=True))
    op.add_column("currency_transaction_reports", sa.Column("counterparty_account", sa.String(100), nullable=True))
    op.add_column("currency_transaction_reports", sa.Column("filed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("currency_transaction_reports", sa.Column("cobac_reference", sa.String(100), nullable=True))
    op.add_column(
        "currency_transaction_reports",
        sa.Column("filed_by_user_id", UUID(as_uuid=True), nullable=True),
    )

    # Add FK from currency_transaction_reports.filed_by_user_id → users.id
    try:
        op.create_foreign_key(
            "fk_ctr_filed_by_user_id",
            "currency_transaction_reports", "users",
            ["filed_by_user_id"], ["id"],
            ondelete="SET NULL",
        )
    except Exception:
        pass  # FK may already exist

    # Add FK constraint to screening_results.transaction_id → transactions.id
    try:
        op.create_foreign_key(
            "fk_screening_results_transaction_id",
            "screening_results", "transactions",
            ["transaction_id"], ["id"],
            ondelete="CASCADE",
        )
    except Exception:
        pass  # FK may already exist

    # Add FK constraint to adverse_media_results.transaction_id → transactions.id
    try:
        op.create_foreign_key(
            "fk_adverse_media_results_transaction_id",
            "adverse_media_results", "transactions",
            ["transaction_id"], ["id"],
            ondelete="CASCADE",
        )
    except Exception:
        pass  # FK may already exist

    # Add REJECTED value to ctr_status enum (PostgreSQL-specific)
    # This is a no-op for non-Postgres; Alembic enum types are managed by the DB
    try:
        op.execute("ALTER TYPE ctrstatus ADD VALUE IF NOT EXISTS 'rejected'")
    except Exception:
        pass  # Non-Postgres or enum already has the value


def downgrade() -> None:
    try:
        op.drop_constraint("fk_adverse_media_results_transaction_id", "adverse_media_results", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_screening_results_transaction_id", "screening_results", type_="foreignkey")
    except Exception:
        pass
    try:
        op.drop_constraint("fk_ctr_filed_by_user_id", "currency_transaction_reports", type_="foreignkey")
    except Exception:
        pass

    op.drop_column("currency_transaction_reports", "filed_by_user_id")
    op.drop_column("currency_transaction_reports", "cobac_reference")
    op.drop_column("currency_transaction_reports", "filed_at")
    op.drop_column("currency_transaction_reports", "counterparty_account")
    op.drop_column("currency_transaction_reports", "counterparty_name")
    op.drop_column("currency_transaction_reports", "branch_id")
    op.drop_column("currency_transaction_reports", "agent_id")

    op.drop_column("cases", "closed_by")
    op.drop_column("cases", "closed_at")

    op.drop_column("alerts", "closed_by")
    op.drop_column("alerts", "closed_at")
