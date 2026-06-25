"""Add sanctions screening, CTR, audit log tables, and explainability fields.

Revision ID: 004
Revises: 003
Create Date: 2026-03-21

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM, UUID

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# New enum types
watchlist_source_enum = ENUM(
    "ofac_sdn", "eu_sanctions", "un_sanctions", "pep", "custom",
    name="watchlistsource", create_type=False,
)
screening_status_enum = ENUM(
    "clear", "potential_match", "confirmed_match", "false_positive",
    name="screeningstatus", create_type=False,
)
ctr_status_enum = ENUM(
    "pending", "filed", "acknowledged",
    name="ctrstatus", create_type=False,
)

# Additional transaction types
new_tx_types = [
    "share_purchase", "share_redemption", "fixed_deposit",
    "recurring_deposit", "charge", "fee", "other",
]


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE watchlistsource AS ENUM ('ofac_sdn', 'eu_sanctions', 'un_sanctions', 'pep', 'custom')")
    op.execute("CREATE TYPE screeningstatus AS ENUM ('clear', 'potential_match', 'confirmed_match', 'false_positive')")
    op.execute("CREATE TYPE ctrstatus AS ENUM ('pending', 'filed', 'acknowledged')")

    # Extend TransactionType enum with new values
    for val in new_tx_types:
        op.execute(f"ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS '{val}'")

    # Watchlist entries table
    op.create_table(
        "watchlist_entries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source", watchlist_source_enum, nullable=False),
        sa.Column("entity_name", sa.String(500), nullable=False, index=True),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("country", sa.String(2)),
        sa.Column("aliases", sa.Text),
        sa.Column("identifiers", sa.Text),
        sa.Column("program", sa.String(255)),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_watchlist_source_name", "watchlist_entries", ["source", "entity_name"])

    # Screening results table
    op.create_table(
        "screening_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("transaction_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("screened_name", sa.String(500), nullable=False),
        sa.Column("matched_entry_id", UUID(as_uuid=True)),
        sa.Column("matched_name", sa.String(500)),
        sa.Column("match_score", sa.Float),
        sa.Column("source", watchlist_source_enum),
        sa.Column("status", screening_status_enum, nullable=False, server_default="clear"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_screening_status", "screening_results", ["status"])

    # Currency Transaction Reports table
    op.create_table(
        "currency_transaction_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("transaction_id", UUID(as_uuid=True), nullable=False, index=True),
        sa.Column("fineract_client_id", sa.String(100), nullable=False, index=True),
        sa.Column("fineract_account_id", sa.String(100), nullable=False),
        sa.Column("amount", sa.Float, nullable=False),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("transaction_type", sa.String(50), nullable=False),
        sa.Column("status", ctr_status_enum, nullable=False, server_default="pending"),
        sa.Column("reference_number", sa.String(100)),
        sa.Column("filed_by", sa.String(100)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_ctr_status", "currency_transaction_reports", ["status"])

    # Audit log table
    op.create_table(
        "audit_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
        sa.Column("user_id", sa.String(100), index=True),
        sa.Column("username", sa.String(100)),
        sa.Column("action", sa.String(100), nullable=False, index=True),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(100)),
        sa.Column("details", sa.Text),
        sa.Column("ip_address", sa.String(45)),
    )

    # Add explainability and shadow score columns to transactions
    op.add_column("transactions", sa.Column("score_explanation", sa.Text))
    op.add_column("transactions", sa.Column("shadow_score", sa.Float))


def downgrade() -> None:
    op.drop_column("transactions", "shadow_score")
    op.drop_column("transactions", "score_explanation")
    op.drop_table("audit_logs")
    op.drop_table("currency_transaction_reports")
    op.drop_table("screening_results")
    op.drop_table("watchlist_entries")
    op.execute("DROP TYPE IF EXISTS ctrstatus")
    op.execute("DROP TYPE IF EXISTS screeningstatus")
    op.execute("DROP TYPE IF EXISTS watchlistsource")
