"""Initial schema — all AML tables.

Revision ID: 001
Revises:
Create Date: 2025-06-15

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column(
            "role",
            sa.Enum("analyst", "senior_analyst", "compliance_officer", "admin", name="userrole"),
            nullable=False,
            server_default="analyst",
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Transactions
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("fineract_transaction_id", sa.String(100), unique=True, nullable=False),
        sa.Column("fineract_account_id", sa.String(100), nullable=False),
        sa.Column("fineract_client_id", sa.String(100), nullable=False),
        sa.Column(
            "transaction_type",
            sa.Enum("deposit", "withdrawal", "transfer", name="transactiontype"),
            nullable=False,
        ),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="USD"),
        sa.Column("transaction_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("counterparty_account_id", sa.String(100), nullable=True),
        sa.Column("counterparty_name", sa.String(255), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column(
            "risk_level",
            sa.Enum("low", "medium", "high", "critical", name="risklevel"),
            nullable=True,
        ),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("raw_payload", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_transactions_fineract_transaction_id", "transactions", ["fineract_transaction_id"])
    op.create_index("ix_transactions_fineract_account_id", "transactions", ["fineract_account_id"])
    op.create_index("ix_transactions_fineract_client_id", "transactions", ["fineract_client_id"])
    op.create_index("ix_transactions_date_account", "transactions", ["transaction_date", "fineract_account_id"])
    op.create_index("ix_transactions_risk", "transactions", ["risk_level", "risk_score"])

    # Alerts
    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "under_review", "confirmed_fraud", "false_positive", "escalated", "dismissed", name="alertstatus"),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "source",
            sa.Enum("rule_engine", "anomaly_detection", "ml_model", "manual", name="alertsource"),
            nullable=False,
        ),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("triggered_rules", sa.Text(), nullable=True),
        sa.Column("assigned_to", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_alerts_status", "alerts", ["status"])
    op.create_index("ix_alerts_assigned", "alerts", ["assigned_to", "status"])

    # Reviews
    op.create_table(
        "reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("alert_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("alerts.id"), nullable=False),
        sa.Column("reviewer_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "decision",
            sa.Enum("legitimate", "suspicious", "confirmed_fraud", name="reviewdecision"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("sar_filed", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("sar_reference", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Rule matches
    op.create_table(
        "rule_matches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column("rule_name", sa.String(100), nullable=False),
        sa.Column("rule_category", sa.String(50), nullable=False),
        sa.Column("severity", sa.Float(), nullable=False),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # Cases
    op.create_table(
        "cases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("case_number", sa.String(50), unique=True, nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("open", "investigating", "escalated", "closed_legitimate", "closed_fraud", "sar_filed", name="casestatus"),
            nullable=False,
            server_default="open",
        ),
        sa.Column("assigned_to", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("fineract_client_id", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_cases_case_number", "cases", ["case_number"])
    op.create_index("ix_cases_fineract_client_id", "cases", ["fineract_client_id"])

    # Case-transaction link
    op.create_table(
        "case_transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("case_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("transactions.id"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("case_transactions")
    op.drop_table("cases")
    op.drop_table("rule_matches")
    op.drop_table("reviews")
    op.drop_table("alerts")
    op.drop_table("transactions")
    op.drop_table("users")

    # Drop enums
    for name in [
        "casestatus", "reviewdecision", "alertsource", "alertstatus",
        "risklevel", "transactiontype", "userrole",
    ]:
        op.execute(f"DROP TYPE IF EXISTS {name}")
