"""Add credit scoring tables and loan transaction types.

Revision ID: 003
Revises: 002
Create Date: 2026-03-19

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM, UUID

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Pre-create enum types so create_table doesn't clash
creditsegment = ENUM(
    "tier_a", "tier_b", "tier_c", "tier_d", "tier_e",
    name="creditsegment", create_type=False,
)
ml_credit_segment = ENUM(
    "tier_a", "tier_b", "tier_c", "tier_d", "tier_e",
    name="ml_credit_segment", create_type=False,
)
request_credit_segment = ENUM(
    "tier_a", "tier_b", "tier_c", "tier_d", "tier_e",
    name="request_credit_segment", create_type=False,
)
scoringmethod = ENUM(
    "rule_based", "ml_cluster", "hybrid",
    name="scoringmethod", create_type=False,
)
creditrequeststatus = ENUM(
    "pending_review", "approved", "rejected", "expired",
    name="creditrequeststatus", create_type=False,
)
creditrecommendation = ENUM(
    "approve", "review_carefully", "reject",
    name="creditrecommendation", create_type=False,
)


def upgrade() -> None:
    # Extend transactiontype enum with loan types
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'loan_disbursement'")
    op.execute("ALTER TYPE transactiontype ADD VALUE IF NOT EXISTS 'loan_repayment'")

    # Create all enum types via raw SQL (idempotent)
    bind = op.get_bind()
    creditsegment.create(bind, checkfirst=True)
    ml_credit_segment.create(bind, checkfirst=True)
    request_credit_segment.create(bind, checkfirst=True)
    scoringmethod.create(bind, checkfirst=True)
    creditrequeststatus.create(bind, checkfirst=True)
    creditrecommendation.create(bind, checkfirst=True)

    # Create customer_credit_profiles table
    op.create_table(
        "customer_credit_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("fineract_client_id", sa.String(100), unique=True, nullable=False),
        sa.Column("credit_score", sa.Float, nullable=False),
        sa.Column("segment", creditsegment, nullable=False),
        sa.Column("max_credit_amount", sa.Float, nullable=False),
        sa.Column("score_components", sa.Text, nullable=True),
        sa.Column("ml_cluster_id", sa.Integer, nullable=True),
        sa.Column("ml_segment_suggestion", ml_credit_segment, nullable=True),
        sa.Column("scoring_method", scoringmethod, default="rule_based", nullable=False),
        sa.Column("last_computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_credit_profiles_client", "customer_credit_profiles", ["fineract_client_id"])
    op.create_index("ix_credit_profiles_segment", "customer_credit_profiles", ["segment"])
    op.create_index("ix_credit_profiles_score", "customer_credit_profiles", ["credit_score"])

    # Create credit_requests table
    op.create_table(
        "credit_requests",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("fineract_client_id", sa.String(100), nullable=False),
        sa.Column("requested_amount", sa.Float, nullable=False),
        sa.Column("credit_score_at_request", sa.Float, nullable=False),
        sa.Column("segment_at_request", request_credit_segment, nullable=False),
        sa.Column("max_credit_at_request", sa.Float, nullable=False),
        sa.Column("recommendation", creditrecommendation, nullable=False),
        sa.Column("status", creditrequeststatus, default="pending_review", nullable=False),
        sa.Column("reviewer_notes", sa.Text, nullable=True),
        sa.Column("assigned_to", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reviewed_by", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_credit_requests_status", "credit_requests", ["status"])
    op.create_index("ix_credit_requests_client", "credit_requests", ["fineract_client_id"])


def downgrade() -> None:
    op.drop_index("ix_credit_requests_client", table_name="credit_requests")
    op.drop_index("ix_credit_requests_status", table_name="credit_requests")
    op.drop_table("credit_requests")

    op.drop_index("ix_credit_profiles_score", table_name="customer_credit_profiles")
    op.drop_index("ix_credit_profiles_segment", table_name="customer_credit_profiles")
    op.drop_index("ix_credit_profiles_client", table_name="customer_credit_profiles")
    op.drop_table("customer_credit_profiles")

    bind = op.get_bind()
    creditrecommendation.drop(bind, checkfirst=True)
    creditrequeststatus.drop(bind, checkfirst=True)
    scoringmethod.drop(bind, checkfirst=True)
    request_credit_segment.drop(bind, checkfirst=True)
    ml_credit_segment.drop(bind, checkfirst=True)
    creditsegment.drop(bind, checkfirst=True)
    # Note: cannot remove enum values from transactiontype in PostgreSQL
