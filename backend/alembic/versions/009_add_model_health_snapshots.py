"""Add model health snapshots table.

Revision ID: 009
Revises: 008
Create Date: 2026-04-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

# revision identifiers, used by Alembic.
revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_health_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_sample_count", sa.Integer(), nullable=True),
        sa.Column("auc_score", sa.Float(), nullable=True),
        sa.Column("precision_score", sa.Float(), nullable=True),
        sa.Column("recall_score", sa.Float(), nullable=True),
        sa.Column("psi_score", sa.Float(), nullable=True),
        sa.Column("drift_status", sa.String(20), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        # TimestampMixin
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_model_health_model_name", "model_health_snapshots", ["model_name"])


def downgrade() -> None:
    op.drop_index("ix_model_health_model_name", "model_health_snapshots")
    op.drop_table("model_health_snapshots")
