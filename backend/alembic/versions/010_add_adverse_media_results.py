"""Add adverse_media_results table.

Revision ID: 010
Revises: 009
Create Date: 2026-04-02
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "adverse_media_results",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("transaction_id", UUID(as_uuid=True), nullable=False),
        sa.Column("entity_name", sa.String(500), nullable=False),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("highest_relevance_score", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("article_snippets", sa.Text, nullable=True),
        sa.Column("screened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_adverse_media_transaction_id", "adverse_media_results", ["transaction_id"])
    op.create_index("ix_adverse_media_entity", "adverse_media_results", ["entity_name"])


def downgrade() -> None:
    op.drop_index("ix_adverse_media_entity", table_name="adverse_media_results")
    op.drop_index("ix_adverse_media_transaction_id", table_name="adverse_media_results")
    op.drop_table("adverse_media_results")
