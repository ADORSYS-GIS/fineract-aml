"""Add credit request gaming fields, case SAR/escalation fields.

Revision ID: 008
Revises: 007
Create Date: 2026-04-02
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── credit_requests: score_inflation_flag + explanation_text ──────────────
    op.add_column(
        "credit_requests",
        sa.Column(
            "score_inflation_flag",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "credit_requests",
        sa.Column("explanation_text", sa.Text(), nullable=True),
    )

    # ── cases: escalation_reason + sla_deadline + sar_document_path ───────────
    op.add_column(
        "cases",
        sa.Column("escalation_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "cases",
        sa.Column(
            "sla_deadline",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "cases",
        sa.Column("sar_document_path", sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("cases", "sar_document_path")
    op.drop_column("cases", "sla_deadline")
    op.drop_column("cases", "escalation_reason")
    op.drop_column("credit_requests", "explanation_text")
    op.drop_column("credit_requests", "score_inflation_flag")
