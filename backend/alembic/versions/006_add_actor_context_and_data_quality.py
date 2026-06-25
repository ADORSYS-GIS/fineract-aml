"""Add WeBank actor context fields and data quality column to transactions.

Revision ID: 006
Revises: 005
Create Date: 2026-04-02

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # WeBank actor context columns
    op.add_column("transactions", sa.Column("actor_type", sa.String(20), nullable=True))
    op.add_column("transactions", sa.Column("agent_id", sa.String(100), nullable=True))
    op.add_column("transactions", sa.Column("branch_id", sa.String(100), nullable=True))
    op.add_column("transactions", sa.Column("merchant_id", sa.String(100), nullable=True))
    op.add_column("transactions", sa.Column("device_id", sa.String(64), nullable=True))
    op.add_column("transactions", sa.Column("kyc_level", sa.Integer(), nullable=True))

    # Data quality
    op.add_column("transactions", sa.Column("data_quality_warnings", sa.Text(), nullable=True))

    # Indexes for agent/merchant lookups used by new rules
    op.create_index("ix_transactions_agent_id", "transactions", ["agent_id"])
    op.create_index("ix_transactions_merchant_id", "transactions", ["merchant_id"])
    op.create_index("ix_transactions_actor_type", "transactions", ["actor_type"])


def downgrade() -> None:
    op.drop_index("ix_transactions_actor_type", table_name="transactions")
    op.drop_index("ix_transactions_merchant_id", table_name="transactions")
    op.drop_index("ix_transactions_agent_id", table_name="transactions")
    op.drop_column("transactions", "data_quality_warnings")
    op.drop_column("transactions", "kyc_level")
    op.drop_column("transactions", "device_id")
    op.drop_column("transactions", "merchant_id")
    op.drop_column("transactions", "branch_id")
    op.drop_column("transactions", "agent_id")
    op.drop_column("transactions", "actor_type")
