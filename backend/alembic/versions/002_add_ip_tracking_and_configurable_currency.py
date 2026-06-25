"""Add IP tracking columns and make currency not default to USD.

Revision ID: 002
Revises: 001
Create Date: 2026-03-16

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("transactions", sa.Column("ip_address", sa.String(45), nullable=True))
    op.add_column("transactions", sa.Column("user_agent", sa.String(500), nullable=True))
    op.add_column("transactions", sa.Column("country_code", sa.String(2), nullable=True))
    op.add_column("transactions", sa.Column("geo_location", sa.String(100), nullable=True))

    op.create_index("ix_transactions_ip_address", "transactions", ["ip_address"])
    op.create_index("ix_transactions_country_code", "transactions", ["country_code"])


def downgrade() -> None:
    op.drop_index("ix_transactions_country_code", table_name="transactions")
    op.drop_index("ix_transactions_ip_address", table_name="transactions")

    op.drop_column("transactions", "geo_location")
    op.drop_column("transactions", "country_code")
    op.drop_column("transactions", "user_agent")
    op.drop_column("transactions", "ip_address")
