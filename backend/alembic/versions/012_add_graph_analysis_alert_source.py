"""Add 'graph_analysis' value to the alertsource enum (ADR 0007 — graph fraud layer).

Ring / guilt-by-association alerts raised by the graph layer use AlertSource.GRAPH_ANALYSIS.

Revision ID: 012
Revises: 011
Create Date: 2026-06-25
"""
from collections.abc import Sequence

from alembic import op

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Enum ALTER is Postgres-specific. Gate on the dialect rather than swallowing every
    # exception, so a genuine failure on Postgres surfaces instead of silently no-opping
    # (which would later break `Alert(source=GRAPH_ANALYSIS)` inserts at runtime).
    if op.get_bind().dialect.name == "postgresql":
        op.execute("ALTER TYPE alertsource ADD VALUE IF NOT EXISTS 'graph_analysis'")


def downgrade() -> None:
    # Postgres cannot drop a single enum value without recreating the type; intentionally a no-op.
    pass
