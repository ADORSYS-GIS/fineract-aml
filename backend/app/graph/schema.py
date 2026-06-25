"""Graph schema — node labels, edge types, and idempotent constraints/indexes (ADR 0007).

Memgraph raises if a constraint/index already exists, so `ensure_schema` runs each
statement independently and swallows "already exists" errors — safe to call on every boot.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Vocabulary (single source of truth for Cypher in client.py) ───────────────
# Entity nodes
LABEL_ACCOUNT = "Account"
LABEL_CLIENT = "Client"
LABEL_AGENT = "Agent"
LABEL_MERCHANT = "Merchant"
# Profile-asset nodes — the high-value PayPal-style signal
LABEL_DEVICE = "Device"
LABEL_IP = "IP"
LABEL_PHONE = "Phone"
LABEL_BANK = "Bank"

# Edge types
REL_TRANSACTED = "TRANSACTED"          # Account → Account, money flow
REL_OWNED_BY = "OWNED_BY"              # Account → Client
REL_AGENT_OF = "AGENT_OF"              # Agent → Account
REL_SERVES = "SERVES"                  # Merchant → Account
REL_SHARES_DEVICE = "SHARES_DEVICE"    # Account → Device
REL_SHARES_IP = "SHARES_IP"            # Account → IP
REL_SHARES_PHONE = "SHARES_PHONE"      # Account → Phone
REL_SHARES_BANK = "SHARES_BANK"        # Account → Bank

# Asset-sharing edges traversed for guilt-by-association / shared-asset fan
SHARED_ASSET_RELS = [REL_SHARES_DEVICE, REL_SHARES_IP, REL_SHARES_PHONE, REL_SHARES_BANK]
# Full set traversed for k-hop distance-to-bad (asset links + money flow)
GUILT_RELS = SHARED_ASSET_RELS + [REL_TRANSACTED]

# Node labels whose `id` must be unique (one row per real-world entity)
_UNIQUE_ID_LABELS = [
    LABEL_ACCOUNT, LABEL_CLIENT, LABEL_AGENT, LABEL_MERCHANT,
    LABEL_DEVICE, LABEL_IP, LABEL_PHONE, LABEL_BANK,
]

SCHEMA_STATEMENTS: list[str] = (
    [f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.id IS UNIQUE;" for label in _UNIQUE_ID_LABELS]
    + [f"CREATE INDEX ON :{label}(id);" for label in _UNIQUE_ID_LABELS]
    # Risk-flag indexes so guilt-by-association anchors resolve fast.
    + [
        f"CREATE INDEX ON :{LABEL_ACCOUNT}(known_bad);",
        f"CREATE INDEX ON :{LABEL_ACCOUNT}(frozen);",
        f"CREATE INDEX ON :{LABEL_ACCOUNT}(sanctioned);",
    ]
)


async def ensure_schema(session) -> None:
    """Apply constraints/indexes idempotently. Per-statement errors are logged, not raised."""
    if session is None:
        return
    for stmt in SCHEMA_STATEMENTS:
        try:
            await session.run(stmt)
        except Exception as exc:  # already-exists or transient — keep going
            logger.debug("Schema statement skipped (%s): %s", exc.__class__.__name__, stmt)
