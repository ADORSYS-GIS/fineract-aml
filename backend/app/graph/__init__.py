"""Graph fraud layer — persistent profile-asset graph in Memgraph (ADR 0007).

Public surface:
- `ensure_schema` / `SCHEMA_STATEMENTS` (schema.py): idempotent constraints + indexes.
- `GraphClient` (client.py): upsert edges, k-hop guilt-by-association, shared-asset fan,
  and the bounded `compute_graph_score` used on the live scoring path.
- `tx_to_graph_payload` (client.py): map a Transaction/FakeTransaction to a plain dict.
"""

from app.graph.client import GraphClient, GraphScore, tx_to_graph_payload
from app.graph.schema import ensure_schema

__all__ = [
    "GraphClient",
    "GraphScore",
    "tx_to_graph_payload",
    "ensure_schema",
]
