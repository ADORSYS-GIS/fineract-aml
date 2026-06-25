"""GraphClient — Cypher operations against Memgraph (ADR 0007).

All Cypher lives here so a Neo4j swap is mechanical. The client is constructed with an
open async session (`neo4j.AsyncSession`, or any object exposing `await run(query, **params)`
returning a result with `await single()` / `async for`), which makes it trivial to inject a
fake session in unit tests.

Write helpers (`upsert_transaction`, `set_account_flag`) run off the critical path in the
Celery pipeline. The read helper (`compute_graph_score`) is a single bounded query used on
the live `/api/v1/score` path and must stay cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.graph.schema import (
    GUILT_RELS,
    SHARED_ASSET_RELS,
)

logger = logging.getLogger(__name__)

# Asset field → (node label, edge type). Only fields present on a Transaction are wired
# automatically; phone/bank are optional and enriched by the backfill (from Customer).
_ASSET_EDGES: list[tuple[str, str, str]] = [
    ("device_id", "Device", "SHARES_DEVICE"),
    ("ip_address", "IP", "SHARES_IP"),
    ("phone", "Phone", "SHARES_PHONE"),
    ("bank", "Bank", "SHARES_BANK"),
]

_VALID_FLAGS = {"known_bad", "frozen", "sanctioned"}

# Variable-length-path rel patterns, precomputed from the schema vocabulary.
_GUILT_PATTERN = "|".join(GUILT_RELS)        # SHARES_DEVICE|SHARES_IP|...|TRANSACTED
_SHARED_PATTERN = "|".join(SHARED_ASSET_RELS)  # SHARES_DEVICE|SHARES_IP|SHARES_PHONE|SHARES_BANK

# Score blend + normalization constants.
_DIST_WEIGHT = 0.7   # guilt-by-association contributes most of the graph signal
_FAN_WEIGHT = 0.3    # shared-asset fan-out is a secondary smurfing signal
_DEFAULT_FAN_CAP = 10


@dataclass
class GraphScore:
    """Result of the bounded live graph query."""

    score: float                      # 0.0–1.0, ready to weight into the combined score
    distance_to_bad: int | None       # hops to nearest known_bad/frozen/sanctioned node
    shared_asset_fan: int             # # of other accounts sharing a device/IP/phone/bank

    @classmethod
    def empty(cls) -> GraphScore:
        return cls(score=0.0, distance_to_bad=None, shared_asset_fan=0)


def tx_to_graph_payload(tx, *, phone: str | None = None, bank: str | None = None) -> dict:
    """Map a Transaction (or test FakeTransaction) to a plain dict for upserts.

    `phone`/`bank` are not on the Transaction model — pass them when enriching from
    Customer during backfill; omitted on the live ingestion path.
    """
    tx_type = getattr(tx, "transaction_type", None)
    tx_type_value = getattr(tx_type, "value", tx_type)
    ts = getattr(tx, "transaction_date", None)
    return {
        "account_id": getattr(tx, "fineract_account_id", None),
        "client_id": getattr(tx, "fineract_client_id", None),
        "counterparty_id": getattr(tx, "counterparty_account_id", None),
        "transaction_type": tx_type_value,
        "amount": float(getattr(tx, "amount", 0) or 0),
        "ts": ts.isoformat() if hasattr(ts, "isoformat") else ts,
        "device_id": getattr(tx, "device_id", None),
        "ip_address": getattr(tx, "ip_address", None),
        "agent_id": getattr(tx, "agent_id", None),
        "merchant_id": getattr(tx, "merchant_id", None),
        "phone": phone,
        "bank": bank,
    }


def _score_from(distance_to_bad: int | None, fan: int, k_hops: int, fan_cap: int) -> float:
    """Blend guilt-by-association distance + shared-asset fan into a 0–1 score."""
    dist_comp = 0.0
    if distance_to_bad is not None and distance_to_bad >= 1:
        # 1 hop → 1.0; falls off linearly to ~0 at the k-hop horizon.
        dist_comp = max(0.0, 1.0 - (distance_to_bad - 1) / max(k_hops, 1))
    fan_comp = min(fan / fan_cap, 1.0) if fan_cap > 0 else 0.0
    return round(min(_DIST_WEIGHT * dist_comp + _FAN_WEIGHT * fan_comp, 1.0), 4)


class GraphClient:
    """Thin async wrapper over a graph session. One instance per session/request."""

    def __init__(self, session):
        self.session = session

    # ── Writes (off the critical path) ────────────────────────────────────────
    async def upsert_transaction(self, payload: dict) -> None:
        """Idempotently MERGE the nodes/edges implied by one transaction.

        Runs several small MERGEs (off-critical-path) rather than one FOREACH-heavy query,
        keeping the Cypher portable and each step independently debuggable.
        """
        account_id = payload.get("account_id")
        if not account_id:
            return

        await self.session.run(
            "MERGE (a:Account {id: $id}) SET a.last_seen = $ts",
            id=account_id, ts=payload.get("ts"),
        )

        if payload.get("client_id"):
            await self.session.run(
                "MERGE (a:Account {id: $a}) MERGE (c:Client {id: $c}) "
                "MERGE (a)-[:OWNED_BY]->(c)",
                a=account_id, c=payload["client_id"],
            )

        if payload.get("counterparty_id"):
            await self.session.run(
                "MERGE (a:Account {id: $a}) MERGE (cp:Account {id: $cp}) "
                "MERGE (a)-[t:TRANSACTED]->(cp) "
                "ON CREATE SET t.total_amount = $amt, t.count = 1, t.last_ts = $ts "
                "ON MATCH SET t.total_amount = coalesce(t.total_amount, 0) + $amt, "
                "t.count = coalesce(t.count, 0) + 1, t.last_ts = $ts",
                a=account_id, cp=payload["counterparty_id"],
                amt=payload.get("amount", 0.0), ts=payload.get("ts"),
            )

        for field, label, rel in _ASSET_EDGES:
            value = payload.get(field)
            if value:
                await self.session.run(
                    f"MERGE (a:Account {{id: $a}}) MERGE (x:{label} {{id: $v}}) "
                    f"MERGE (a)-[:{rel}]->(x)",
                    a=account_id, v=value,
                )

        if payload.get("agent_id"):
            await self.session.run(
                "MERGE (ag:Agent {id: $ag}) MERGE (a:Account {id: $a}) "
                "MERGE (ag)-[:AGENT_OF]->(a)",
                ag=payload["agent_id"], a=account_id,
            )

        if payload.get("merchant_id"):
            await self.session.run(
                "MERGE (m:Merchant {id: $m}) MERGE (a:Account {id: $a}) "
                "MERGE (m)-[:SERVES]->(a)",
                m=payload["merchant_id"], a=account_id,
            )

    async def set_account_metrics(
        self, account_id: str, *, pagerank: float, community_size: int, is_in_cycle: int
    ) -> None:
        """Write batch-computed centrality metrics onto an Account node (Phase 4 analyzer)."""
        await self.session.run(
            "MERGE (a:Account {id: $id}) "
            "SET a.pagerank = $pagerank, a.community_size = $community_size, "
            "a.is_in_cycle = $is_in_cycle",
            id=account_id, pagerank=float(pagerank),
            community_size=int(community_size), is_in_cycle=int(is_in_cycle),
        )

    async def set_account_flag(self, account_id: str, flag: str, value: bool = True) -> None:
        """Set a risk anchor flag (known_bad/frozen/sanctioned) on an account node."""
        # SECURITY: `flag` is interpolated into Cypher as a *property name* (Cypher cannot
        # parametrize property names). The _VALID_FLAGS allowlist is the injection guard —
        # only widen it with values that are safe bare Cypher identifiers.
        if flag not in _VALID_FLAGS:
            raise ValueError(f"invalid flag {flag!r}; expected one of {_VALID_FLAGS}")
        await self.session.run(
            f"MERGE (a:Account {{id: $id}}) SET a.{flag} = $v",
            id=account_id, v=value,
        )

    # ── Reads ─────────────────────────────────────────────────────────────────
    async def compute_graph_score(
        self, account_id: str, k_hops: int = 2, fan_cap: int = _DEFAULT_FAN_CAP
    ) -> GraphScore:
        """Single bounded query: k-hop distance-to-bad + shared-asset fan → 0–1 score.

        `k_hops` is clamped to [1, 3] and interpolated into the variable-length pattern
        (Cypher cannot parametrize path bounds); it is an int, so this is injection-safe.
        """
        k = max(1, min(int(k_hops), 3))
        query = (
            "MATCH (a:Account {id: $account_id}) "
            f"OPTIONAL MATCH p = (a)-[:{_GUILT_PATTERN} *1..{k}]-(bad) "
            "WHERE bad.known_bad = true OR bad.frozen = true OR bad.sanctioned = true "
            "WITH a, min(size(relationships(p))) AS distance_to_bad "
            f"OPTIONAL MATCH (a)-[:{_SHARED_PATTERN}]-(asset)-[:{_SHARED_PATTERN}]-(other:Account) "
            "WHERE other.id <> $account_id "
            "RETURN distance_to_bad AS distance_to_bad, "
            "count(DISTINCT other) AS shared_asset_fan"
        )
        result = await self.session.run(query, account_id=account_id)
        record = await result.single()
        if record is None:
            return GraphScore.empty()
        distance = record["distance_to_bad"]
        fan = int(record["shared_asset_fan"] or 0)
        return GraphScore(
            score=_score_from(distance, fan, k, fan_cap),
            distance_to_bad=int(distance) if distance is not None else None,
            shared_asset_fan=fan,
        )

    async def network_features(self, account_id: str, k_hops: int = 2) -> dict:
        """Per-account graph features for the ML vector (bare keys consumed by FeatureExtractor).

        Local/cheap signals (distance-to-bad, shared-asset fans, money-flow degree) are computed
        live; `pagerank`/`community_size`/`is_in_cycle` are read from node properties populated by
        the batch analyzer (Phase 4) and default to 0 until that runs.
        """
        empty = {
            "distance_to_known_bad": 0.0, "shared_device_count": 0, "shared_ip_count": 0,
            "degree": 0, "pagerank": 0.0, "is_in_cycle": 0, "community_size": 0,
        }
        if not account_id:
            return empty
        k = max(1, min(int(k_hops), 3))
        query = (
            "MATCH (a:Account {id: $account_id}) "
            f"OPTIONAL MATCH p = (a)-[:{_GUILT_PATTERN} *1..{k}]-(bad) "
            "WHERE bad.known_bad = true OR bad.frozen = true OR bad.sanctioned = true "
            "WITH a, min(size(relationships(p))) AS dist "
            "OPTIONAL MATCH (a)-[:SHARES_DEVICE]->(:Device)<-[:SHARES_DEVICE]-(o1:Account) "
            "WHERE o1.id <> $account_id "
            "WITH a, dist, count(DISTINCT o1) AS shared_device "
            "OPTIONAL MATCH (a)-[:SHARES_IP]->(:IP)<-[:SHARES_IP]-(o2:Account) "
            "WHERE o2.id <> $account_id "
            "WITH a, dist, shared_device, count(DISTINCT o2) AS shared_ip "
            "OPTIONAL MATCH (a)-[t:TRANSACTED]-() "
            "WITH a, dist, shared_device, shared_ip, count(t) AS degree "
            "RETURN dist AS dist, shared_device AS shared_device, shared_ip AS shared_ip, "
            "degree AS degree, coalesce(a.pagerank, 0.0) AS pagerank, "
            "coalesce(a.community_size, 0) AS community_size, "
            "coalesce(a.is_in_cycle, 0) AS is_in_cycle"
        )
        result = await self.session.run(query, account_id=account_id)
        record = await result.single()
        if record is None:
            return empty
        dist = record["dist"]
        return {
            # Closer to a bad node → larger signal; 0 when none within k hops.
            "distance_to_known_bad": (1.0 / dist) if dist else 0.0,
            "shared_device_count": int(record["shared_device"] or 0),
            "shared_ip_count": int(record["shared_ip"] or 0),
            "degree": int(record["degree"] or 0),
            "pagerank": float(record["pagerank"] or 0.0),
            "is_in_cycle": int(record["is_in_cycle"] or 0),
            "community_size": int(record["community_size"] or 0),
        }

    async def shared_asset_clusters(self, min_size: int = 3) -> list[dict]:
        """Assets shared by ≥ min_size accounts — the collector-smurfing / collusion signal.

        Used by the async ring-detection task (Phase 4) to raise ring alerts.
        """
        query = (
            f"MATCH (asset)<-[:{_SHARED_PATTERN}]-(acc:Account) "
            "WITH asset, collect(DISTINCT acc.id) AS accounts "
            "WHERE size(accounts) >= $min_size "
            "RETURN labels(asset)[0] AS asset_type, asset.id AS asset_id, "
            "accounts AS accounts, size(accounts) AS fan "
            "ORDER BY fan DESC"
        )
        result = await self.session.run(query, min_size=min_size)
        clusters: list[dict] = []
        async for record in result:
            clusters.append({
                "asset_type": record["asset_type"],
                "asset_id": record["asset_id"],
                "accounts": list(record["accounts"]),
                "fan": int(record["fan"]),
            })
        return clusters
