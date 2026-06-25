"""Graph service — builds transaction network graphs for visualization.

Uses NetworkX to compute 2-hop neighborhoods around an account.
Results are cached in Redis (TTL 15 minutes) to reduce DB load.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

import networkx as nx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.transaction import Transaction
from app.schemas.graph import GraphEdge, GraphNode, GraphResponse

logger = logging.getLogger(__name__)

_GRAPH_CACHE_TTL = 900  # 15 minutes


class GraphService:
    """Builds and caches account transaction networks."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_account_graph(
        self, account_id: str, days: int = 30, depth: int = 2
    ) -> GraphResponse:
        """Return the 2-hop transaction network around an account.

        Args:
            account_id: Fineract account ID (center node).
            days: How many days of transaction history to include.
            depth: Network hop depth (1 = direct, 2 = indirect).

        Returns:
            GraphResponse with nodes and edges for D3/Cytoscape.
        """
        cache_key = f"graph:account:{account_id}:{days}d:{depth}hop"
        cached = self._get_cached(cache_key)
        if cached:
            return GraphResponse(**cached)

        graph = await self._build_graph(account_id, days, depth)
        response = self._serialize_graph(graph, account_id, depth)
        self._set_cached(cache_key, response.model_dump())
        return response

    async def get_case_graph(self, case_id: str) -> GraphResponse:
        """Return the transaction network for all accounts in a case."""
        from app.models.case import CaseTransaction

        result = await self.db.execute(
            select(CaseTransaction.transaction_id).where(
                CaseTransaction.case_id == case_id  # type: ignore[arg-type]
            )
        )
        tx_ids = [row[0] for row in result.all()]
        if not tx_ids:
            return GraphResponse(account_id=str(case_id), nodes=[], edges=[])

        # Get all transactions
        tx_result = await self.db.execute(
            select(Transaction).where(Transaction.id.in_(tx_ids))
        )
        transactions = list(tx_result.scalars().all())

        graph = nx.DiGraph()
        for tx in transactions:
            src = tx.fineract_account_id or tx.fineract_client_id
            dst = tx.counterparty_account_id or tx.counterparty_name or "external"
            if src and dst and src != dst:
                if graph.has_edge(src, dst):
                    graph[src][dst]["weight"] += tx.amount
                    graph[src][dst]["tx_count"] += 1
                else:
                    graph.add_edge(
                        src, dst,
                        weight=tx.amount,
                        tx_count=1,
                        edge_type=tx.transaction_type.value,
                    )

        response = self._serialize_graph(graph, str(case_id), depth=1)
        return response

    async def _build_graph(
        self, account_id: str, days: int, depth: int
    ) -> nx.DiGraph:
        """Build a directed graph starting from account_id up to `depth` hops."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        graph = nx.DiGraph()
        frontier = {account_id}
        visited = set()

        for _ in range(depth):
            if not frontier:
                break
            next_frontier = set()

            # Outgoing transactions from frontier accounts
            out_result = await self.db.execute(
                select(Transaction).where(
                    Transaction.fineract_account_id.in_(frontier),
                    Transaction.transaction_date >= cutoff,
                    Transaction.counterparty_account_id != None,  # noqa: E711
                )
            )
            for tx in out_result.scalars():
                src = tx.fineract_account_id
                dst = tx.counterparty_account_id
                self._add_or_update_edge(graph, src, dst, tx)
                if dst not in visited:
                    next_frontier.add(dst)

            # Incoming transactions to frontier accounts
            in_result = await self.db.execute(
                select(Transaction).where(
                    Transaction.counterparty_account_id.in_(frontier),
                    Transaction.transaction_date >= cutoff,
                )
            )
            for tx in in_result.scalars():
                src = tx.fineract_account_id or tx.counterparty_account_id
                dst = tx.fineract_account_id
                if src and src != dst:
                    self._add_or_update_edge(graph, src, dst, tx)

            visited.update(frontier)
            frontier = next_frontier - visited

        return graph

    def _add_or_update_edge(self, graph: nx.DiGraph, src: str, dst: str, tx) -> None:
        if not src or not dst or src == dst:
            return
        if graph.has_edge(src, dst):
            graph[src][dst]["weight"] += tx.amount
            graph[src][dst]["tx_count"] += 1
        else:
            graph.add_edge(
                src, dst,
                weight=tx.amount,
                tx_count=1,
                edge_type=tx.transaction_type.value if tx.transaction_type else "transfer",
            )
        # Track node metadata
        for node_id in (src, dst):
            if node_id not in graph.nodes:
                graph.add_node(node_id)
            node = graph.nodes[node_id]
            node["total_volume"] = node.get("total_volume", 0) + tx.amount
            node["tx_count"] = node.get("tx_count", 0) + 1

    def _serialize_graph(
        self, graph: nx.DiGraph, center_id: str, depth: int
    ) -> GraphResponse:
        nodes = [
            GraphNode(
                id=node_id,
                label=node_id[:12],
                node_type="account",
                transaction_count=attrs.get("tx_count", 0),
                total_volume=attrs.get("total_volume", 0.0),
                is_flagged=False,
            )
            for node_id, attrs in graph.nodes(data=True)
        ]
        edges = [
            GraphEdge(
                source=src,
                target=dst,
                weight=attrs.get("weight", 0.0),
                tx_count=attrs.get("tx_count", 0),
                edge_type=attrs.get("edge_type", "transfer"),
            )
            for src, dst, attrs in graph.edges(data=True)
        ]
        return GraphResponse(
            account_id=center_id,
            nodes=nodes,
            edges=edges,
            depth=depth,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    def _get_cached(self, key: str) -> dict | None:
        try:
            import redis

            r = redis.from_url(settings.redis_url, socket_connect_timeout=1)
            raw = r.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _set_cached(self, key: str, data: dict) -> None:
        try:
            import redis

            r = redis.from_url(settings.redis_url, socket_connect_timeout=1)
            r.setex(key, _GRAPH_CACHE_TTL, json.dumps(data))
        except Exception:
            pass
