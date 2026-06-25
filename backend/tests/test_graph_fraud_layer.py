"""Unit tests for the graph fraud layer (ADR 0007).

Pure-logic + fake-session tests — no live Memgraph required. A `FakeSession` returns canned
records so the Cypher-consuming code paths and the fail-open behaviour are exercised in
milliseconds, matching the project's no-testcontainers convention.
"""

import pytest

from app.graph.client import (
    GraphClient,
    GraphScore,
    _score_from,
    tx_to_graph_payload,
)
from tests.conftest import FakeTransaction


# ── Fake async neo4j session ──────────────────────────────────────────────────
class FakeResult:
    def __init__(self, records: list[dict] | None):
        self._records = records or []

    async def single(self):
        return self._records[0] if self._records else None

    def __aiter__(self):
        self._it = iter(self._records)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class FakeSession:
    """Records every query and returns a preconfigured result."""

    def __init__(self, single_record=None, iter_records=None):
        self._single = single_record
        self._iter = iter_records
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        if self._iter is not None:
            return FakeResult(self._iter)
        return FakeResult([self._single] if self._single is not None else [])


# ── tx_to_graph_payload ───────────────────────────────────────────────────────
def test_tx_to_graph_payload_maps_identifier_fields():
    tx = FakeTransaction(
        fineract_account_id="ACC-1",
        fineract_client_id="CLI-1",
        counterparty_account_id="ACC-2",
        device_id="dev-abc",
        ip_address="10.0.0.1",
        amount=1500.0,
    )
    payload = tx_to_graph_payload(tx, phone="+237600000000", bank="MoMo-123")

    assert payload["account_id"] == "ACC-1"
    assert payload["client_id"] == "CLI-1"
    assert payload["counterparty_id"] == "ACC-2"
    assert payload["device_id"] == "dev-abc"
    assert payload["ip_address"] == "10.0.0.1"
    assert payload["phone"] == "+237600000000"
    assert payload["bank"] == "MoMo-123"
    assert payload["amount"] == 1500.0
    # transaction_date is serialized to an ISO string for the graph driver.
    assert isinstance(payload["ts"], str)


# ── _score_from blend ─────────────────────────────────────────────────────────
def test_score_from_no_signal_is_zero():
    assert _score_from(distance_to_bad=None, fan=0, k_hops=2, fan_cap=10) == 0.0


def test_score_from_directly_linked_to_bad_is_strong():
    # distance 1 → full distance component (0.7); fan 0 → no fan component.
    assert _score_from(distance_to_bad=1, fan=0, k_hops=2, fan_cap=10) == 0.7


def test_score_from_blends_distance_and_fan():
    # distance 1 → 0.7; fan 5/10 → 0.3*0.5 = 0.15 → 0.85
    assert _score_from(distance_to_bad=1, fan=5, k_hops=2, fan_cap=10) == 0.85


def test_score_from_decays_with_distance():
    near = _score_from(distance_to_bad=1, fan=0, k_hops=3, fan_cap=10)
    far = _score_from(distance_to_bad=3, fan=0, k_hops=3, fan_cap=10)
    assert near > far


# ── GraphClient.compute_graph_score ───────────────────────────────────────────
async def test_compute_graph_score_reads_record():
    session = FakeSession(single_record={"distance_to_bad": 1, "shared_asset_fan": 3})
    score = await GraphClient(session).compute_graph_score("ACC-1", k_hops=2, fan_cap=10)

    assert isinstance(score, GraphScore)
    assert score.distance_to_bad == 1
    assert score.shared_asset_fan == 3
    assert score.score == _score_from(1, 3, 2, 10)


async def test_compute_graph_score_empty_graph_returns_empty():
    session = FakeSession(single_record={"distance_to_bad": None, "shared_asset_fan": 0})
    score = await GraphClient(session).compute_graph_score("ACC-unknown")
    assert score.score == 0.0
    assert score.distance_to_bad is None


async def test_compute_graph_score_clamps_k_hops_into_query():
    session = FakeSession(single_record={"distance_to_bad": None, "shared_asset_fan": 0})
    await GraphClient(session).compute_graph_score("ACC-1", k_hops=99)
    query = session.calls[0][0]
    # k clamped to 3 — the variable-length bound must reflect that, never 99.
    assert "*1..3" in query
    assert "*1..99" not in query


# ── GraphClient.shared_asset_clusters ─────────────────────────────────────────
async def test_shared_asset_clusters_parses_rows():
    rows = [
        {"asset_type": "Device", "asset_id": "dev-x", "accounts": ["A", "B", "C"], "fan": 3},
    ]
    clusters = await GraphClient(FakeSession(iter_records=rows)).shared_asset_clusters(min_size=3)
    assert len(clusters) == 1
    assert clusters[0]["accounts"] == ["A", "B", "C"]
    assert clusters[0]["fan"] == 3


# ── GraphClient.network_features ──────────────────────────────────────────────
async def test_network_features_transforms_distance_to_signal():
    record = {
        "dist": 2, "shared_device": 4, "shared_ip": 1, "degree": 7,
        "pagerank": 0.12, "community_size": 9, "is_in_cycle": 1,
    }
    feats = await GraphClient(FakeSession(single_record=record)).network_features("ACC-1")
    assert feats["distance_to_known_bad"] == pytest.approx(0.5)  # 1/2
    assert feats["shared_device_count"] == 4
    assert feats["degree"] == 7
    assert feats["community_size"] == 9


async def test_set_account_flag_rejects_unknown_flag():
    with pytest.raises(ValueError):
        await GraphClient(FakeSession()).set_account_flag("ACC-1", "definitely_not_a_flag")
