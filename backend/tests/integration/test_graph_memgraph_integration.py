"""Integration tests for the graph fraud layer against a LIVE Memgraph (ADR 0007).

Unlike ``tests/test_graph_fraud_layer.py`` (fake-session unit tests), these exercise the
real Cypher in ``app/graph/client.py`` / ``app/graph/schema.py`` against a running Memgraph
over Bolt. They are **skipped cleanly** when the ``neo4j`` driver is missing or no Memgraph
is reachable, so the normal CI unit suite stays green without the graph DB.

How to run
----------
    docker compose up -d memgraph        # from the repo that defines the memgraph service
    AML_GRAPH_ENABLED=true pytest tests/integration -m integration

The tests connect to ``settings.graph_database_url`` (default ``bolt://localhost:7687``).
They build and tear down their own driver so they do not depend on the cached module-global
driver in ``app.core.graph``.
"""

from __future__ import annotations

import pytest

from app.core.config import settings

pytestmark = pytest.mark.integration

# Force the feature flag on for this module so any code path that consults
# settings.graph_enabled (and the test's own assumptions) behaves as "graph on".
settings.graph_enabled = True

# ── Module-level skip: no driver or no reachable Memgraph → skip the whole file ──
try:
    from neo4j import AsyncGraphDatabase
except ImportError:  # pragma: no cover - driver absent in unit-only CI
    pytest.skip(
        "neo4j driver not installed — graph integration tests skipped",
        allow_module_level=True,
    )


async def _can_connect() -> bool:
    """Open a throwaway driver and run a trivial round-trip. Never raises."""
    driver = AsyncGraphDatabase.driver(settings.graph_database_url, auth=None)
    try:
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            return bool(record and record["ok"] == 1)
    except Exception:
        return False
    finally:
        await driver.close()


def _check_memgraph_reachable() -> None:
    """Run the async healthcheck on a fresh event loop; skip the module if it fails."""
    import asyncio

    try:
        reachable = asyncio.run(_can_connect())
    except Exception:
        reachable = False
    if not reachable:
        pytest.skip(
            f"no live Memgraph at {settings.graph_database_url} — "
            "run `docker compose up -d memgraph`",
            allow_module_level=True,
        )


_check_memgraph_reachable()


# Imports that are only meaningful once we know the graph layer is testable.
from app.graph.client import GraphClient  # noqa: E402
from app.graph.schema import ensure_schema  # noqa: E402
from tests.conftest import FakeTransaction  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
async def driver():
    """A dedicated async Bolt driver, fully owned and torn down by the test session."""
    drv = AsyncGraphDatabase.driver(settings.graph_database_url, auth=None)
    try:
        yield drv
    finally:
        await drv.close()


@pytest.fixture
async def session(driver):
    """A clean-slate async session: wipes the graph and (re)applies the schema per test."""
    async with driver.session() as sess:
        await sess.run("MATCH (n) DETACH DELETE n")
        await ensure_schema(sess)
        yield sess


@pytest.fixture
def client(session):
    """GraphClient bound to the clean-slate session."""
    return GraphClient(session)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _payload(account_id, **overrides):
    """Build an upsert payload from a FakeTransaction, applying field overrides."""
    from app.graph.client import tx_to_graph_payload

    tx = FakeTransaction(fineract_account_id=account_id, **overrides)
    return tx_to_graph_payload(tx)


async def _read_single(session, query, **params):
    result = await session.run(query, **params)
    return await result.single()


# ── Tests ─────────────────────────────────────────────────────────────────────
async def test_ensure_schema_is_idempotent(session):
    # Arrange / Act — schema already applied once by the fixture; apply again.
    await ensure_schema(session)
    await ensure_schema(session)

    # Assert — a basic write/read still works, proving the session is healthy.
    await session.run("MERGE (a:Account {id: 'SCHEMA-OK'})")
    record = await _read_single(
        session, "MATCH (a:Account {id: 'SCHEMA-OK'}) RETURN a.id AS id"
    )
    assert record["id"] == "SCHEMA-OK"


async def test_upsert_creates_account_device_and_shares_device_edge(session, client):
    # Arrange
    payload = _payload("ACC-A", device_id="dev-shared", fineract_client_id=None)

    # Act
    await client.upsert_transaction(payload)

    # Assert — Account + Device nodes and the directed SHARES_DEVICE edge exist.
    record = await _read_single(
        session,
        "MATCH (a:Account {id: 'ACC-A'})-[:SHARES_DEVICE]->(d:Device {id: 'dev-shared'}) "
        "RETURN a.id AS account, d.id AS device",
    )
    assert record is not None
    assert record["account"] == "ACC-A"
    assert record["device"] == "dev-shared"


async def test_two_accounts_sharing_device_have_shared_asset_fan(session, client):
    # Arrange — two accounts upserted with the same device fingerprint.
    await client.upsert_transaction(_payload("ACC-A", device_id="dev-1", fineract_client_id=None))
    await client.upsert_transaction(_payload("ACC-B", device_id="dev-1", fineract_client_id=None))

    # Act
    score_a = await client.compute_graph_score("ACC-A")
    score_b = await client.compute_graph_score("ACC-B")

    # Assert — each sees exactly one other account sharing its device.
    assert score_a.shared_asset_fan >= 1
    assert score_b.shared_asset_fan >= 1


async def test_guilt_by_association_distance_and_score(session, client):
    # Arrange — A —SHARES_DEVICE→ D ←SHARES_DEVICE— B (known_bad).
    await client.upsert_transaction(_payload("ACC-A", device_id="dev-bad", fineract_client_id=None))
    await client.upsert_transaction(_payload("ACC-B", device_id="dev-bad", fineract_client_id=None))
    await client.set_account_flag("ACC-B", "known_bad", True)
    # An unrelated account with no shared assets and no bad neighbour.
    await client.upsert_transaction(
        _payload("ACC-LONE", device_id="dev-lone", fineract_client_id=None)
    )

    # Act
    score_a = await client.compute_graph_score("ACC-A")
    score_lone = await client.compute_graph_score("ACC-LONE")

    # Assert — A is 2 hops from the bad node (A→D, D←B) and scores positive.
    assert score_a.distance_to_bad == 2
    assert score_a.score > 0
    # The unrelated account has no path to a bad node.
    assert score_lone.distance_to_bad is None
    assert score_lone.score == 0.0


async def test_transacted_edge_accumulates_count_and_total(session, client):
    # Arrange — two transfers along the same (account → counterparty) edge.
    await client.upsert_transaction(
        _payload(
            "ACC-SRC", counterparty_account_id="ACC-DST", amount=100.0, fineract_client_id=None
        )
    )
    await client.upsert_transaction(
        _payload(
            "ACC-SRC", counterparty_account_id="ACC-DST", amount=250.0, fineract_client_id=None
        )
    )

    # Act — read the TRANSACTED edge properties directly.
    record = await _read_single(
        session,
        "MATCH (:Account {id: 'ACC-SRC'})-[t:TRANSACTED]->(:Account {id: 'ACC-DST'}) "
        "RETURN t.count AS count, t.total_amount AS total",
    )

    # Assert — count incremented, amounts summed.
    assert record is not None
    assert record["count"] == 2
    assert record["total"] == pytest.approx(350.0)


async def test_shared_asset_clusters_respects_min_size(session, client):
    # Arrange — three accounts share one device.
    for acc in ("ACC-1", "ACC-2", "ACC-3"):
        await client.upsert_transaction(
            _payload(acc, device_id="dev-cluster", fineract_client_id=None)
        )

    # Act
    clusters_3 = await client.shared_asset_clusters(min_size=3)
    clusters_4 = await client.shared_asset_clusters(min_size=4)

    # Assert — a single 3-account cluster on the device; none when raising the floor to 4.
    device_clusters = [c for c in clusters_3 if c["asset_id"] == "dev-cluster"]
    assert len(device_clusters) == 1
    assert device_clusters[0]["asset_type"] == "Device"
    assert device_clusters[0]["fan"] == 3
    assert sorted(device_clusters[0]["accounts"]) == ["ACC-1", "ACC-2", "ACC-3"]
    assert [c for c in clusters_4 if c["asset_id"] == "dev-cluster"] == []


async def test_network_features_shape_and_values(session, client):
    # Arrange — ACC-A shares a device with one other account and sends one transfer.
    await client.upsert_transaction(
        _payload(
            "ACC-A",
            device_id="dev-nf",
            counterparty_account_id="ACC-CP",
            fineract_client_id=None,
        )
    )
    await client.upsert_transaction(_payload("ACC-B", device_id="dev-nf", fineract_client_id=None))

    # Act
    feats = await client.network_features("ACC-A")

    # Assert — exact set of bare keys consumed by the FeatureExtractor.
    assert set(feats.keys()) == {
        "distance_to_known_bad",
        "shared_device_count",
        "shared_ip_count",
        "degree",
        "pagerank",
        "is_in_cycle",
        "community_size",
    }
    # One other account on the same device; one TRANSACTED edge → degree 1.
    assert feats["shared_device_count"] == 1
    assert feats["degree"] == 1
    # No bad neighbour wired here, so the distance signal is 0.
    assert feats["distance_to_known_bad"] == 0.0


async def test_smurfing_scenario_score_rises_after_flagging_collector(session, client):
    # Arrange — three collector accounts share one device and all fund one beneficiary.
    collectors = ("COL-1", "COL-2", "COL-3")
    for col in collectors:
        await client.upsert_transaction(
            _payload(
                col,
                device_id="dev-ring",
                counterparty_account_id="BENEFICIARY",
                amount=300.0,
                fineract_client_id=None,
            )
        )

    # Baseline — beneficiary scores before any collector is flagged.
    before = await client.compute_graph_score("BENEFICIARY")

    # Act — flag one collector as known_bad; the beneficiary is now 1 hop (TRANSACTED) away.
    await client.set_account_flag("COL-1", "known_bad", True)
    after = await client.compute_graph_score("BENEFICIARY")

    # Assert — guilt-by-association lifts the beneficiary's graph score.
    assert before.distance_to_bad is None
    assert after.distance_to_bad == 1
    assert after.score > before.score
