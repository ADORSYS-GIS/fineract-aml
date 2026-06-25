"""Fail-open + feature-gating tests for the graph fraud layer (ADR 0007).

The graph layer is a scoring *enrichment*, not a compliance gate — it must degrade silently
and never block the path. These tests assert the live component returns (0.0, degraded=True)
on any failure, and that the ML feature vector only widens when the flag is on.
"""

import numpy as np

from app.core.config import settings
from app.features.extractor import FEATURE_NAMES, GRAPH_FEATURE_NAMES, FeatureExtractor
from app.services.scoring_service import ScoringService
from tests.conftest import FakeTransaction


# ── Live graph component fail-open ────────────────────────────────────────────
async def test_graph_component_skipped_for_empty_account():
    score, degraded = await ScoringService(db=None)._graph_component("")
    assert score == 0.0
    assert degraded is True


async def test_graph_component_failopen_on_error(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", True)

    # Make the graph session raise — the component must swallow it and degrade.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("memgraph down")

    monkeypatch.setattr("app.core.graph.graph_session", _boom)

    score, degraded = await ScoringService(db=None)._graph_component("ACC-1")
    assert score == 0.0
    assert degraded is True


# ── Feature gating ────────────────────────────────────────────────────────────
def test_feature_names_unchanged_when_graph_disabled(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", False)
    names = FeatureExtractor.get_feature_names()
    assert names == FEATURE_NAMES
    assert len(names) == 38


def test_feature_names_widen_when_graph_enabled(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", True)
    names = FeatureExtractor.get_feature_names()
    assert len(names) == 38 + len(GRAPH_FEATURE_NAMES)
    assert names[-len(GRAPH_FEATURE_NAMES):] == GRAPH_FEATURE_NAMES


def test_extract_appends_zeros_when_graph_features_missing(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", True)
    tx = FakeTransaction()
    vec = FeatureExtractor.extract(tx, [], [], graph_features=None)
    assert vec.shape[0] == len(FeatureExtractor.get_feature_names())
    # The trailing graph slots default to zero when no graph features are supplied.
    assert np.all(vec[-len(GRAPH_FEATURE_NAMES):] == 0.0)


def test_extract_uses_supplied_graph_features(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", True)
    tx = FakeTransaction()
    gf = {
        "distance_to_known_bad": 0.5, "shared_device_count": 4, "shared_ip_count": 2,
        "degree": 7, "pagerank": 0.1, "is_in_cycle": 1, "community_size": 9,
    }
    vec = FeatureExtractor.extract(tx, [], [], graph_features=gf)
    tail = vec[-len(GRAPH_FEATURE_NAMES):]
    assert tail[0] == 0.5     # distance_to_known_bad
    assert tail[1] == 4.0     # shared_device_count
    assert tail[-1] == 9.0    # community_size
