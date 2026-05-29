"""Smoke tests for the GRAPH.AI FastAPI surface.

These don't exercise the SSE stream end-to-end (would require a running event
loop with a real generator task). They just confirm every router is mounted
and returns the expected shape.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "GRAPH.AI"
    assert "/generator/start" in body["endpoints"]
    assert "/stream/events" in body["endpoints"]
    assert "/alerts" in body["endpoints"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generator_status_shape(client):
    r = client.get("/generator/status")
    assert r.status_code == 200
    body = r.json()
    assert {"running", "events_emitted", "config"} <= set(body.keys())
    assert {"min_amount", "max_amount", "interval_seconds"} <= set(body["config"].keys())


def test_alerts_listing(client):
    r = client.get("/alerts?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body and "count" in body
    if body["items"]:
        first = body["items"][0]
        # UI-shape fields
        assert "transaction_id" in first and "layer_scores" in first
        assert {"rule", "graph", "supervised", "anomaly", "tgn"} <= set(first["layer_scores"].keys())


def test_stats_evaluation_shape(client):
    r = client.get("/stats/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert {"events_seen", "alerts_total", "risk_distribution", "pattern_counts", "rule_counts"} <= set(body.keys())


def test_unknown_case_404(client):
    r = client.get("/graph/case/does-not-exist")
    assert r.status_code == 404
    r2 = client.get("/geo/case/does-not-exist")
    assert r2.status_code == 404
    r3 = client.get("/report/does-not-exist")
    assert r3.status_code == 404
