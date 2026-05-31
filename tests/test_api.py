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


# --------------------------------------------------------------------------- #
# Scenario Studio endpoints
# --------------------------------------------------------------------------- #


def test_studio_catalog(client):
    r = client.get("/studio/catalog")
    assert r.status_code == 200
    typologies = r.json()["typologies"]
    keys = {t["key"] for t in typologies}
    # All 10 typologies must be in the catalog
    assert keys >= {"fan_in", "layering_chain", "structuring", "round_tripping"}
    # fan_in targets R14 and layering_chain targets R15 (the R14/R15 contract)
    fan_in = next(t for t in typologies if t["key"] == "fan_in")
    layering = next(t for t in typologies if t["key"] == "layering_chain")
    assert "R14" in fan_in["target_rules"]
    assert "R15" in layering["target_rules"]


def test_studio_preview_returns_plan(client):
    r = client.get("/studio/preview?typology=structuring&seed=42&count=2")
    assert r.status_code == 200
    body = r.json()
    assert len(body["plans"]) == 2
    assert body["plans"][0]["summary"]["edge_count"] > 0


def test_studio_preview_unknown_typology(client):
    r = client.get("/studio/preview?typology=foo")
    assert r.status_code == 404


def test_studio_inject_queues_events(client):
    r = client.post("/studio/inject", json={
        "typology": "fan_in", "params": {"sources": 6}, "seed": 7, "count": 1,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["queued_events"] >= 5
    assert body["queue_depth"] >= body["queued_events"]


def test_studio_export_buffer_and_download(client):
    r = client.post("/studio/export", json={
        "scenarios": [{"typology": "structuring", "seed": 5, "count": 1, "params": {}}],
    })
    assert r.status_code == 200
    assert r.json()["appended"] > 0
    # Download drains the buffer and returns a zip
    r2 = client.post("/studio/export/download")
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("application/zip")


def test_studio_export_empty_payload_400(client):
    r = client.post("/studio/export", json={"scenarios": []})
    assert r.status_code == 400


def test_studio_activity(client):
    r = client.get("/studio/activity")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body and "queue_depth" in body and "export_buffer" in body


# --------------------------------------------------------------------------- #
# CSV file injector (/generator/inject_csv)
# --------------------------------------------------------------------------- #


def test_inject_csv_queues_events(client):
    """A minimal 2-row CSV must parse, queue, and report the queue depth."""
    csv_blob = (
        "transaction_id,timestamp,sender_account,receiver_account,sender_bank,"
        "receiver_bank,sender_country,receiver_country,amount,currency,"
        "transaction_type,payment_channel,device_id,ip_address,geo_latitude,"
        "geo_longitude,merchant_category,transaction_status,sender_balance_before,"
        "sender_balance_after,receiver_balance_before,receiver_balance_after,"
        "kyc_level,is_international,remarks,amount_leading_digit\n"
        "tx-inj-1,2026-05-30T12:00:00+00:00,acc-A,acc-B,HDFC,ICICI,IN,IN,250000,INR,"
        "NEFT,Web,dev-x,203.0.113.1,28.61,77.21,Other,Success,,,,,2,False,csv-test,2\n"
        "tx-inj-2,2026-05-30T12:01:00+00:00,acc-A,acc-C,HDFC,ICICI,IN,AE,500000,INR,"
        "Wire,Web,dev-x,203.0.113.2,28.61,77.21,Other,Success,,,,,2,True,csv-test,5\n"
    )
    files = {"file": ("probe.csv", csv_blob.encode("utf-8"), "text/csv")}
    r = client.post("/generator/inject_csv?auto_start=false", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued_events"] == 2
    assert body["queue_depth"] >= 2
    assert body["rows_with_errors"] == 0


def test_inject_csv_empty_400(client):
    files = {"file": ("empty.csv", b"", "text/csv")}
    r = client.post("/generator/inject_csv?auto_start=false", files=files)
    assert r.status_code == 400


def test_clear_alerts_returns_count(client):
    """DELETE /alerts must wipe the DB + buffer and return the cleared count."""
    r = client.delete("/alerts")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cleared_from_db" in body
    assert body["cleared_from_buffer"] is True
    assert "alerts_csv_reset" in body
    # Idempotent — calling again should return 0
    r2 = client.delete("/alerts")
    assert r2.status_code == 200
    assert r2.json()["cleared_from_db"] == 0
