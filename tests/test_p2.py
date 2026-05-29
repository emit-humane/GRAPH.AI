"""Tests for the P2 Alert Management System."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.system2_detection.post_detection.d8_fusion import (
    LayerScores,
    RiskFusionEngine,
)
from src.system2_detection.post_detection.p2_alerts.alert_manager import (
    CSV_COLUMNS,
    AlertManager,
)
from src.system2_detection.post_detection.p2_alerts.api import build_router
from src.system2_detection.post_detection.p2_alerts.models import (
    AlertStatus,
    create_engine_from_url,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def manager(tmp_path: Path) -> AlertManager:
    """Fresh SQLite-backed AlertManager per test, in a tempdir."""
    db_path = tmp_path / "alerts.db"
    return AlertManager(database_url=f"sqlite:///{db_path.as_posix()}")


@pytest.fixture
def fusion() -> RiskFusionEngine:
    return RiskFusionEngine()


def _make_layer_scores(**overrides) -> LayerScores:
    base = LayerScores(transaction_id="tx-p2", sender_account="A-test")
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _high_risk_scores() -> LayerScores:
    """A combo of scores that fuses well above the High threshold (61)."""
    return _make_layer_scores(
        rule_score=80, graph_score=80, supervised_score=80,
        anomaly_score=80, tgn_score=80,
        triggered_rules=["R01", "R10"],
        rule_explanations=["high amt", "cycle"],
        top_features=["amount_raw", "channel_Web"],
        anomaly_drivers=["amount_log"],
        temporal_graph_explanations=["Edge closes a 3-hop cycle."],
        has_cycle=True,
        sender_community_risk_score=70,
        community_density=0.5,
    )


def _low_risk_scores() -> LayerScores:
    return _make_layer_scores(
        rule_score=10, graph_score=10, supervised_score=5,
        anomaly_score=5, tgn_score=5,
    )


# --------------------------------------------------------------------------- #
# Trigger logic
# --------------------------------------------------------------------------- #


def test_low_risk_event_does_not_create_alert(manager, fusion):
    fused = fusion.fuse(_low_risk_scores())
    assert manager.process(fused) is None
    assert manager.count() == 0


def test_high_risk_event_creates_alert(manager, fusion):
    fused = fusion.fuse(_high_risk_scores())
    alert = manager.process(fused, community_id=42)
    assert alert is not None
    assert alert.alert_status == AlertStatus.OPEN.value
    assert alert.community_id == 42
    assert manager.count() == 1


def test_alert_includes_all_five_layer_scores_in_breakdown(manager, fusion):
    fused = fusion.fuse(_high_risk_scores())
    manager.process(fused)
    items = manager.list()
    assert len(items) == 1
    breakdown = items[0]["score_breakdown"]
    scores = breakdown["scores"]
    for name in ("rule_score", "graph_score", "supervised_score", "anomaly_score", "tgn_score"):
        assert name in scores, f"breakdown missing {name}"


def test_top_shap_features_persisted_on_alert(manager, fusion):
    s = _high_risk_scores()
    s.top_features = ["amount_raw", "channel_Web", "ip_change_count_24h"]
    fused = fusion.fuse(s)
    manager.process(fused)
    items = manager.list()
    assert items[0]["top_shap_features"] == ["amount_raw", "channel_Web", "ip_change_count_24h"]


def test_breakdown_missing_layer_keys_raises(manager, fusion):
    """If anyone hands us a fused output whose breakdown is missing one of the
    five layer scores, we refuse to persist — protects the audit trail."""
    fused = fusion.fuse(_high_risk_scores())
    # Manually clobber the score_breakdown so it's missing a layer.
    fused.score_breakdown["scores"].pop("tgn_score")
    with pytest.raises(ValueError):
        manager.process(fused)


# --------------------------------------------------------------------------- #
# Listing / status updates
# --------------------------------------------------------------------------- #


def test_list_filters_by_risk_level(manager, fusion):
    # Two High alerts (different transactions)
    for i, suffix in enumerate(("A", "B")):
        s = _high_risk_scores()
        s.transaction_id = f"tx-{suffix}"
        s.sender_account = f"acc-{suffix}"
        manager.process(fusion.fuse(s))
    # A Low one too (shouldn't trigger)
    manager.process(fusion.fuse(_low_risk_scores()))

    all_alerts = manager.list()
    assert len(all_alerts) == 2
    high_only = manager.list(risk_level="High")
    critical_only = manager.list(risk_level="Critical")
    # Total filter results equal the total triggers
    assert len(high_only) + len(critical_only) == 2


def test_update_status_to_investigating_persists(manager, fusion):
    alert = manager.process(fusion.fuse(_high_risk_scores()))
    assert alert is not None
    updated = manager.update_status(
        alert.alert_id, AlertStatus.INVESTIGATING.value, assigned_to="analyst-1"
    )
    assert updated is not None
    assert updated["alert_status"] == "Investigating"
    assert updated["assigned_to"] == "analyst-1"
    # Persisted across read
    reread = manager.get(alert.alert_id)
    assert reread is not None
    assert reread["alert_status"] == "Investigating"


def test_update_status_invalid_value_raises(manager, fusion):
    alert = manager.process(fusion.fuse(_high_risk_scores()))
    assert alert is not None
    with pytest.raises(ValueError):
        manager.update_status(alert.alert_id, "Resolved")  # not in the enum


def test_update_status_missing_alert_returns_none(manager):
    assert manager.update_status("does-not-exist", AlertStatus.CLOSED.value) is None


# --------------------------------------------------------------------------- #
# CSV export (spec column set)
# --------------------------------------------------------------------------- #


def test_export_csv_writes_spec_columns(manager, fusion, tmp_path):
    s = _high_risk_scores()
    s.transaction_id = "tx-csv"
    s.sender_account = "acc-csv"
    manager.process(fusion.fuse(s), community_id=7)

    out = manager.export_csv(tmp_path / "generated_alerts.csv")
    assert out.exists()
    with open(out, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == list(CSV_COLUMNS)
    assert len(rows) == 1
    row = rows[0]
    assert row["transaction_id"] == "tx-csv"
    assert row["community_id"] == "7"
    # triggered_patterns column is JSON-encoded
    patterns = json.loads(row["triggered_patterns"])
    assert isinstance(patterns, list)


def test_export_csv_with_no_alerts_writes_header_only(manager, tmp_path):
    out = manager.export_csv(tmp_path / "empty.csv")
    assert out.exists()
    with open(out, "r", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows == [list(CSV_COLUMNS)]


# --------------------------------------------------------------------------- #
# FastAPI router
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(manager: AlertManager) -> TestClient:
    app = FastAPI()
    app.include_router(build_router(manager))
    return TestClient(app)


def test_api_list_and_get(client, manager, fusion):
    alert = manager.process(fusion.fuse(_high_risk_scores()))
    assert alert is not None
    r = client.get("/alerts")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["alert_id"] == alert.alert_id

    r = client.get(f"/alerts/{alert.alert_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["transaction_id"] == "tx-p2"


def test_api_get_missing_returns_404(client):
    r = client.get("/alerts/does-not-exist")
    assert r.status_code == 404


def test_api_patch_status(client, manager, fusion):
    alert = manager.process(fusion.fuse(_high_risk_scores()))
    assert alert is not None
    r = client.patch(
        f"/alerts/{alert.alert_id}/status",
        json={"alert_status": "Investigating", "assigned_to": "analyst-2"},
    )
    assert r.status_code == 200
    assert r.json()["alert_status"] == "Investigating"

    # Bad value -> 400
    bad = client.patch(
        f"/alerts/{alert.alert_id}/status",
        json={"alert_status": "Done"},
    )
    assert bad.status_code == 400


def test_api_export_csv_endpoint(client, manager, fusion, tmp_path):
    manager.process(fusion.fuse(_high_risk_scores()))
    target = tmp_path / "api_export.csv"
    r = client.post("/alerts/export-csv", json={"out_path": str(target)})
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 1
    assert body["columns"] == list(CSV_COLUMNS)
    assert Path(body["path"]).exists()


def test_api_statuses_meta_endpoint(client):
    r = client.get("/alerts/_meta/statuses")
    assert r.status_code == 200
    assert set(r.json()) == {"Open", "Investigating", "SAR_Filed", "Closed"}


# --------------------------------------------------------------------------- #
# Clear / admin
# --------------------------------------------------------------------------- #


def test_clear_removes_all_alerts(manager, fusion):
    manager.process(fusion.fuse(_high_risk_scores()))
    assert manager.count() == 1
    removed = manager.clear()
    assert removed == 1
    assert manager.count() == 0
