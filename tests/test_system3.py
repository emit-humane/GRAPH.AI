"""System 3 — Evaluation Engine tests (E1 → E5).

The five evaluators all operate on tiny synthetic CSVs we build inline.
Coverage:

* E1 emits a frame with the spec's y_true / y_pred / y_score columns and
  attaches per-layer scores when the alerts DB is available.
* E2 returns the spec's eleven transaction-level metrics AND the new
  ``per_layer_detection_credit`` block — verified with controlled inputs
  where we know the supervised layer should always lead.
* E3 produces fraud_ring_detection_rate / mean_ring_coverage that match
  hand-computed expectations.
* E4 emits per_rule_precision covering ALL 15 rules including R14_fan_in
  AND R15_layering_relay, AND a per_pattern_detection_rate that lists every
  typology key.
* E5 carries the external benchmarks block with ``your_pr_auc`` and the
  delta vs each published benchmark.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.system3_evaluation.e1_joiner import (
    E1Joiner,
    JOINED_FRAME_COLUMNS,
    LAYER_KEYS,
)
from src.system3_evaluation.e2_transaction_eval import (
    E2TransactionEvaluator,
    LAYER_DISPLAY_TO_CONTRIB,
)
from src.system3_evaluation.e3_community_eval import E3CommunityEvaluator
from src.system3_evaluation.e4_pattern_eval import (
    ALL_TYPOLOGIES,
    E4PatternEvaluator,
    RULE_DISPLAY_NAMES,
)
from src.system3_evaluation.e5_report import EXTERNAL_BENCHMARKS, E5ReportGenerator


# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build a tiny but interesting dataset:
        * 10 stream transactions (5 suspicious, 5 normal)
        * 6 alerts (4 TPs, 2 FPs)
        * 1 suspicious is missed (1 FN) — that's a layering_chain
        * 4 fraud rings, two fully alerted, one partial, one missed
        * Alerts carry triggered_patterns including R02, R10, R14, R15
        * Alerts DB carries score_breakdowns where 'supervised' is the leader
    """
    base = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    # ---- stream ----------------------------------------------------------
    stream_rows = []
    for i in range(10):
        stream_rows.append({
            "transaction_id": f"tx{i:02d}",
            "timestamp": (base + timedelta(seconds=i * 60)).isoformat(),
            "sender_account": f"acct-{i % 4}",
        })
    stream_path = tmp_path / "stream_transactions.csv"
    pd.DataFrame(stream_rows).to_csv(stream_path, index=False)

    # ---- ground truth ----------------------------------------------------
    # tx00,tx01 = structuring (ring A) — both flagged
    # tx02,tx03 = layering_chain (ring B) — only tx02 flagged (FN on tx03)
    # tx04      = fan_in (ring C) — flagged
    # tx05      = round_tripping (ring D) — NOT flagged (missed ring)
    # tx06-09   = normal
    gt_rows = [
        # (tx_id, sus, ring, cluster, pattern, severity)
        ("tx00", "Suspicious", "ringA", "clusterA", "structuring", "High"),
        ("tx01", "Suspicious", "ringA", "clusterA", "structuring", "High"),
        ("tx02", "Suspicious", "ringB", "clusterB", "layering_chain", "Critical"),
        ("tx03", "Suspicious", "ringB", "clusterB", "layering_chain", "Critical"),
        ("tx04", "Suspicious", "ringC", "clusterC", "fan_in", "High"),
        ("tx05", "Suspicious", "ringD", "clusterD", "round_tripping", "Medium"),
        ("tx06", "Normal",     "", "",       "", ""),
        ("tx07", "Normal",     "", "",       "", ""),
        ("tx08", "Normal",     "", "",       "", ""),
        ("tx09", "Normal",     "", "",       "", ""),
    ]
    gt_path = tmp_path / "hidden_ground_truth.csv"
    pd.DataFrame(gt_rows, columns=[
        "transaction_id", "suspicious_flag", "fraud_ring_id",
        "suspicious_cluster_id", "synthetic_pattern_type", "scenario_severity",
    ]).to_csv(gt_path, index=False)

    # ---- alerts ----------------------------------------------------------
    # Six alerts → 4 TPs (tx00..tx02, tx04), 2 FPs (tx06, tx07)
    alerts_rows = []
    pattern_map = {
        "tx00": ["L1:R02", "L1:R11", "L3:amount_raw"],                    # structuring -> R02
        "tx01": ["L1:R02", "L3:amount_raw"],                              # structuring
        "tx02": ["L1:R15", "L1:R06", "L3:channel_Web"],                   # layering -> R15
        "tx04": ["L1:R14", "L2:community_has_cycle", "L4:fan_in_score"],  # fan_in -> R14
        "tx06": ["L1:R03"],                                               # FP
        "tx07": ["L1:R10"],                                               # FP
    }
    for tx_id, patterns in pattern_map.items():
        alerts_rows.append({
            "alert_id":             f"a-{tx_id}",
            "transaction_id":       tx_id,
            "sender_account":       f"acct-{int(tx_id[2:]) % 4}",
            "community_id":         int(tx_id[2:]) % 3,
            "transaction_risk_score": 75.0 if tx_id != "tx07" else 65.0,
            "group_risk_score":     85.0,
            "risk_level":           "High",
            "triggered_patterns":   json.dumps(patterns),
            "alert_status":         "Open",
            "created_at":           (base + timedelta(seconds=int(tx_id[2:]) * 60, milliseconds=200)).isoformat(),
        })
    alerts_path = tmp_path / "generated_alerts.csv"
    pd.DataFrame(alerts_rows).to_csv(alerts_path, index=False)

    # ---- alerts DB (per-layer score_breakdown) ---------------------------
    alerts_db_path = tmp_path / "alerts.db"
    _build_alerts_db(alerts_db_path, alerts_rows)

    return {
        "stream": stream_path,
        "alerts": alerts_path,
        "ground_truth": gt_path,
        "alerts_db": alerts_db_path,
    }


def _build_alerts_db(db_path: Path, alerts_rows: list[dict]) -> None:
    """Create a minimal SQLite alerts.db with the columns E1 reads.

    For TPs we set ``supervised_score`` highest → those alerts should be
    credited to the supervised layer. For FPs we let the rule layer lead.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE alerts (
            alert_id TEXT PRIMARY KEY,
            transaction_id TEXT,
            sender_account TEXT,
            community_id INTEGER,
            transaction_risk_score REAL,
            group_risk_score REAL,
            risk_level TEXT,
            risk_level_group TEXT,
            triggered_patterns TEXT,
            rule_explanations TEXT,
            anomaly_drivers TEXT,
            structural_anomaly_explanations TEXT,
            score_breakdown TEXT,
            top_shap_features TEXT,
            explanation TEXT,
            alert_status TEXT,
            assigned_to TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    tp_ids = {"tx00", "tx01", "tx02", "tx04"}
    for row in alerts_rows:
        if row["transaction_id"] in tp_ids:
            scores = {"rule_score": 60, "graph_score": 30, "supervised_score": 92,
                      "anomaly_score": 50, "tgn_score": 25}
        else:
            scores = {"rule_score": 70, "graph_score": 20, "supervised_score": 40,
                      "anomaly_score": 30, "tgn_score": 10}
        weights = {"rule_score": 0.20, "graph_score": 0.15, "supervised_score": 0.35,
                   "anomaly_score": 0.20, "tgn_score": 0.10}
        contributions = {k: round(scores[k] * weights[k], 4) for k in scores}
        breakdown = {
            "scores": scores,
            "weights": weights,
            "weighted_contributions": contributions,
            "total": sum(contributions.values()),
            "fusion_mode": "static_weights",
        }
        cur.execute(
            "INSERT INTO alerts (alert_id, transaction_id, sender_account, community_id,"
            " transaction_risk_score, group_risk_score, risk_level, risk_level_group,"
            " triggered_patterns, rule_explanations, anomaly_drivers,"
            " structural_anomaly_explanations, score_breakdown, top_shap_features,"
            " explanation, alert_status, assigned_to, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["alert_id"], row["transaction_id"], row["sender_account"],
                row["community_id"], row["transaction_risk_score"], row["group_risk_score"],
                row["risk_level"], row["risk_level"],
                row["triggered_patterns"], json.dumps([]), json.dumps([]), json.dumps([]),
                json.dumps(breakdown), json.dumps([]),
                "", row["alert_status"], None, row["created_at"], row["created_at"],
            ),
        )
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# E1 — joiner
# --------------------------------------------------------------------------- #


def test_e1_joined_frame_shape(synthetic_inputs, tmp_path):
    joiner = E1Joiner(
        stream_path=synthetic_inputs["stream"],
        alerts_path=synthetic_inputs["alerts"],
        ground_truth_path=synthetic_inputs["ground_truth"],
        alerts_db_path=synthetic_inputs["alerts_db"],
        output_path=tmp_path / "joined.parquet",
    )
    frame = joiner.run()
    # Every published column is present
    missing = [c for c in JOINED_FRAME_COLUMNS if c not in frame.columns]
    assert not missing, f"missing columns from joined frame: {missing}"
    # Every stream tx is represented
    assert len(frame) == 10
    # y_true / y_pred match the fixture: 6 suspicious (tx00-tx05), 6 alerts (2 FNs)
    assert int(frame["y_true"].sum()) == 6
    assert int(frame["y_pred"].sum()) == 6
    # Layer score columns populated for the 6 alerted rows
    has_scores = frame.dropna(subset=[f"layer_{k}" for k in LAYER_KEYS], how="all")
    assert len(has_scores) == 6


def test_e1_graceful_without_alerts_db(synthetic_inputs, tmp_path):
    """Frame should still build when no alerts DB is provided."""
    joiner = E1Joiner(
        stream_path=synthetic_inputs["stream"],
        alerts_path=synthetic_inputs["alerts"],
        ground_truth_path=synthetic_inputs["ground_truth"],
        alerts_db_path=None,
        output_path=tmp_path / "joined.parquet",
    )
    frame = joiner.run()
    assert len(frame) == 10
    # Layer score columns are present (per the schema) but all NaN
    for k in LAYER_KEYS:
        assert frame[f"layer_{k}"].isna().all()


# --------------------------------------------------------------------------- #
# E2 — transaction metrics
# --------------------------------------------------------------------------- #


def test_e2_emits_all_required_keys(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E2TransactionEvaluator(frame=frame, output_path=tmp_path / "tx.json").run()
    required = {
        "precision", "recall", "f1_score_macro", "minority_class_f1",
        "pr_auc", "roc_auc",
        "false_positive_rate", "false_negative_rate",
        "total_alerts_generated",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
        "per_layer_detection_credit",
    }
    assert required.issubset(metrics.keys()), required - metrics.keys()

    # Sanity on the counts: 6 truth-pos, 6 alerts, 2 FNs -> 4 TPs, 2 FPs
    assert metrics["true_positives"] == 4
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 2
    assert metrics["true_negatives"] == 2


def test_e2_per_layer_detection_credit_keys(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E2TransactionEvaluator(frame=frame, output_path=tmp_path / "tx.json").run()
    credit = metrics["per_layer_detection_credit"]
    assert {"rule", "graph", "supervised", "anomaly", "tgn"} <= set(credit.keys())
    assert "total_tp_with_breakdown" in credit
    assert "by_layer_mean_score" in credit
    assert "credit_basis" in credit
    # We rigged the breakdowns so supervised is the highest contribution
    # on every TP — supervised should lead 100% of TPs.
    assert credit["total_tp_with_breakdown"] == 4
    assert credit["supervised"] == pytest.approx(1.0)
    assert credit["rule"] == pytest.approx(0.0)
    assert credit["credit_basis"] == "weighted_contribution"


def test_e2_per_layer_credit_with_no_breakdown_is_zero(synthetic_inputs, tmp_path):
    """When the DB isn't available, per_layer_detection_credit stays
    structured but reports zero / 'unavailable'."""
    joiner = E1Joiner(
        stream_path=synthetic_inputs["stream"],
        alerts_path=synthetic_inputs["alerts"],
        ground_truth_path=synthetic_inputs["ground_truth"],
        alerts_db_path=None,
        output_path=tmp_path / "joined.parquet",
    )
    frame = joiner.run()
    metrics = E2TransactionEvaluator(frame=frame, output_path=tmp_path / "tx.json").run()
    credit = metrics["per_layer_detection_credit"]
    # Schema preserved
    assert {"rule", "graph", "supervised", "anomaly", "tgn"} <= set(credit.keys())
    assert credit["credit_basis"] in ("unavailable", "none")


# --------------------------------------------------------------------------- #
# E3 — community metrics
# --------------------------------------------------------------------------- #


def test_e3_ring_detection_rate(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E3CommunityEvaluator(frame=frame, output_path=tmp_path / "comm.json").run()

    # 4 rings: A (2/2 alerted), B (1/2 alerted = exactly 0.5 ⇒ detected),
    # C (1/1 alerted), D (0/1) → 3/4 detected
    assert metrics["total_rings"] == 4
    assert metrics["detected_rings"] == 3
    assert metrics["partial_rings"] == 3
    assert metrics["fraud_ring_detection_rate"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# E4 — pattern + per-rule precision (incl. R14, R15)
# --------------------------------------------------------------------------- #


def test_e4_per_rule_precision_includes_r14_and_r15(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E4PatternEvaluator(frame=frame, output_path=tmp_path / "pat.json").run()

    per_rule = metrics["per_rule_precision"]
    # All 15 rules must be present
    expected_keys = set(RULE_DISPLAY_NAMES.values())
    assert set(per_rule.keys()) == expected_keys
    assert "R14_fan_in" in per_rule
    assert "R15_layering_relay" in per_rule

    # R14 fired once on tx04 (a TP) → precision 1.0
    assert metrics["per_rule_fired_count"]["R14_fan_in"] == 1
    assert per_rule["R14_fan_in"] == pytest.approx(1.0)
    # R15 fired once on tx02 (a TP) → precision 1.0
    assert metrics["per_rule_fired_count"]["R15_layering_relay"] == 1
    assert per_rule["R15_layering_relay"] == pytest.approx(1.0)
    # R02 fired on tx00 + tx01 (both TPs) → precision 1.0
    assert metrics["per_rule_fired_count"]["R02_structuring"] == 2
    assert per_rule["R02_structuring"] == pytest.approx(1.0)
    # R10 fired only on FP tx07 → precision 0.0
    assert metrics["per_rule_fired_count"]["R10_cycle_closure"] == 1
    assert per_rule["R10_cycle_closure"] == pytest.approx(0.0)


def test_e4_per_pattern_keys_cover_all_typologies(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E4PatternEvaluator(frame=frame, output_path=tmp_path / "pat.json").run()
    per_pattern = metrics["per_pattern_detection_rate"]
    # Every published typology key is present
    assert set(ALL_TYPOLOGIES).issubset(per_pattern.keys())
    # Structuring 2/2 truth and 2/2 alerted → 1.0
    assert per_pattern["structuring"] == pytest.approx(1.0)
    # Layering chain 2 truth, 1 alerted → 0.5
    assert per_pattern["layering_chain"] == pytest.approx(0.5)
    # Round-tripping is the missed ring → 0
    assert per_pattern["round_tripping"] == pytest.approx(0.0)


def test_e4_latency_metrics_present(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    metrics = E4PatternEvaluator(frame=frame, output_path=tmp_path / "pat.json").run()
    lat = metrics["latency_metrics"]
    assert {"mean_alert_latency_ms", "p95_alert_latency_ms", "p99_alert_latency_ms"} <= set(lat.keys())
    # We baked in a 200ms latency on every alert
    assert lat["mean_alert_latency_ms"] == pytest.approx(200.0, abs=1.0)


# --------------------------------------------------------------------------- #
# E5 — report card
# --------------------------------------------------------------------------- #


def test_e5_external_benchmarks_block(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    tx = E2TransactionEvaluator(frame=frame, output_path=tmp_path / "tx.json").run()
    comm = E3CommunityEvaluator(frame=frame, output_path=tmp_path / "comm.json").run()
    pat = E4PatternEvaluator(frame=frame, output_path=tmp_path / "pat.json").run()
    report = E5ReportGenerator(
        frame=frame,
        transaction_metrics=tx,
        community_metrics=comm,
        pattern_metrics=pat,
        output_path=tmp_path / "report.json",
    ).run()

    bench = report["external_benchmarks"]
    # All published benchmarks present + our two numbers
    for key in EXTERNAL_BENCHMARKS:
        assert key in bench, f"missing benchmark {key}"
    assert "your_pr_auc" in bench
    assert "your_minority_f1" in bench
    # Deltas present
    assert "vs_tide_hi_delta_pr_auc" in bench
    assert "vs_mega_gnn_delta_minority_f1" in bench
    # When we have layer scores, supervised PR-AUC should also be reported
    assert "your_supervised_pr_auc" in bench


def test_e5_report_card_carries_all_sections(synthetic_inputs, tmp_path):
    frame = _build_frame(synthetic_inputs, tmp_path)
    tx = E2TransactionEvaluator(frame=frame, output_path=tmp_path / "tx.json").run()
    comm = E3CommunityEvaluator(frame=frame, output_path=tmp_path / "comm.json").run()
    pat = E4PatternEvaluator(frame=frame, output_path=tmp_path / "pat.json").run()
    report = E5ReportGenerator(
        frame=frame,
        transaction_metrics=tx,
        community_metrics=comm,
        pattern_metrics=pat,
        output_path=tmp_path / "report.json",
    ).run()
    required_sections = {
        "run_id", "evaluated_at", "dataset_stats",
        "transaction_level_metrics", "community_level_metrics",
        "pattern_coverage", "rule_performance", "latency_metrics",
        "per_layer_detection_credit", "external_benchmarks",
    }
    assert required_sections.issubset(report.keys()), required_sections - report.keys()
    # rule_performance carries all 15 rule keys (incl. R14, R15)
    rp = report["rule_performance"]["per_rule_precision"]
    assert "R14_fan_in" in rp
    assert "R15_layering_relay" in rp
    # per_layer_detection_credit carries the five layer keys
    assert {"rule", "graph", "supervised", "anomaly", "tgn"} <= set(
        report["per_layer_detection_credit"].keys()
    )


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #


def _build_frame(inputs: dict[str, Path], tmp_path: Path) -> pd.DataFrame:
    return E1Joiner(
        stream_path=inputs["stream"],
        alerts_path=inputs["alerts"],
        ground_truth_path=inputs["ground_truth"],
        alerts_db_path=inputs["alerts_db"],
        output_path=tmp_path / "joined.parquet",
    ).run()
