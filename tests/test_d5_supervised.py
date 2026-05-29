"""Tests for the Layer 3 supervised detector (D5).

Trains once per session on a tiny generated dataset, then asserts:
    * all four artifacts (xgb, lgbm, rf, shap) exist
    * a labeled-suspicious-like vector scores high (>60)
    * a clearly-normal vector scores low (<30)
    * shap_values keys are within the feature_names set
    * top_features is non-empty and ordered by |shap|
    * inference path does NOT read hidden_ground_truth.csv
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.system1_generator import g1_account_builder, g2_normal_generator, g3_scenario_injector, g4_splitter
from src.system1_generator.common import CONFIG_PATH, GeneratorConfig, INTERNAL_DIR
from src.system2_detection.layer3_supervised import d5a_training, d5b_inference
from src.system2_detection.layer3_supervised.schemas import SupervisedOutput
from src.system2_detection.shared.s3_artifact_store import ArtifactStore
from src.system2_detection.shared.schemas import LiveFeatureVector, LiveGraphFeatureVector


# --------------------------------------------------------------------------- #
# Session fixture — train once on a small dataset
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Generate a small warmup-like CSV, train D5a, return loaded inferencer."""
    work = tmp_path_factory.mktemp("d5-fixture")
    # Build a small generator population. We use the warmup-style settings:
    # smaller account count, ~8K labeled txns, fraud_ratio nudged a bit higher
    # so the tiny test fixture has enough positives for a stratified split.
    base_cfg = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base_cfg.override(
        seed=base_cfg.seed + 2000,
        num_accounts=400,
        num_historical_transactions=8_000,
        num_stream_transactions=0,
        fraud_ratio=0.05,  # nudge for a few hundred positives at this scale
    )
    ctx = g1_account_builder.build_accounts(cfg)
    accounts_path = work / "accounts.parquet"
    ctx.accounts.to_parquet(accounts_path, index=False)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, suspicious, gt = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    # Write labeled CSV (keep is_suspicious + synthetic_pattern_type)
    drop_cols = [c for c in ("fraud_ring_id", "laundering_stage") if c in combined.columns]
    labeled = combined.drop(columns=drop_cols)
    labeled_csv = work / "warmup_labeled.csv"
    labeled.to_csv(labeled_csv, index=False)

    # Build training matrix & train
    X, y, feat_names, _ = d5a_training.build_training_matrix(
        warmup_csv=labeled_csv, accounts_parquet=accounts_path
    )
    assert y.sum() >= 50, f"too few positives in fixture: {y.sum()}"
    trained = d5a_training.train_models(X, y, feat_names, seed=42)
    artifacts_dir = work / "artifacts"
    d5a_training.save_artifacts(trained, out_dir=artifacts_dir)

    return {
        "work": work,
        "artifacts_dir": artifacts_dir,
        "feature_names": feat_names,
        "trained": trained,
        "X": X,
        "y": y,
        "labeled_df": labeled,
        "suspicious": suspicious,
    }


@pytest.fixture(scope="session")
def inferencer(trained_artifacts):
    store = ArtifactStore(base_dir=trained_artifacts["artifacts_dir"])
    return d5b_inference.SupervisedInferencer(store=store)


# --------------------------------------------------------------------------- #
# Artifact presence
# --------------------------------------------------------------------------- #


def test_all_four_artifacts_persisted(trained_artifacts):
    d = trained_artifacts["artifacts_dir"]
    for name in d5b_inference.SupervisedInferencer.REQUIRED_ARTIFACTS:
        assert (d / name).exists(), f"missing artifact: {name}"


def test_artifacts_loadable(trained_artifacts):
    d = trained_artifacts["artifacts_dir"]
    for name in d5b_inference.SupervisedInferencer.REQUIRED_ARTIFACTS:
        payload = joblib.load(d / name)
        assert isinstance(payload, tuple) and len(payload) == 2
        _, feat_names = payload
        assert feat_names == trained_artifacts["feature_names"]


def test_inferencer_constructs(inferencer):
    assert inferencer.feature_names
    assert hasattr(inferencer, "xgb_cal")
    assert hasattr(inferencer, "lgb_cal")
    assert hasattr(inferencer, "rf_cal")
    assert hasattr(inferencer, "explainer")


# --------------------------------------------------------------------------- #
# Helpers — build a Live*Vector + rule-out triple from a real labeled row
# --------------------------------------------------------------------------- #


class _RuleOutStub:
    """A minimal RuleEngineOutput shim that exposes the two fields D5b reads."""

    def __init__(self, rule_score: float, triggered: list[str]):
        self.rule_score = rule_score
        self.triggered_rules = triggered


def _row_to_inputs(
    inferencer: d5b_inference.SupervisedInferencer,
    X: pd.DataFrame,
    y: np.ndarray,
    suspicious: bool,
):
    """Take a real training row of the requested class and rebuild the three
    live inputs from it so we can call ``inferencer.score()``.

    We don't carry around the actual TransactionEvent / S1 / L2A objects in
    the test — they're already collapsed into ``X``. We unpack ``X`` back into
    the three live vectors so the scoring path exercises the real inference
    code (build_feature_vector -> three predict_proba calls -> shap).
    """
    target = 1 if suspicious else 0
    mask = y == target
    if not mask.any():
        pytest.skip(f"no rows with is_suspicious={target} in fixture")
    row = X[mask].iloc[0]
    row_dict = row.to_dict()

    # LiveFeatureVector — pull all 45 behavioural columns straight off the row.
    from src.system2_detection.shared.schemas import FEATURE_COLUMNS
    feat_kwargs = {c: row_dict[c] for c in FEATURE_COLUMNS}
    features = LiveFeatureVector(
        transaction_id=f"test-{target}",
        sender_account="A-test",
        scaled_feature_vector=[],
        **feat_kwargs,
    )

    # LiveGraphFeatureVector — pull the g_* columns and unmap to sender_*
    from src.system2_detection.layer3_supervised.d5b_inference import _LIVE_GRAPH_FIELD
    graph_kwargs = {}
    for col, (live_field, default) in _LIVE_GRAPH_FIELD.items():
        if live_field is None:
            continue
        graph_kwargs[live_field] = row_dict.get(col, default)
    graph = LiveGraphFeatureVector(
        transaction_id=f"test-{target}",
        sender_account="A-test",
        receiver_account="B-test",
        sender_in_degree=int(graph_kwargs.get("sender_in_degree", 0)),
        sender_out_degree=int(graph_kwargs.get("sender_out_degree", 0)),
        sender_in_degree_unique=int(graph_kwargs.get("sender_in_degree_unique", 0)),
        sender_out_degree_unique=int(graph_kwargs.get("sender_out_degree_unique", 0)),
        sender_2hop_cycle_count=int(graph_kwargs.get("sender_2hop_cycle_count", 0)),
        sender_3hop_cycle_count=int(graph_kwargs.get("sender_3hop_cycle_count", 0)),
        sender_fan_in_score=float(graph_kwargs.get("sender_fan_in_score", 0.0)),
        sender_fan_out_score=float(graph_kwargs.get("sender_fan_out_score", 0.0)),
        sender_pagerank=float(graph_kwargs.get("sender_pagerank", 0.0)),
        sender_betweenness=float(graph_kwargs.get("sender_betweenness", 0.0)),
        sender_community_id=int(graph_kwargs.get("sender_community_id", -1)),
        sender_community_size=int(graph_kwargs.get("sender_community_size", 0)),
        sender_community_density=float(graph_kwargs.get("sender_community_density", 0.0)),
        sender_community_risk_score=0.0,
        sender_benford_chi2_community=float(graph_kwargs.get("sender_benford_chi2_community", 0.0)),
        receiver_in_degree=0,
        receiver_out_degree=0,
        receiver_community_id=-1,
        receiver_community_risk_score=0.0,
        edge_creates_cycle=False,
        cycle_length=0,
        shared_community=False,
    )

    # Rule output — pull the rule_score + which rules fired
    triggered = [c.split("rule_")[1] for c in inferencer.feature_names
                 if c.startswith("rule_R") and row_dict.get(c, 0) > 0.5]
    rule_out = _RuleOutStub(rule_score=float(row_dict.get("rule_score", 0.0)), triggered=triggered)

    return features, graph, rule_out


# --------------------------------------------------------------------------- #
# Score behaviour
# --------------------------------------------------------------------------- #


def test_score_returns_supervised_output(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    assert isinstance(out, SupervisedOutput)
    assert 0.0 <= out.fraud_probability <= 1.0
    assert abs(out.supervised_score - out.fraud_probability * 100.0) < 1e-6


def test_suspicious_vector_scores_high(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    assert out.supervised_score > 60.0, f"suspicious sample scored {out.supervised_score:.1f}"


def test_normal_vector_scores_low(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=False
    )
    out = inferencer.score(features, graph, rule_out)
    assert out.supervised_score < 30.0, f"normal sample scored {out.supervised_score:.1f}"


# --------------------------------------------------------------------------- #
# SHAP / model agreement structure
# --------------------------------------------------------------------------- #


def test_shap_values_keys_are_known_features(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    name_set = set(inferencer.feature_names)
    assert set(out.shap_values.keys()) <= name_set
    assert out.top_features, "top_features is empty"
    assert all(t in name_set for t in out.top_features)


def test_top_features_ordered_by_abs_shap(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    abs_vals = [abs(out.shap_values[name]) for name in out.top_features]
    # Monotone non-increasing
    assert all(abs_vals[i] >= abs_vals[i + 1] - 1e-9 for i in range(len(abs_vals) - 1))


def test_model_agreement_in_unit_range(inferencer, trained_artifacts):
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    assert 0.0 <= out.model_agreement <= 1.0


# --------------------------------------------------------------------------- #
# Provenance contract
# --------------------------------------------------------------------------- #


def test_inference_never_reads_hidden_ground_truth(inferencer, trained_artifacts, monkeypatch):
    """D5b must not touch hidden_ground_truth.csv during inference.

    We intercept ``pandas.read_csv`` and assert no call argument contains
    ``hidden_ground_truth``. Any read of that file would be a contract
    violation — the file is reserved for System 3.
    """
    real_read_csv = pd.read_csv

    def guard(*args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, (str, Path)) and "hidden_ground_truth" in str(a).lower():
                raise AssertionError("inferencer attempted to read hidden_ground_truth.csv")
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard)
    features, graph, rule_out = _row_to_inputs(
        inferencer, trained_artifacts["X"], trained_artifacts["y"], suspicious=True
    )
    out = inferencer.score(features, graph, rule_out)
    assert out is not None


def test_training_doesnt_open_hidden_ground_truth(tmp_path, monkeypatch):
    """Same contract for training: build_training_matrix must not open
    hidden_ground_truth.csv. We assert against real I/O via monkeypatching
    pandas.read_csv during a tiny training-matrix build.
    """
    # Build a tiny synthetic CSV so the matrix builder has something to read
    base_cfg = GeneratorConfig.from_file(CONFIG_PATH)
    cfg = base_cfg.override(
        seed=base_cfg.seed + 5000,
        num_accounts=80, num_historical_transactions=400,
        num_stream_transactions=0, fraud_ratio=0.10,
    )
    ctx = g1_account_builder.build_accounts(cfg)
    accounts_path = tmp_path / "accounts.parquet"
    ctx.accounts.to_parquet(accounts_path, index=False)
    normal_df = g2_normal_generator.generate_normal(cfg, ctx)
    combined, _, _ = g3_scenario_injector.inject_scenarios(cfg, ctx, normal_df)
    labeled_csv = tmp_path / "warmup_labeled.csv"
    combined.to_csv(labeled_csv, index=False)

    real_read_csv = pd.read_csv

    def guard(*args, **kwargs):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, (str, Path)) and "hidden_ground_truth" in str(a).lower():
                raise AssertionError("training attempted to read hidden_ground_truth.csv")
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guard)
    X, y, feat_names, _ = d5a_training.build_training_matrix(
        warmup_csv=labeled_csv, accounts_parquet=accounts_path
    )
    assert len(X) > 0 and len(feat_names) > 0
