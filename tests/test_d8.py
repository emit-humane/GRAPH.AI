"""Tests for the D8 Risk Fusion + Explanation Assembler."""

from __future__ import annotations

import pytest

from src.system2_detection.post_detection.d8_fusion import (
    FUSION_WEIGHTS,
    LayerScores,
    RiskFusionEngine,
    collect_layer_scores,
    risk_level_for,
)
from src.system2_detection.post_detection.schemas import FusedRiskOutput


# --------------------------------------------------------------------------- #
# Weights + thresholds
# --------------------------------------------------------------------------- #


def test_default_fusion_weights_sum_to_one():
    assert abs(sum(FUSION_WEIGHTS.values()) - 1.0) < 1e-9


def test_v2_weight_ordering_rule_and_graph_dominant():
    """v2 contract: rule + graph carry > 50% of total weight, supervised < 25%.

    Locks in the design rationale documented in d8_fusion's module docstring.
    Regulators / auditors expect deterministic + structural layers to dominate
    over the black-box ML.
    """
    rule_plus_graph = FUSION_WEIGHTS["rule_score"] + FUSION_WEIGHTS["graph_score"]
    assert rule_plus_graph > 0.50, (
        f"rule+graph weight {rule_plus_graph:.2f} must dominate fusion (>0.50)"
    )
    assert FUSION_WEIGHTS["supervised_score"] < 0.25, (
        "supervised weight must be < 0.25 to limit black-box dependence"
    )
    # Rule is the single largest signal
    assert FUSION_WEIGHTS["rule_score"] == max(FUSION_WEIGHTS.values()), (
        "Layer-1 rules must be the heaviest single signal"
    )


def test_engine_asserts_weight_sum():
    with pytest.raises(AssertionError):
        RiskFusionEngine(weights={
            "rule_score": 0.5, "graph_score": 0.5, "supervised_score": 0.5,
            "anomaly_score": 0.5, "tgn_score": 0.5,
        })


def test_engine_rejects_wrong_keys():
    with pytest.raises(ValueError):
        RiskFusionEngine(weights={"a": 0.5, "b": 0.5})


def test_risk_level_thresholds():
    assert risk_level_for(0.0) == "Low"
    assert risk_level_for(30.0) == "Low"
    assert risk_level_for(30.5) == "Medium"
    assert risk_level_for(60.0) == "Medium"
    assert risk_level_for(60.5) == "High"
    assert risk_level_for(80.0) == "High"
    assert risk_level_for(80.5) == "Critical"
    assert risk_level_for(100.0) == "Critical"


def test_engine_default_is_static_weights():
    eng = RiskFusionEngine()
    assert eng.fusion_mode == "static_weights"
    assert eng.weights == FUSION_WEIGHTS


# --------------------------------------------------------------------------- #
# Fusion arithmetic
# --------------------------------------------------------------------------- #


def _make_scores(**overrides) -> LayerScores:
    base = LayerScores(transaction_id="tx-1", sender_account="A-1")
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_all_zero_scores_yield_low_risk():
    eng = RiskFusionEngine()
    out = eng.fuse(_make_scores())
    assert isinstance(out, FusedRiskOutput)
    assert out.transaction_risk_score == 0.0
    assert out.risk_level == "Low"
    # Group score has only the density / cycle adders (both zero here) → 0
    assert out.group_risk_score == 0.0
    assert out.risk_level_group == "Low"


def test_all_one_hundred_scores_yield_critical():
    eng = RiskFusionEngine()
    out = eng.fuse(_make_scores(
        rule_score=100, graph_score=100, supervised_score=100,
        anomaly_score=100, tgn_score=100,
    ))
    # Weights sum to 1 → exact 100.0
    assert out.transaction_risk_score == pytest.approx(100.0, abs=1e-9)
    assert out.risk_level == "Critical"
    assert out.risk_level_group == "Critical"


def test_weighted_sum_matches_spec_formula():
    """Hand-computed sanity check using the v2 (rule + graph dominant) weights."""
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=10, graph_score=20, supervised_score=80,
        anomaly_score=40, tgn_score=50,
    )
    # v2 weights: rule 0.30, graph 0.25, supervised 0.20, anomaly 0.15, tgn 0.10
    expected = 0.30 * 10 + 0.25 * 20 + 0.20 * 80 + 0.15 * 40 + 0.10 * 50
    out = eng.fuse(s)
    assert out.transaction_risk_score == pytest.approx(expected, abs=1e-9)


def test_score_breakdown_contains_all_layers_and_weights():
    eng = RiskFusionEngine()
    out = eng.fuse(_make_scores(supervised_score=70))
    bd = out.score_breakdown
    assert set(bd["scores"].keys()) == set(FUSION_WEIGHTS.keys())
    assert set(bd["weights"].keys()) == set(FUSION_WEIGHTS.keys())
    assert set(bd["weighted_contributions"].keys()) == set(FUSION_WEIGHTS.keys())
    assert bd["fusion_mode"] == "static_weights"
    # Weighted contribution is score × weight (v2 supervised weight = 0.20)
    assert bd["weighted_contributions"]["supervised_score"] == pytest.approx(70.0 * 0.20, abs=1e-6)


# --------------------------------------------------------------------------- #
# Group risk score
# --------------------------------------------------------------------------- #


def test_group_risk_score_includes_density_and_cycle_bumps():
    eng = RiskFusionEngine()
    s = _make_scores(
        supervised_score=40,  # tx_score → 0.20*40 = 8 (v2 supervised weight = 0.20)
        community_density=1.0,
        has_cycle=True,
        sender_community_risk_score=50,
    )
    out = eng.fuse(s)
    # group = max(tx_score=8, 50) + 0.15*100 + 0.10*100 = 50 + 15 + 10 = 75
    assert out.group_risk_score == pytest.approx(75.0, abs=1e-6)
    assert out.risk_level_group == "High"


def test_group_risk_clipped_to_100():
    eng = RiskFusionEngine()
    s = _make_scores(supervised_score=100, community_density=1.0, has_cycle=True,
                     sender_community_risk_score=90)
    out = eng.fuse(s)
    # tx_score = 0.20*100 = 20; max(20, 90) = 90; + 15 + 10 = 115 → clipped to 100
    assert out.group_risk_score == 100.0


# --------------------------------------------------------------------------- #
# Explanation assembler
# --------------------------------------------------------------------------- #


def test_explanation_leads_with_supervised_when_dominant():
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=10, graph_score=10, anomaly_score=10, tgn_score=10,
        supervised_score=90,
        top_features=["amount_raw", "channel_Web", "ip_change_count_24h"],
        shap_values={"amount_raw": 1.2, "channel_Web": 0.8, "ip_change_count_24h": 0.4},
    )
    out = eng.fuse(s)
    assert "Supervised ML" in out.explanation.split("\n")[0]
    assert "amount_raw" in out.explanation
    # Top SHAP features carried through
    assert out.top_shap_features == ["amount_raw", "channel_Web", "ip_change_count_24h"]


def test_explanation_leads_with_rules_when_dominant():
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=100, graph_score=0, supervised_score=0,
        anomaly_score=0, tgn_score=0,
        triggered_rules=["R01", "R10"],
        rule_explanations=["high amt", "cycle"],
    )
    out = eng.fuse(s)
    assert "Rule engine" in out.explanation.split("\n")[0]
    assert "R01" in out.explanation


def test_explanation_leads_with_anomaly_when_dominant():
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=5, graph_score=5, supervised_score=5,
        anomaly_score=100, tgn_score=5,
        anomaly_drivers=["amount_raw", "fan_in_score", "scatter_gather_score"],
    )
    out = eng.fuse(s)
    first_line = out.explanation.split("\n")[0]
    assert "Behavioural anomaly" in first_line
    assert "amount_raw" in out.explanation


def test_explanation_leads_with_tgn_when_dominant():
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=5, graph_score=5, supervised_score=5,
        anomaly_score=5, tgn_score=100,
        temporal_graph_explanations=["Edge closes a 3-hop cycle"],
    )
    out = eng.fuse(s)
    assert "TGN" in out.explanation.split("\n")[0]


def test_consolidated_patterns_dedupe_and_tag_by_layer():
    eng = RiskFusionEngine()
    s = _make_scores(
        rule_score=70, graph_score=70,
        supervised_score=70, anomaly_score=70, tgn_score=70,
        triggered_rules=["R10", "R10"],   # dup on purpose
        has_cycle=True,
        sender_community_risk_score=80,
        top_features=["amount_raw"],
        anomaly_drivers=["fan_in_score"],
        temporal_graph_explanations=["Edge closes a 3-hop cycle."],
    )
    out = eng.fuse(s)
    # Each layer has its prefix and the duplicate R10 is collapsed
    assert "L1:R10" in out.triggered_patterns
    assert out.triggered_patterns.count("L1:R10") == 1
    assert "L2:community_has_cycle" in out.triggered_patterns
    assert "L3:amount_raw" in out.triggered_patterns
    assert "L4:fan_in_score" in out.triggered_patterns
    assert any(p.startswith("L5:") for p in out.triggered_patterns)


# --------------------------------------------------------------------------- #
# Adapter (collect_layer_scores)
# --------------------------------------------------------------------------- #


class _RuleStub:
    def __init__(self, score, triggered, explanations):
        self.transaction_id = "tx-adapter"
        self.rule_score = score
        self.triggered_rules = triggered
        self.rule_explanations = explanations


class _SupStub:
    def __init__(self, score, top, shap_dict):
        self.transaction_id = "tx-adapter"
        self.supervised_score = score
        self.top_features = top
        self.shap_values = shap_dict
        self.model_agreement = 0.9


class _AnomStub:
    def __init__(self, score, drivers):
        self.transaction_id = "tx-adapter"
        self.anomaly_score = score
        self.anomaly_drivers = drivers


class _TGNStub:
    def __init__(self, score, explanations, mode="tgn"):
        self.transaction_id = "tx-adapter"
        self.tgn_score = score
        self.temporal_graph_explanations = explanations
        self.tgn_mode = mode


class _GraphStub:
    def __init__(self, **kwargs):
        defaults = dict(
            sender_account="A-1",
            sender_community_id=2,
            sender_community_density=0.4,
            sender_community_risk_score=55,
            edge_creates_cycle=False,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_collect_layer_scores_round_trip_through_engine():
    rule = _RuleStub(40, ["R01"], ["high-value"])
    sup = _SupStub(60, ["amount_raw"], {"amount_raw": 0.8})
    anom = _AnomStub(25, ["amount_log"])
    tgn = _TGNStub(35, ["Edge closes a 3-hop cycle"])
    graph = _GraphStub(edge_creates_cycle=True)

    s = collect_layer_scores(
        rule_out=rule, graph_vec=graph, supervised_out=sup,
        anomaly_out=anom, tgn_out=tgn,
        transaction_id="tx-adapter", sender_account="A-1",
    )
    assert s.rule_score == 40
    assert s.graph_score == 55
    assert s.supervised_score == 60
    assert s.anomaly_score == 25
    assert s.tgn_score == 35
    assert s.has_cycle is True

    eng = RiskFusionEngine()
    out = eng.fuse(s)
    # v2 weights: rule 0.30, graph 0.25, supervised 0.20, anomaly 0.15, tgn 0.10
    expected = 0.30 * 40 + 0.25 * 55 + 0.20 * 60 + 0.15 * 25 + 0.10 * 35
    assert out.transaction_risk_score == pytest.approx(expected, abs=1e-6)


def test_meta_learner_mode_requires_artifact(tmp_path):
    """fusion_mode='meta_learner' should raise when artifact is absent."""
    from src.system2_detection.shared.s3_artifact_store import (
        ArtifactNotFoundError,
        ArtifactStore,
    )
    store = ArtifactStore(base_dir=tmp_path / "empty")
    with pytest.raises(ArtifactNotFoundError):
        RiskFusionEngine(fusion_mode="meta_learner", store=store)
