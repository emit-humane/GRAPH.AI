"""D8 — Risk Fusion + Explanation Assembler (online).

Combines the five layer outputs into a single ``FusedRiskOutput``:

    transaction_risk_score =
        0.20 * rule_score         +   # Layer 1 — deterministic governance
        0.15 * graph_score        +   # Layer 2 — structural AML intelligence
        0.35 * supervised_score   +   # Layer 3 — PRIMARY calibrated detector
        0.20 * anomaly_score      +   # Layer 4 — unseen behavior deviation
        0.10 * tgn_score              # Layer 5 — advanced specialist intelligence

Why these weights:
    * Supervised ML (0.35) is the highest-precision, calibrated detector. It
      directly models the labels we have. It gets the largest single slice.
    * Rules (0.20) and Anomaly (0.20) tie for second: rules give us
      deterministic, auditable governance (regulators love this); anomaly
      gives us coverage on patterns the supervised model has never seen.
    * Graph (0.15) provides structural / topological AML signal that's
      complementary to the per-event features.
    * TGN (0.10) is advanced specialist intelligence — modest weight until
      independent validation accumulates.
    All five weights sum to 1.0 and that invariant is asserted at construction.

group_risk_score formula (Cheng et al. 2023, slightly adapted):
    group_risk_score = clip(
        max(tx_score, sender_community_risk_score)        # community level
        + 0.15 * 100 * community_density                  # dense → riskier
        + 0.10 * 100 * (1 if has_cycle else 0),           # cycle bumps risk
        0, 100,
    )

risk_level thresholds (unchanged):
    [0, 30]  → Low
    [31, 60] → Medium
    [61, 80] → High
    [81, 100] → Critical

The Explanation Assembler builds one human-readable summary led by the
highest-contributing layer plus a structured score_breakdown for downstream
consumers (dashboard, audit log, evaluator).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..shared.s3_artifact_store import ArtifactNotFoundError, ArtifactStore
from .schemas import FusedRiskOutput

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Weights — single source of truth
# --------------------------------------------------------------------------- #


FUSION_WEIGHTS: dict[str, float] = {
    "rule_score": 0.20,
    "graph_score": 0.15,
    "supervised_score": 0.35,
    "anomaly_score": 0.20,
    "tgn_score": 0.10,
}

WEIGHT_RATIONALES: dict[str, str] = {
    "rule_score": "deterministic governance",
    "graph_score": "structural AML intelligence",
    "supervised_score": "PRIMARY calibrated detector",
    "anomaly_score": "unseen behaviour deviation",
    "tgn_score": "advanced specialist intelligence",
}

LAYER_DISPLAY_NAMES: dict[str, str] = {
    "rule_score": "Rule engine (Layer 1)",
    "graph_score": "Graph analytics (Layer 2)",
    "supervised_score": "Supervised ML (Layer 3)",
    "anomaly_score": "Behavioural anomaly (Layer 4)",
    "tgn_score": "TGN / GNN (Layer 5)",
}


# --------------------------------------------------------------------------- #
# Risk-level thresholds
# --------------------------------------------------------------------------- #


def risk_level_for(score: float) -> str:
    """Map a 0-100 score to the spec's four-tier label."""
    s = float(np.clip(score, 0.0, 100.0))
    if s <= 30:
        return "Low"
    if s <= 60:
        return "Medium"
    if s <= 80:
        return "High"
    return "Critical"


# --------------------------------------------------------------------------- #
# Layer-output dataclass (a thin adapter so callers don't have to import
# every layer's Pydantic class just to call fusion)
# --------------------------------------------------------------------------- #


@dataclass
class LayerScores:
    """Container for the five score values + the structured metadata each
    layer emits, used by the Explanation Assembler.
    """

    transaction_id: str
    sender_account: str

    # Scores in [0, 100]
    rule_score: float = 0.0
    graph_score: float = 0.0
    supervised_score: float = 0.0
    anomaly_score: float = 0.0
    tgn_score: float = 0.0

    # Layer 1
    triggered_rules: list[str] = None  # type: ignore[assignment]
    rule_explanations: list[str] = None  # type: ignore[assignment]

    # Layer 2 (graph)
    community_id: int = -1
    community_density: float = 0.0
    has_cycle: bool = False
    sender_community_risk_score: float = 0.0

    # Layer 3 (supervised)
    top_features: list[str] = None  # type: ignore[assignment]
    shap_values: dict[str, float] = None  # type: ignore[assignment]
    model_agreement: float = 1.0

    # Layer 4 (anomaly)
    anomaly_drivers: list[str] = None  # type: ignore[assignment]

    # Layer 5 (TGN)
    temporal_graph_explanations: list[str] = None  # type: ignore[assignment]
    tgn_mode: str = "tgn"

    def __post_init__(self):
        for attr in (
            "triggered_rules", "rule_explanations",
            "top_features", "anomaly_drivers", "temporal_graph_explanations",
        ):
            if getattr(self, attr) is None:
                setattr(self, attr, [])
        if self.shap_values is None:
            self.shap_values = {}


# --------------------------------------------------------------------------- #
# Risk Fusion Engine
# --------------------------------------------------------------------------- #


class RiskFusionEngine:
    """Combines the five layer outputs into a FusedRiskOutput.

    Args:
        weights: Override the static fusion weights. Must sum to 1.0.
        fusion_mode: ``"static_weights"`` (default) or ``"meta_learner"``.
            ``meta_learner`` mode loads a Phase-2 XGBoost meta-fusion model
            from the artifact store (see scripts/train_meta_fusion.py).
        store: ArtifactStore used to load the meta-learner. Only consulted
            when ``fusion_mode == "meta_learner"``.
    """

    META_ARTIFACT_NAME = "meta_fusion_model.pkl"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        fusion_mode: str = "static_weights",
        store: ArtifactStore | None = None,
    ) -> None:
        self.weights: dict[str, float] = dict(weights) if weights is not None else dict(FUSION_WEIGHTS)
        weight_sum = sum(self.weights.values())
        assert abs(weight_sum - 1.0) < 1e-9, (
            f"Fusion weights must sum to 1.0, got {weight_sum:.6f}: {self.weights}"
        )
        # All five layers must be represented
        expected = set(FUSION_WEIGHTS.keys())
        if set(self.weights.keys()) != expected:
            missing = expected - set(self.weights.keys())
            extra = set(self.weights.keys()) - expected
            raise ValueError(f"weights mismatch — missing={missing} extra={extra}")

        self.fusion_mode = fusion_mode
        self._meta_model = None
        self._meta_features: list[str] = []
        self._meta_explainer = None
        if fusion_mode == "meta_learner":
            self._load_meta_fusion(store or ArtifactStore())
        elif fusion_mode != "static_weights":
            raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    # --------------------------- meta-learner loading --------------------- #

    def _load_meta_fusion(self, store: ArtifactStore) -> None:
        try:
            payload = store.load(self.META_ARTIFACT_NAME)
        except ArtifactNotFoundError as exc:
            raise ArtifactNotFoundError(
                f"fusion_mode='meta_learner' but {self.META_ARTIFACT_NAME} not present "
                "in the artifact store — run scripts/train_meta_fusion.py first."
            ) from exc
        if isinstance(payload, dict):
            self._meta_model = payload.get("model")
            self._meta_features = list(payload.get("feature_names", []))
            self._meta_explainer = payload.get("explainer")
        else:
            raise RuntimeError(f"{self.META_ARTIFACT_NAME} did not contain a model dict")
        if self._meta_model is None:
            raise RuntimeError(f"{self.META_ARTIFACT_NAME} missing 'model' key")

    # --------------------------- public api ------------------------------- #

    def fuse(self, scores: LayerScores, *, context: dict[str, Any] | None = None) -> FusedRiskOutput:
        """Fuse the five layer scores and emit a complete FusedRiskOutput."""
        # 1) Transaction-level risk score
        if self.fusion_mode == "meta_learner":
            tx_score, fusion_label = self._meta_score(scores, context or {})
        else:
            tx_score = self._static_score(scores)
            fusion_label = "static_weights"

        tx_score = float(np.clip(tx_score, 0.0, 100.0))

        # 2) Group-level risk score
        group_score = self._group_risk_score(scores, tx_score)

        # 3) Risk levels
        tx_level = risk_level_for(tx_score)
        group_level = risk_level_for(group_score)

        # 4) Structured score breakdown
        breakdown = self._score_breakdown(scores, tx_score)

        # 5) Consolidated triggered patterns
        triggered_patterns = self._consolidate_patterns(scores)

        # 6) Top SHAP / driver features for the explanation surface
        top_shap_features = list(scores.top_features)[:5]

        # 7) Human-readable explanation led by the dominant layer
        explanation = self._assemble_explanation(scores, tx_score, breakdown, triggered_patterns)

        return FusedRiskOutput(
            transaction_id=scores.transaction_id,
            sender_account=scores.sender_account,
            transaction_risk_score=tx_score,
            group_risk_score=group_score,
            risk_level=tx_level,
            risk_level_group=group_level,
            score_breakdown=breakdown,
            triggered_patterns=triggered_patterns,
            top_shap_features=top_shap_features,
            explanation=explanation,
            fusion_mode=fusion_label,
        )

    # --------------------------- scoring helpers -------------------------- #

    def _static_score(self, s: LayerScores) -> float:
        return (
            self.weights["rule_score"] * s.rule_score
            + self.weights["graph_score"] * s.graph_score
            + self.weights["supervised_score"] * s.supervised_score
            + self.weights["anomaly_score"] * s.anomaly_score
            + self.weights["tgn_score"] * s.tgn_score
        )

    def _meta_score(self, s: LayerScores, context: dict[str, Any]) -> tuple[float, str]:
        # Assemble the meta-learner's expected feature vector
        feat_values = []
        feat_map = {
            "rule_score": s.rule_score,
            "graph_score": s.graph_score,
            "supervised_score": s.supervised_score,
            "anomaly_score": s.anomaly_score,
            "tgn_score": s.tgn_score,
            "rule_count": len(s.triggered_rules),
            "community_density": s.community_density,
            "community_size": context.get("community_size", 0),
            "transaction_type_enc": context.get("transaction_type_enc", 0),
            "customer_type_enc": context.get("customer_type_enc", 0),
            "kyc_level": context.get("kyc_level", 0),
            "sender_embedding_drift": context.get("sender_embedding_drift", 0.0),
            "has_cycle": int(bool(s.has_cycle)),
        }
        for name in self._meta_features:
            feat_values.append(float(feat_map.get(name, 0.0)))
        x = np.array(feat_values, dtype=float).reshape(1, -1)
        try:
            prob = float(self._meta_model.predict_proba(x)[0, 1])
        except Exception:
            # Fall back to raw decision-function-ish output
            prob = float(self._meta_model.predict(x)[0])
        return float(np.clip(prob * 100.0, 0.0, 100.0)), "meta_learner"

    def _group_risk_score(self, s: LayerScores, tx_score: float) -> float:
        """Group-level rollup. Live: we don't have the full community
        materialised, so we use the max(tx_score, community_risk_score) as a
        proxy for max-across-community.
        """
        community_max = max(tx_score, s.sender_community_risk_score)
        bump = 0.15 * 100.0 * float(np.clip(s.community_density, 0.0, 1.0))
        bump += 0.10 * 100.0 * (1.0 if s.has_cycle else 0.0)
        return float(np.clip(community_max + bump, 0.0, 100.0))

    # --------------------------- breakdown / patterns --------------------- #

    def _score_breakdown(self, s: LayerScores, tx_score: float) -> dict[str, Any]:
        """Structured breakdown: per-layer score, weight, weighted contribution,
        and the totals. Suitable for the dashboard tooltip + the audit log.
        """
        scores_dict = {
            "rule_score": s.rule_score,
            "graph_score": s.graph_score,
            "supervised_score": s.supervised_score,
            "anomaly_score": s.anomaly_score,
            "tgn_score": s.tgn_score,
        }
        contributions = {
            name: round(scores_dict[name] * self.weights[name], 4)
            for name in self.weights
        }
        return {
            "scores": {k: round(v, 4) for k, v in scores_dict.items()},
            "weights": dict(self.weights),
            "weight_rationale": dict(WEIGHT_RATIONALES),
            "weighted_contributions": contributions,
            "total": round(tx_score, 4),
            "fusion_mode": self.fusion_mode,
            "model_agreement": round(s.model_agreement, 4),
        }

    @staticmethod
    def _consolidate_patterns(s: LayerScores) -> list[str]:
        patterns: list[str] = []
        # L1: rule ids (compact)
        for rule_id in s.triggered_rules or []:
            patterns.append(f"L1:{rule_id}")
        # L2: graph cycle signal
        if s.has_cycle:
            patterns.append("L2:community_has_cycle")
        if s.sender_community_risk_score >= 60:
            patterns.append("L2:high_community_risk")
        # L3: top supervised driver
        for feat in (s.top_features or [])[:2]:
            patterns.append(f"L3:{feat}")
        # L4: top anomaly driver
        for driver in (s.anomaly_drivers or [])[:2]:
            patterns.append(f"L4:{driver}")
        # L5: temporal graph signals — pull a compact tag from each explanation
        for ex in (s.temporal_graph_explanations or [])[:2]:
            tag = ex.split(":")[0] if ":" in ex else ex.split(".")[0]
            patterns.append(f"L5:{tag[:40].strip()}")
        # Dedupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # --------------------------- explanation assembly --------------------- #

    def _assemble_explanation(
        self,
        s: LayerScores,
        tx_score: float,
        breakdown: dict[str, Any],
        triggered_patterns: list[str],
    ) -> str:
        """Build a single human-readable summary, led by the highest-contributing
        layer. Layer 3 (supervised) is intentionally prioritised when it's the
        leader — it's the primary calibrated detector and its SHAP features
        are the most defensible drivers to present first.
        """
        contribs = breakdown["weighted_contributions"]
        # Determine the leading layer by weighted contribution
        ordered = sorted(contribs.items(), key=lambda kv: -kv[1])
        leader_key, leader_value = ordered[0]
        leader_name = LAYER_DISPLAY_NAMES[leader_key]

        # Lead sentence
        level = risk_level_for(tx_score)
        lead = (
            f"Transaction risk_score={tx_score:.1f} ({level}). "
            f"Led by {leader_name} (contribution {leader_value:.1f})"
        )

        # Layer-specific framing for the leader
        if leader_key == "supervised_score" and s.top_features:
            top = ", ".join(s.top_features[:3])
            lead += f" — top SHAP drivers: {top}"
        elif leader_key == "rule_score" and s.triggered_rules:
            lead += f" — triggered: {','.join(s.triggered_rules[:5])}"
        elif leader_key == "anomaly_score" and s.anomaly_drivers:
            lead += f" — anomaly drivers: {', '.join(s.anomaly_drivers[:3])}"
        elif leader_key == "tgn_score" and s.temporal_graph_explanations:
            lead += f" — structural: {s.temporal_graph_explanations[0][:80]}"
        elif leader_key == "graph_score":
            if s.has_cycle:
                lead += " — community contains a directed cycle"
            if s.sender_community_risk_score >= 60:
                lead += f" — sender community risk {s.sender_community_risk_score:.1f}"
        lead += "."

        # Per-layer bullets, ordered by contribution
        bullets: list[str] = []
        for name, contrib in ordered:
            score_val = breakdown["scores"][name]
            if score_val <= 0.0 and contrib <= 0.0:
                continue
            display = LAYER_DISPLAY_NAMES[name]
            rationale = WEIGHT_RATIONALES[name]
            tail = self._per_layer_tail(name, s)
            bullets.append(
                f"  • {display} (weight {self.weights[name]:.2f}, {rationale}): "
                f"score={score_val:.1f} → +{contrib:.1f}{tail}"
            )

        # Triggered patterns at the end (compact)
        pattern_line = ""
        if triggered_patterns:
            pattern_line = "\nPatterns: " + ", ".join(triggered_patterns[:10])

        body = "\n".join(bullets)
        if body:
            return lead + "\n" + body + pattern_line
        return lead

    @staticmethod
    def _per_layer_tail(name: str, s: LayerScores) -> str:
        if name == "rule_score" and s.triggered_rules:
            return f" [{','.join(s.triggered_rules[:6])}]"
        if name == "graph_score":
            tags = []
            if s.has_cycle:
                tags.append("cycle")
            if s.sender_community_risk_score >= 60:
                tags.append(f"comm_risk={s.sender_community_risk_score:.0f}")
            if tags:
                return " [" + ",".join(tags) + "]"
        if name == "supervised_score" and s.top_features:
            return " [shap:" + ",".join(s.top_features[:3]) + "]"
        if name == "anomaly_score" and s.anomaly_drivers:
            return " [drivers:" + ",".join(s.anomaly_drivers[:3]) + "]"
        if name == "tgn_score" and s.temporal_graph_explanations:
            return " [" + s.temporal_graph_explanations[0][:50].rstrip(".") + "]"
        return ""


# --------------------------------------------------------------------------- #
# Convenience adapter — collapse the existing Pydantic outputs into LayerScores
# --------------------------------------------------------------------------- #


def collect_layer_scores(
    *,
    rule_out=None,
    graph_vec=None,
    supervised_out=None,
    anomaly_out=None,
    tgn_out=None,
    transaction_id: str | None = None,
    sender_account: str | None = None,
) -> LayerScores:
    """Build a LayerScores from whatever subset of layer outputs is available."""
    if transaction_id is None:
        for o in (rule_out, supervised_out, anomaly_out, tgn_out):
            if o is not None and getattr(o, "transaction_id", None):
                transaction_id = o.transaction_id
                break
    if sender_account is None and supervised_out is not None:
        sender_account = supervised_out.transaction_id  # fallback
    if sender_account is None and graph_vec is not None:
        sender_account = graph_vec.sender_account
    if transaction_id is None or sender_account is None:
        raise ValueError("collect_layer_scores needs at least one source for transaction_id + sender_account")

    s = LayerScores(transaction_id=transaction_id, sender_account=sender_account)

    if rule_out is not None:
        s.rule_score = float(rule_out.rule_score)
        s.triggered_rules = list(rule_out.triggered_rules)
        s.rule_explanations = list(rule_out.rule_explanations)
    if graph_vec is not None:
        s.graph_score = float(graph_vec.sender_community_risk_score)
        s.community_id = int(graph_vec.sender_community_id)
        s.community_density = float(graph_vec.sender_community_density)
        # has_cycle: either the event closes a cycle or the community already has one
        s.has_cycle = bool(graph_vec.edge_creates_cycle) or s.community_density > 0.5
        s.sender_community_risk_score = float(graph_vec.sender_community_risk_score)
    if supervised_out is not None:
        s.supervised_score = float(supervised_out.supervised_score)
        s.top_features = list(supervised_out.top_features)
        s.shap_values = dict(supervised_out.shap_values)
        s.model_agreement = float(supervised_out.model_agreement)
    if anomaly_out is not None:
        s.anomaly_score = float(anomaly_out.anomaly_score)
        s.anomaly_drivers = list(anomaly_out.anomaly_drivers)
    if tgn_out is not None:
        s.tgn_score = float(tgn_out.tgn_score)
        s.temporal_graph_explanations = list(tgn_out.temporal_graph_explanations)
        s.tgn_mode = str(tgn_out.tgn_mode)
    return s
