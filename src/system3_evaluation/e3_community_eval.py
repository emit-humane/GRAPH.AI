"""E3 — Community-Level Evaluator.

Spec processing (architecture.md §E3):

    Groups by fraud_ring_id. A ring is "detected" if ≥ 50% of its
    transactions were alerted. Computes community ID overlap between
    suspicious_cluster_id and community_id_alert.

Output ``community_metrics.json``:

    {
        "fraud_ring_detection_rate":   <≥50% alerted>,
        "partial_ring_detection_rate": <any alert>,
        "community_id_match_score":    <mode-overlap between ground-truth
                                       cluster and detector community>,
        "group_risk_precision":        <precision of group_risk_score>=High>,
        "mean_ring_coverage":          <mean fraction-of-ring-alerted>,
        "total_rings":                 ...,
        "detected_rings":              ...,
        "partial_rings":               ...,
    }
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .e1_joiner import ARTIFACT_DIR

logger = logging.getLogger(__name__)


# A ring is "detected" when at least half its transactions were alerted.
RING_DETECTION_THRESHOLD: float = 0.50
# group_risk_score threshold for the group_risk_precision metric.
GROUP_RISK_HIGH: float = 61.0


@dataclass
class E3CommunityEvaluator:
    """Compute community-level (fraud-ring) metrics."""

    frame: pd.DataFrame
    output_path: Path = ARTIFACT_DIR / "community_metrics.json"

    def run(self) -> dict[str, Any]:
        df = self.frame
        result = self._compute(df)
        self._persist(result)
        return result

    # ------------------------------ core ----------------------------------- #

    def _compute(self, df: pd.DataFrame) -> dict[str, Any]:
        rings = df.dropna(subset=["fraud_ring_id"]).copy()
        rings = rings[rings["fraud_ring_id"].astype(str).str.len() > 0]
        if rings.empty:
            return self._empty()

        # Per-ring coverage = fraction of ring transactions that were alerted
        coverage = rings.groupby("fraud_ring_id")["y_pred"].agg(["sum", "count"])
        coverage["fraction"] = coverage["sum"] / coverage["count"].clip(lower=1)
        detected_mask = coverage["fraction"] >= RING_DETECTION_THRESHOLD
        partial_mask = coverage["sum"] > 0

        total_rings = int(len(coverage))
        detected_rings = int(detected_mask.sum())
        partial_rings = int(partial_mask.sum())
        mean_coverage = float(coverage["fraction"].mean())

        community_match = self._community_id_match_score(rings)
        group_risk_precision = self._group_risk_precision(df)

        return {
            "fraud_ring_detection_rate":   round(detected_rings / total_rings, 4) if total_rings else 0.0,
            "partial_ring_detection_rate": round(partial_rings / total_rings, 4) if total_rings else 0.0,
            "community_id_match_score":    round(community_match, 4),
            "group_risk_precision":        round(group_risk_precision, 4),
            "mean_ring_coverage":          round(mean_coverage, 4),
            "total_rings":                 total_rings,
            "detected_rings":              detected_rings,
            "partial_rings":               partial_rings,
        }

    @staticmethod
    def _community_id_match_score(rings: pd.DataFrame) -> float:
        """Per ground-truth cluster, look at the alerted transactions and
        check whether they cluster together under one detector community.
        Score = mean (modal-community-share) across ground-truth clusters.
        """
        sub = rings.dropna(subset=["suspicious_cluster_id"])
        sub = sub[sub["y_pred"] == 1]
        sub = sub.dropna(subset=["community_id_alert"])
        if sub.empty:
            return 0.0
        scores: list[float] = []
        for _, group in sub.groupby("suspicious_cluster_id"):
            community_counts = Counter(int(c) for c in group["community_id_alert"])
            if not community_counts:
                continue
            modal = max(community_counts.values())
            scores.append(modal / len(group))
        if not scores:
            return 0.0
        return float(np.mean(scores))

    @staticmethod
    def _group_risk_precision(df: pd.DataFrame) -> float:
        """Precision of the group_risk_score ≥ 61 (High/Critical) trigger."""
        alerted = df.dropna(subset=["group_risk_score"]).copy()
        if alerted.empty:
            return 0.0
        flagged = alerted[pd.to_numeric(alerted["group_risk_score"], errors="coerce") >= GROUP_RISK_HIGH]
        if flagged.empty:
            return 0.0
        return float((flagged["y_true"] == 1).sum() / len(flagged))

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "fraud_ring_detection_rate":   0.0,
            "partial_ring_detection_rate": 0.0,
            "community_id_match_score":    0.0,
            "group_risk_precision":        0.0,
            "mean_ring_coverage":          0.0,
            "total_rings":                 0,
            "detected_rings":              0,
            "partial_rings":               0,
        }

    def _persist(self, metrics: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(
            "[E3] rings=%d detected=%d partial=%d mean_coverage=%.3f",
            metrics["total_rings"], metrics["detected_rings"],
            metrics["partial_rings"], metrics["mean_ring_coverage"],
        )


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def community_metrics(frame: pd.DataFrame, *, output_path: Path | None = None) -> dict[str, Any]:
    return E3CommunityEvaluator(
        frame=frame,
        output_path=output_path or ARTIFACT_DIR / "community_metrics.json",
    ).run()
