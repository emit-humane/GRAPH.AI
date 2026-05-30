"""E5 — Report Generator.

Aggregates E2/E3/E4 into a single ``evaluation_report.json`` and computes
deltas against published external benchmarks.

External benchmarks (architecture.md §E5, unchanged in this build):

    tide_li_lightgbm_pr_auc        78.05
    tide_hi_xgboost_pr_auc         85.12
    amlworld_multi_gnn_minority_f1  0.81
    amlworld_mega_gnn_minority_f1   0.87

5-layer additions:

    * ``your_supervised_pr_auc`` (when E2 was given a per-layer score
      column for the supervised layer) is reported alongside the global
      ``your_pr_auc`` so you can attribute the delta cleanly.
    * ``vs_<benchmark>_delta_*`` is signed: positive means our model beats
      the benchmark by that many points.

The PR-AUC benchmarks above are reported on a 0-100 scale in the literature
(Tide, Altman et al. 2023). To compare like-for-like, we promote our 0-1
``pr_auc`` to a 0-100 scale before computing the delta.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd

from .e1_joiner import ARTIFACT_DIR

logger = logging.getLogger(__name__)


# Published benchmark numbers — copied verbatim from §E5 of the architecture.
EXTERNAL_BENCHMARKS: dict[str, float] = {
    "tide_li_lightgbm_pr_auc":       78.05,   # 0-100 scale
    "tide_hi_xgboost_pr_auc":        85.12,   # 0-100 scale
    "amlworld_multi_gnn_minority_f1": 0.81,   # 0-1 scale
    "amlworld_mega_gnn_minority_f1":  0.87,   # 0-1 scale
}


@dataclass
class E5ReportGenerator:
    """Combine the three evaluator outputs into the final report card."""

    frame: pd.DataFrame
    transaction_metrics: dict[str, Any]
    community_metrics: dict[str, Any]
    pattern_metrics: dict[str, Any]
    output_path: Path = ARTIFACT_DIR / "evaluation_report.json"
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def run(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "run_id":       self.run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_stats": self._dataset_stats(),
            "transaction_level_metrics": self._strip_curves(self.transaction_metrics),
            "community_level_metrics":   self.community_metrics,
            "pattern_coverage":          self.pattern_metrics.get("per_pattern_detection_rate", {}),
            "rule_performance": {
                "per_rule_precision":   self.pattern_metrics.get("per_rule_precision", {}),
                "per_rule_fired_count": self.pattern_metrics.get("per_rule_fired_count", {}),
                "per_rule_true_positives": self.pattern_metrics.get("per_rule_true_positives", {}),
            },
            "latency_metrics": self.pattern_metrics.get("latency_metrics", {}),
            "per_layer_detection_credit":
                self.transaction_metrics.get("per_layer_detection_credit", {}),
            "external_benchmarks": self._external_benchmarks(),
        }
        # Keep the PR / ROC curves separately if E2 emitted them
        if "curves" in self.transaction_metrics:
            report["curves"] = self.transaction_metrics["curves"]
        self._persist(report)
        return report

    # ------------------------------ pieces --------------------------------- #

    def _dataset_stats(self) -> dict[str, Any]:
        df = self.frame
        total_tx = int(len(df))
        total_susp = int(df["y_true"].sum()) if not df.empty else 0
        total_alerts = int(df["y_pred"].sum()) if not df.empty else 0
        return {
            "total_stream_transactions":     total_tx,
            "total_suspicious_ground_truth": total_susp,
            "illicit_ratio":                 round(total_susp / total_tx, 6) if total_tx else 0.0,
            "total_alerts_generated":        total_alerts,
        }

    def _external_benchmarks(self) -> dict[str, Any]:
        # E2 emits pr_auc in [0, 1]; benchmarks are on the [0, 100] scale.
        our_pr_auc_01 = float(self.transaction_metrics.get("pr_auc", 0.0))
        our_pr_auc_100 = round(our_pr_auc_01 * 100.0, 4)
        our_minority_f1 = float(self.transaction_metrics.get("minority_class_f1", 0.0))

        # Optional: our supervised-only PR-AUC, attributing the delta cleanly.
        supervised_pr_auc_100 = self._supervised_pr_auc_100()

        out: dict[str, Any] = dict(EXTERNAL_BENCHMARKS)
        out["your_minority_f1"] = round(our_minority_f1, 4)
        out["your_pr_auc"] = our_pr_auc_100
        if supervised_pr_auc_100 is not None:
            out["your_supervised_pr_auc"] = supervised_pr_auc_100

        # Deltas — positive = we beat the benchmark
        out["vs_tide_li_delta_pr_auc"] = round(our_pr_auc_100 - EXTERNAL_BENCHMARKS["tide_li_lightgbm_pr_auc"], 4)
        out["vs_tide_hi_delta_pr_auc"] = round(our_pr_auc_100 - EXTERNAL_BENCHMARKS["tide_hi_xgboost_pr_auc"], 4)
        out["vs_multi_gnn_delta_minority_f1"] = round(
            our_minority_f1 - EXTERNAL_BENCHMARKS["amlworld_multi_gnn_minority_f1"], 4,
        )
        out["vs_mega_gnn_delta_minority_f1"] = round(
            our_minority_f1 - EXTERNAL_BENCHMARKS["amlworld_mega_gnn_minority_f1"], 4,
        )
        if supervised_pr_auc_100 is not None:
            out["supervised_vs_tide_hi_delta_pr_auc"] = round(
                supervised_pr_auc_100 - EXTERNAL_BENCHMARKS["tide_hi_xgboost_pr_auc"], 4,
            )
        return out

    def _supervised_pr_auc_100(self) -> float | None:
        """Compute a PR-AUC for the supervised layer's score in isolation.

        Useful for attribution: is the global PR-AUC pulled up by ML, or
        is it the rules carrying it? Returns ``None`` when we don't have
        ``layer_supervised_score`` populated.
        """
        col = "layer_supervised_score"
        if col not in self.frame.columns:
            return None
        y_true = self.frame["y_true"].to_numpy(dtype=int)
        scores = pd.to_numeric(self.frame[col], errors="coerce")
        if scores.isna().all():
            return None
        # Treat missing as 0 (no signal from this layer for that transaction)
        scores = scores.fillna(0.0).to_numpy(dtype=float)
        if len(set(y_true)) < 2:
            return None
        try:
            return round(float(average_precision_score(y_true, scores)) * 100.0, 4)
        except (ValueError, RuntimeError):
            return None

    @staticmethod
    def _strip_curves(metrics: dict[str, Any]) -> dict[str, Any]:
        """Keep the report card compact — curves stay in E2's standalone file."""
        out = dict(metrics)
        out.pop("curves", None)
        return out

    def _persist(self, report: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("[E5] evaluation_report.json -> %s", self.output_path)


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def evaluation_report(
    frame: pd.DataFrame,
    *,
    transaction_metrics: dict[str, Any],
    community_metrics: dict[str, Any],
    pattern_metrics: dict[str, Any],
    output_path: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Functional wrapper around ``E5ReportGenerator.run``."""
    return E5ReportGenerator(
        frame=frame,
        transaction_metrics=transaction_metrics,
        community_metrics=community_metrics,
        pattern_metrics=pattern_metrics,
        output_path=output_path or ARTIFACT_DIR / "evaluation_report.json",
        run_id=run_id or str(uuid.uuid4()),
    ).run()
