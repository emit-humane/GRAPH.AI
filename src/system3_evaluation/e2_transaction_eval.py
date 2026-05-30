"""E2 — Transaction-Level Evaluator.

Spec metrics (architecture.md §E2):

    precision, recall, f1_score_macro, minority_class_f1,
    pr_auc, roc_auc, false_positive_rate, false_negative_rate,
    total_alerts_generated, true_positives, false_positives,
    false_negatives, true_negatives

5-layer addition — ``per_layer_detection_credit``:

    For every TRUE-positive alert, identify which of the five layers
    (rule / graph / supervised / anomaly / tgn) contributed the most
    weighted credit. We report:

        {
            "rule":       n_TP_led_by_rule / n_TP_with_breakdown,
            "graph":      ...,
            "supervised": ...,
            "anomaly":    ...,
            "tgn":        ...,
            "total_tp_with_breakdown": ...,
            "by_layer_mean_score": {...}   # mean raw layer score on TPs
        }

    This shows which layers actually pull weight in production — a TGN
    that's never the leader is a sign we're over-weighting it; a rule layer
    that leads 80% of TPs means we're not leaning enough on ML.

Operates on ``joined_evaluation_frame.parquet`` (the only input E2 needs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .e1_joiner import ARTIFACT_DIR, JOINED_FRAME_COLUMNS, LAYER_KEYS

logger = logging.getLogger(__name__)


# Display name → contribution column name used by per_layer_detection_credit
LAYER_DISPLAY_TO_CONTRIB: dict[str, str] = {
    "rule":       "layer_rule_contribution",
    "graph":      "layer_graph_contribution",
    "supervised": "layer_supervised_contribution",
    "anomaly":    "layer_anomaly_contribution",
    "tgn":        "layer_tgn_contribution",
}

LAYER_DISPLAY_TO_SCORE: dict[str, str] = {
    "rule":       "layer_rule_score",
    "graph":      "layer_graph_score",
    "supervised": "layer_supervised_score",
    "anomaly":    "layer_anomaly_score",
    "tgn":        "layer_tgn_score",
}


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #


@dataclass
class E2TransactionEvaluator:
    """Compute all transaction-level metrics from the joined frame.

    Args:
        frame: Output of ``E1Joiner.run()``.
        output_path: where to write ``transaction_metrics.json``.
    """

    frame: pd.DataFrame
    output_path: Path = ARTIFACT_DIR / "transaction_metrics.json"

    # ------------------------------ public api ----------------------------- #

    def run(self) -> dict[str, Any]:
        df = self.frame
        if df.empty:
            metrics = self._empty_metrics()
            self._persist(metrics)
            return metrics

        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        y_score = pd.to_numeric(df["y_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # Confusion matrix — sklearn returns (TN, FP, FN, TP)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        minority_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        # PR-AUC / ROC-AUC use the continuous score
        pr_auc = self._safe_score(average_precision_score, y_true, y_score)
        roc_auc = self._safe_score(roc_auc_score, y_true, y_score)

        fpr_value = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_value = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        per_layer_credit = self._per_layer_detection_credit(df)
        curves = self._curve_payload(y_true, y_score)

        metrics: dict[str, Any] = {
            "precision":                round(precision, 4),
            "recall":                   round(recall, 4),
            "f1_score_macro":           round(f1_macro, 4),
            "minority_class_f1":        round(minority_f1, 4),
            "pr_auc":                   round(pr_auc, 4),
            "roc_auc":                  round(roc_auc, 4),
            "false_positive_rate":      round(fpr_value, 4),
            "false_negative_rate":      round(fnr_value, 4),
            "total_alerts_generated":   int(y_pred.sum()),
            "true_positives":           int(tp),
            "false_positives":          int(fp),
            "false_negatives":          int(fn),
            "true_negatives":           int(tn),
            "per_layer_detection_credit": per_layer_credit,
            "curves":                   curves,
        }
        self._persist(metrics)
        return metrics

    # ------------------------------ helpers -------------------------------- #

    @staticmethod
    def _safe_score(fn, y_true, y_score) -> float:
        if len(set(y_true)) < 2:
            return 0.0
        try:
            return float(fn(y_true, y_score))
        except (ValueError, RuntimeError):
            return 0.0

    @staticmethod
    def _curve_payload(y_true: np.ndarray, y_score: np.ndarray, n_points: int = 100) -> dict:
        if len(set(y_true)) < 2:
            return {"pr": [], "roc": [], "thresholds": []}
        prec, rec, thresh_pr = precision_recall_curve(y_true, y_score)
        fpr, tpr, thresh_roc = roc_curve(y_true, y_score)
        # Downsample uniformly for compact JSON
        def _down(arr, n):
            if len(arr) <= n:
                return [float(x) for x in arr]
            idx = np.linspace(0, len(arr) - 1, n).astype(int)
            return [float(arr[i]) for i in idx]
        return {
            "pr": list(zip(_down(rec, n_points), _down(prec, n_points))),
            "roc": list(zip(_down(fpr, n_points), _down(tpr, n_points))),
            "pr_auc_curve_points": len(prec),
            "roc_curve_points": len(fpr),
        }

    def _per_layer_detection_credit(self, df: pd.DataFrame) -> dict[str, Any]:
        """For true-positive alerts, count which layer's weighted contribution
        was the highest. Falls back to layer score (without weight) if
        contribution columns are entirely missing.
        """
        tp_mask = (df["y_true"] == 1) & (df["y_pred"] == 1)
        tps = df.loc[tp_mask]
        if tps.empty:
            return {
                "rule": 0.0, "graph": 0.0, "supervised": 0.0,
                "anomaly": 0.0, "tgn": 0.0,
                "total_tp_with_breakdown": 0,
                "by_layer_mean_score": {k: 0.0 for k in LAYER_DISPLAY_TO_SCORE},
                "leader_counts": {k: 0 for k in LAYER_DISPLAY_TO_CONTRIB},
                "credit_basis": "none",
            }

        # Prefer weighted contributions; fall back to raw scores
        contrib_cols = list(LAYER_DISPLAY_TO_CONTRIB.values())
        usable = tps.dropna(subset=contrib_cols, how="all")
        if not usable.empty:
            mat = usable[contrib_cols].fillna(0.0).to_numpy(dtype=float)
            basis = "weighted_contribution"
        else:
            score_cols = list(LAYER_DISPLAY_TO_SCORE.values())
            usable = tps.dropna(subset=score_cols, how="all")
            if usable.empty:
                return {
                    "rule": 0.0, "graph": 0.0, "supervised": 0.0,
                    "anomaly": 0.0, "tgn": 0.0,
                    "total_tp_with_breakdown": 0,
                    "by_layer_mean_score": {k: 0.0 for k in LAYER_DISPLAY_TO_SCORE},
                    "leader_counts": {k: 0 for k in LAYER_DISPLAY_TO_CONTRIB},
                    "credit_basis": "unavailable",
                }
            mat = usable[score_cols].fillna(0.0).to_numpy(dtype=float)
            basis = "layer_score"

        leader_idx = mat.argmax(axis=1)
        display_keys = list(LAYER_DISPLAY_TO_CONTRIB.keys())
        leader_counts = {k: 0 for k in display_keys}
        for i in leader_idx:
            leader_counts[display_keys[i]] += 1
        total = int(len(leader_idx))
        share = {k: round(v / total, 4) if total else 0.0 for k, v in leader_counts.items()}

        # Mean raw score per layer (on TPs that had any score)
        mean_score: dict[str, float] = {}
        for display, col in LAYER_DISPLAY_TO_SCORE.items():
            vals = pd.to_numeric(tps[col], errors="coerce").dropna()
            mean_score[display] = round(float(vals.mean()), 4) if not vals.empty else 0.0

        return {
            **share,
            "total_tp_with_breakdown": total,
            "by_layer_mean_score": mean_score,
            "leader_counts": leader_counts,
            "credit_basis": basis,
        }

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        return {
            "precision": 0.0, "recall": 0.0, "f1_score_macro": 0.0,
            "minority_class_f1": 0.0, "pr_auc": 0.0, "roc_auc": 0.0,
            "false_positive_rate": 0.0, "false_negative_rate": 0.0,
            "total_alerts_generated": 0,
            "true_positives": 0, "false_positives": 0,
            "false_negatives": 0, "true_negatives": 0,
            "per_layer_detection_credit": {
                "rule": 0.0, "graph": 0.0, "supervised": 0.0,
                "anomaly": 0.0, "tgn": 0.0,
                "total_tp_with_breakdown": 0,
                "by_layer_mean_score": {k: 0.0 for k in LAYER_DISPLAY_TO_SCORE},
                "leader_counts": {k: 0 for k in LAYER_DISPLAY_TO_CONTRIB},
                "credit_basis": "none",
            },
            "curves": {"pr": [], "roc": []},
        }

    def _persist(self, metrics: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        logger.info(
            "[E2] minority_f1=%.4f pr_auc=%.4f roc_auc=%.4f tps=%d fps=%d",
            metrics["minority_class_f1"], metrics["pr_auc"], metrics["roc_auc"],
            metrics["true_positives"], metrics["false_positives"],
        )


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def transaction_metrics(
    frame: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Functional wrapper around ``E2TransactionEvaluator.run``."""
    return E2TransactionEvaluator(
        frame=frame,
        output_path=output_path or ARTIFACT_DIR / "transaction_metrics.json",
    ).run()
