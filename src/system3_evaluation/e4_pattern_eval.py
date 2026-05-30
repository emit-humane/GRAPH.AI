"""E4 — Pattern Coverage Analyzer.

Spec processing (architecture.md §E4):

    Per-pattern detection rate    — groupby synthetic_pattern_type
    Per-rule precision            — for each R## that appears in triggered_patterns
    Latency metrics               — mean / p95 / p99 alert_latency_ms

5-layer update (this build):

    * ``per_rule_precision`` now covers ALL 15 rules including
      ``R14_fan_in`` and ``R15_layering_relay`` (the addendum rules).
      Every key in ``RULE_DISPLAY_NAMES`` is always present in the output —
      a rule that never fires shows precision 0.0 with fired_count 0.

Output ``pattern_metrics.json``:

    {
        "per_pattern_detection_rate": {<typology>: <rate>, ...},
        "per_rule_precision":         {<R##_name>: <precision>, ...},
        "per_rule_fired_count":       {<R##_name>: <int>, ...},
        "per_rule_true_positives":    {<R##_name>: <int>, ...},
        "latency_metrics": {
            "mean_alert_latency_ms": ...,
            "p95_alert_latency_ms":  ...,
            "p99_alert_latency_ms":  ...,
        },
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .e1_joiner import ARTIFACT_DIR

logger = logging.getLogger(__name__)


# The 15 rules — all five layers worth of names. R14 and R15 are the addendum
# rules. Every key here is ALWAYS present in per_rule_precision (even when
# the rule never fires), so the schema is stable for the dashboard.
RULE_DISPLAY_NAMES: dict[str, str] = {
    "R01": "R01_high_value",
    "R02": "R02_structuring",
    "R03": "R03_velocity_spike",
    "R04": "R04_dormant_activation",
    "R05": "R05_impossible_travel",
    "R06": "R06_high_risk_jurisdiction",
    "R07": "R07_device_anomaly",
    "R08": "R08_shared_device",
    "R09": "R09_excessive_beneficiaries",
    "R10": "R10_cycle_closure",
    "R11": "R11_round_amount",
    "R12": "R12_kyc_mismatch",
    "R13": "R13_benford",
    "R14": "R14_fan_in",
    "R15": "R15_layering_relay",
}


# All ten typologies from System 1's scenario set. Like rules, every typology
# is ALWAYS present so a UI dashboard doesn't have to guess.
ALL_TYPOLOGIES: tuple[str, ...] = (
    "structuring", "circular_laundering", "layering_chain",
    "fan_in", "fan_out", "fraud_ring",
    "dormant_activation", "velocity_burst",
    "cross_border_layering", "round_tripping",
)


# Triggered patterns from D8 look like "L1:R02", "L1:R14", "L2:community_has_cycle",
# "L3:amount_raw", "L4:fan_in_score", "L5:Edge closes a 4-hop ..."
_RULE_PATTERN_RE = re.compile(r"^L1:(R\d{2})")


@dataclass
class E4PatternEvaluator:
    """Compute per-pattern, per-rule, and latency metrics."""

    frame: pd.DataFrame
    output_path: Path = ARTIFACT_DIR / "pattern_metrics.json"

    def run(self) -> dict[str, Any]:
        df = self.frame
        per_pattern = self._per_pattern_detection_rate(df)
        per_rule, fired, tps_per_rule = self._per_rule_precision(df)
        latency = self._latency_metrics(df)

        out: dict[str, Any] = {
            "per_pattern_detection_rate": per_pattern,
            "per_rule_precision":         per_rule,
            "per_rule_fired_count":       fired,
            "per_rule_true_positives":    tps_per_rule,
            "latency_metrics":            latency,
        }
        self._persist(out)
        return out

    # ------------------------------ per-pattern ---------------------------- #

    @staticmethod
    def _per_pattern_detection_rate(df: pd.DataFrame) -> dict[str, float]:
        """Detection rate per typology = fraction of ground-truth suspicious
        transactions in that typology that were alerted.
        """
        out: dict[str, float] = {p: 0.0 for p in ALL_TYPOLOGIES}
        truth = df[df["y_true"] == 1].copy()
        if truth.empty:
            return out
        for typology in ALL_TYPOLOGIES:
            sub = truth[truth["synthetic_pattern_type"] == typology]
            if sub.empty:
                continue
            out[typology] = round(float((sub["y_pred"] == 1).mean()), 4)
        # Surface any extra patterns we didn't anticipate (e.g. typology
        # added later) so the report stays honest.
        for typology in truth["synthetic_pattern_type"].dropna().unique():
            if typology in out:
                continue
            sub = truth[truth["synthetic_pattern_type"] == typology]
            out[str(typology)] = round(float((sub["y_pred"] == 1).mean()), 4)
        return out

    # ------------------------------ per-rule precision --------------------- #

    @staticmethod
    def _per_rule_precision(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        """Precision per rule = TP / (TP + FP) over alerts where that rule fired."""
        fired_counts: dict[str, int] = {name: 0 for name in RULE_DISPLAY_NAMES.values()}
        tp_counts: dict[str, int] = {name: 0 for name in RULE_DISPLAY_NAMES.values()}

        for _, row in df.iterrows():
            if row.get("y_pred", 0) != 1:
                continue
            patterns = row.get("triggered_patterns") or []
            if isinstance(patterns, str):
                try:
                    patterns = json.loads(patterns)
                except (TypeError, ValueError):
                    patterns = [patterns]
            fired_ids: set[str] = set()
            for p in patterns:
                m = _RULE_PATTERN_RE.match(str(p))
                if m:
                    fired_ids.add(m.group(1))
            is_tp = int(row.get("y_true", 0)) == 1
            for rid in fired_ids:
                if rid not in RULE_DISPLAY_NAMES:
                    continue
                name = RULE_DISPLAY_NAMES[rid]
                fired_counts[name] += 1
                if is_tp:
                    tp_counts[name] += 1

        precision: dict[str, float] = {}
        for name in RULE_DISPLAY_NAMES.values():
            n = fired_counts[name]
            precision[name] = round(tp_counts[name] / n, 4) if n > 0 else 0.0
        return precision, fired_counts, tp_counts

    # ------------------------------ latency -------------------------------- #

    @staticmethod
    def _latency_metrics(df: pd.DataFrame) -> dict[str, float]:
        latencies = pd.to_numeric(df["alert_latency_ms"], errors="coerce").dropna()
        # Keep only positive (forward-in-time) latencies; pipelines occasionally
        # produce negative skew during replay.
        latencies = latencies[latencies >= 0]
        if latencies.empty:
            return {
                "mean_alert_latency_ms": 0.0,
                "p95_alert_latency_ms":  0.0,
                "p99_alert_latency_ms":  0.0,
                "sample_size":           0,
            }
        return {
            "mean_alert_latency_ms": round(float(latencies.mean()), 2),
            "p95_alert_latency_ms":  round(float(np.percentile(latencies, 95)), 2),
            "p99_alert_latency_ms":  round(float(np.percentile(latencies, 99)), 2),
            "sample_size":           int(len(latencies)),
        }

    def _persist(self, metrics: dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(
            "[E4] typologies=%d rules_observed=%d mean_latency_ms=%.1f",
            sum(1 for v in metrics["per_pattern_detection_rate"].values() if v > 0),
            sum(1 for v in metrics["per_rule_fired_count"].values() if v > 0),
            metrics["latency_metrics"]["mean_alert_latency_ms"],
        )


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def pattern_metrics(frame: pd.DataFrame, *, output_path: Path | None = None) -> dict[str, Any]:
    return E4PatternEvaluator(
        frame=frame,
        output_path=output_path or ARTIFACT_DIR / "pattern_metrics.json",
    ).run()
