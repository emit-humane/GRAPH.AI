"""System 3 — Evaluation Engine (E1 → E5).

A fully standalone post-mortem evaluator. It compares P2's alert log against
the SEALED ground truth from G4 and emits transaction-, community-, and
pattern-level metrics, plus a single aggregated report card that includes
deltas vs. published AML benchmarks.

The 5-layer architecture additions:

* E4 ``per_rule_precision`` covers all 15 rules including ``R14_fan_in`` and
  ``R15_layering_relay`` (added in the addendum).
* E2 emits ``per_layer_detection_credit`` — for true-positive alerts, which
  of the five layer scores (rule / graph / supervised / anomaly / tgn)
  contributed the most weighted credit. Shows which layers actually pull
  weight in production.
* E5 ``external_benchmarks`` carries the project's own ``your_pr_auc`` and
  ``your_minority_f1`` alongside the published benchmark numbers.
"""

from __future__ import annotations

from .e1_joiner import E1Joiner, joined_evaluation_frame
from .e2_transaction_eval import E2TransactionEvaluator, transaction_metrics
from .e3_community_eval import E3CommunityEvaluator, community_metrics
from .e4_pattern_eval import E4PatternEvaluator, pattern_metrics
from .e5_report import E5ReportGenerator, evaluation_report

__all__ = [
    "E1Joiner",
    "joined_evaluation_frame",
    "E2TransactionEvaluator",
    "transaction_metrics",
    "E3CommunityEvaluator",
    "community_metrics",
    "E4PatternEvaluator",
    "pattern_metrics",
    "E5ReportGenerator",
    "evaluation_report",
]
