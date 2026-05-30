"""System 3 — run E1 → E5 end-to-end.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --stream data/stream_transactions.csv \
                                     --alerts data/generated_alerts.csv \
                                     --ground-truth data/hidden_ground_truth.csv \
                                     --no-db
    python scripts/run_evaluation.py --output-dir artifacts/eval_run_42

Prints the headline minority_class_f1 + pr_auc + per-layer credit table at the
end so a CI run surfaces the numbers without parsing JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system3_evaluation import (  # noqa: E402
    E1Joiner,
    E2TransactionEvaluator,
    E3CommunityEvaluator,
    E4PatternEvaluator,
    E5ReportGenerator,
)
from src.system3_evaluation.e1_joiner import DEFAULT_ALERTS_DB  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run System 3 — Evaluation Engine")
    p.add_argument("--stream", type=Path, default=PROJECT_ROOT / "data" / "stream_transactions.csv")
    p.add_argument("--alerts", type=Path, default=PROJECT_ROOT / "data" / "generated_alerts.csv")
    p.add_argument("--ground-truth", type=Path,
                   default=PROJECT_ROOT / "data" / "hidden_ground_truth.csv")
    p.add_argument("--alerts-db", type=Path, default=DEFAULT_ALERTS_DB,
                   help="Optional logs/alerts.db for per-layer score breakdowns.")
    p.add_argument("--no-db", action="store_true",
                   help="Skip the alerts DB and rely on the CSV only.")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts",
                   help="Where E1/E2/E3/E4/E5 write their JSON/parquet artifacts.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(message)s",
    )


def main() -> int:
    args = _parse_args()
    _setup_logging(args.quiet)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # E1 — join
    e1 = E1Joiner(
        stream_path=args.stream,
        alerts_path=args.alerts,
        ground_truth_path=args.ground_truth,
        alerts_db_path=(None if args.no_db else args.alerts_db),
        output_path=out_dir / "joined_evaluation_frame.parquet",
    )
    frame = e1.run()

    # E2/E3/E4 in parallel-ish (they only need the joined frame)
    tx_metrics = E2TransactionEvaluator(
        frame=frame, output_path=out_dir / "transaction_metrics.json"
    ).run()
    comm_metrics = E3CommunityEvaluator(
        frame=frame, output_path=out_dir / "community_metrics.json"
    ).run()
    pat_metrics = E4PatternEvaluator(
        frame=frame, output_path=out_dir / "pattern_metrics.json"
    ).run()

    # E5 — final report
    report = E5ReportGenerator(
        frame=frame,
        transaction_metrics=tx_metrics,
        community_metrics=comm_metrics,
        pattern_metrics=pat_metrics,
        output_path=out_dir / "evaluation_report.json",
    ).run()

    # ----------------------- headline ---------------------------------- #
    _print_headline(report)
    return 0


def _print_headline(report: dict) -> None:
    tx = report.get("transaction_level_metrics", {})
    bench = report.get("external_benchmarks", {})
    credit = report.get("per_layer_detection_credit", {})
    pat = report.get("pattern_coverage", {})

    print()
    print("=" * 64)
    print("  SYSTEM 3 - EVALUATION HEADLINE")
    print("=" * 64)
    print(f"  Run ID:                 {report.get('run_id')}")
    print(f"  Stream transactions:    {report['dataset_stats']['total_stream_transactions']}")
    print(f"  Suspicious (truth):     {report['dataset_stats']['total_suspicious_ground_truth']}")
    print(f"  Alerts generated:       {report['dataset_stats']['total_alerts_generated']}")
    print("-" * 64)
    print(f"  minority_class_f1       {tx.get('minority_class_f1', 0.0):.4f}")
    print(f"  pr_auc                  {tx.get('pr_auc', 0.0):.4f}")
    print(f"  roc_auc                 {tx.get('roc_auc', 0.0):.4f}")
    print(f"  precision / recall      {tx.get('precision', 0.0):.4f} / {tx.get('recall', 0.0):.4f}")
    print(f"  TP / FP / FN / TN       {tx.get('true_positives', 0)} / "
          f"{tx.get('false_positives', 0)} / {tx.get('false_negatives', 0)} / "
          f"{tx.get('true_negatives', 0)}")
    print("-" * 64)
    print("  per_layer_detection_credit  (share of TPs each layer led)")
    for layer in ("rule", "graph", "supervised", "anomaly", "tgn"):
        share = credit.get(layer, 0.0)
        print(f"    {layer:>11s}:           {share:.4f}")
    print(f"    basis:               {credit.get('credit_basis', 'n/a')}")
    print(f"    total_tp_seen:       {credit.get('total_tp_with_breakdown', 0)}")
    print("-" * 64)
    print("  external_benchmarks (delta = us minus benchmark; +ve = we win)")
    print(f"    your_pr_auc                       {bench.get('your_pr_auc', 0.0):.2f}")
    if "your_supervised_pr_auc" in bench:
        print(f"    your_supervised_pr_auc            {bench['your_supervised_pr_auc']:.2f}")
    print(f"    your_minority_f1                  {bench.get('your_minority_f1', 0.0):.4f}")
    print(f"    delta vs tide_hi_xgboost_pr_auc   {bench.get('vs_tide_hi_delta_pr_auc', 0.0):+.2f}")
    print(f"    delta vs tide_li_lightgbm_pr_auc  {bench.get('vs_tide_li_delta_pr_auc', 0.0):+.2f}")
    print(f"    delta vs amlworld_multi_gnn (F1)  {bench.get('vs_multi_gnn_delta_minority_f1', 0.0):+.4f}")
    print(f"    delta vs amlworld_mega_gnn (F1)   {bench.get('vs_mega_gnn_delta_minority_f1', 0.0):+.4f}")
    print("-" * 64)
    print("  pattern_coverage (top 5 by detection rate)")
    for typology, rate in sorted(pat.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {typology:25s}  {rate:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    sys.exit(main())
