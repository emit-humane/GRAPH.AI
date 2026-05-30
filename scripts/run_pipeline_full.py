"""Full online inference over the entire stream — production sweep.

The 5-layer runtime order, per transaction:

    Transaction (D0 replay)
      |  D1 Feature Update
      |  D2 Graph Update
      |  D3 Rule Engine            (Layer 1)
      |  D4 Graph Analytics        (Layer 2)
      |  D5 Supervised ML          (Layer 3)
      |  D6 Behavioural Anomaly    (Layer 4)
      |  D7 TGN / GNN              (Layer 5)
      |  D8 Risk Fusion + Explanation Assembly
      v  P2 Alert (if High/Critical) -> generated_alerts.csv

This is the runtime-order version of ``run_pipeline_dry.py``. It loads every
offline artifact, REQUIRES every layer to be present (raise loud otherwise so
a CI run doesn't silently degrade), then sweeps the full stream and exports
``data/generated_alerts.csv``.

Use ``--limit N`` only for smoke tests; the default sweeps the full stream.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.system2_detection.layer1_rules.rule_engine import RuleEngine            # noqa: E402
from src.system2_detection.layer3_supervised.d5b_inference import SupervisedInferencer  # noqa: E402
from src.system2_detection.layer4_anomaly.d6b_inference import BehavioralAnomalyInferencer  # noqa: E402
from src.system2_detection.layer5_gnn.d7b_inference import TGNInferencer         # noqa: E402
from src.system2_detection.post_detection.d8_fusion import (                      # noqa: E402
    RiskFusionEngine,
    collect_layer_scores,
)
from src.system2_detection.post_detection.p2_alerts import AlertManager           # noqa: E402
from src.system2_detection.shared.d0_replay_driver import ReplayDriver            # noqa: E402
from src.system2_detection.shared.d1_feature_updater import LiveFeatureUpdater    # noqa: E402
from src.system2_detection.shared.d2_graph_updater import LiveGraphUpdater        # noqa: E402
from src.system2_detection.shared.s3_artifact_store import ArtifactStore          # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"

log = logging.getLogger("run_pipeline_full")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline_full",
        description="Run the 5-layer detection pipeline over the full stream.",
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Cap on stream events processed. 0 = whole stream.")
    p.add_argument("--warm-days", type=int, default=0,
                   help="Warm-start D1 with the last N days of history (0 = skip).")
    p.add_argument("--clear-alerts", action="store_true",
                   help="Delete alerts before the run (recommended for cold pipeline).")
    p.add_argument("--strict-layers", action="store_true",
                   help="Fail if any layer artifact is missing instead of degrading.")
    p.add_argument("--alerts-csv", type=Path, default=DATA_DIR / "generated_alerts.csv",
                   help="Where to export the alerts CSV after the run.")
    p.add_argument("--progress-every", type=int, default=2_000,
                   help="Print a progress line every N events.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    store = ArtifactStore()

    # ------------------------------------------------------------------ #
    # Layer 0 — load every offline artifact
    # ------------------------------------------------------------------ #
    log.info("[load] artifact store ...")
    graph = store.load("transaction_multigraph.pkl")
    scaler = store.load("feature_scaler.pkl")
    community_profiles = store.load("community_profiles.parquet")
    graph_features = store.load("graph_features.parquet")
    log.info("[load] graph nodes=%d edges=%d", graph.number_of_nodes(), graph.number_of_edges())

    # ------------------------------------------------------------------ #
    # Wire components
    # ------------------------------------------------------------------ #
    driver = ReplayDriver()
    log.info("[D0] stream events available: %d", driver.count)

    fu = LiveFeatureUpdater(scaler=scaler)
    if args.warm_days > 0:
        log.info("[D1] warm-starting last %d days of history ...", args.warm_days)
        n_warm = fu.warm_start(DATA_DIR / "historical_transactions.csv",
                               only_last_days=args.warm_days)
        log.info("[D1] ingested %d historical events", n_warm)

    gu = LiveGraphUpdater(
        graph=graph, community_profiles=community_profiles, graph_features=graph_features,
    )

    rule_engine = RuleEngine()
    log.info("[D3] %d rules loaded (max_possible_raw=%.2f)",
             len(rule_engine.rules), rule_engine.max_possible_raw)

    sup_inf = _load_layer("supervised", SupervisedInferencer, store, args.strict_layers)
    anom_inf = _load_layer("anomaly",    BehavioralAnomalyInferencer, store, args.strict_layers)
    tgn_inf = _load_layer("tgn",         TGNInferencer, store, args.strict_layers)

    fusion = RiskFusionEngine(fusion_mode="static_weights")
    log.info("[D8] fusion weights=%s", fusion.weights)

    alerts = AlertManager()
    if args.clear_alerts:
        removed = alerts.clear()
        log.info("[P2] cleared %d existing alerts", removed)
    log.info("[P2] alert manager ready (db=%s)", alerts.engine.url)

    # ------------------------------------------------------------------ #
    # Stream sweep
    # ------------------------------------------------------------------ #
    limit = args.limit if args.limit > 0 else driver.count
    log.info("[run] sweeping %d events (warm_days=%d, limit=%s) ...",
             limit, args.warm_days, "all" if args.limit == 0 else args.limit)

    t0 = time.time()
    rule_counter: Counter[str] = Counter()
    level_counter: Counter[str] = Counter()
    score_sum = {"rule": 0.0, "graph": 0.0, "supervised": 0.0, "anomaly": 0.0, "tgn": 0.0}
    n_alerts = 0
    n_processed = 0

    for i, event in enumerate(driver.events()):
        if i >= limit:
            break

        feat = fu.process_event(event)
        graph_vec = gu.process_event(event)
        rule_out = rule_engine.evaluate(feat, graph_vec, event)
        rule_counter.update(rule_out.triggered_rules)

        sup_out = _safe_score(sup_inf, "supervised", lambda: sup_inf.score(feat, graph_vec, rule_out))
        anom_out = _safe_score(anom_inf, "anomaly", lambda: anom_inf.score(feat, graph_vec))
        tgn_out = _safe_score(tgn_inf, "tgn", lambda: tgn_inf.score(feat, graph_vec, event))

        layer_scores = collect_layer_scores(
            rule_out=rule_out, graph_vec=graph_vec,
            supervised_out=sup_out, anomaly_out=anom_out, tgn_out=tgn_out,
            transaction_id=event.transaction_id, sender_account=event.sender_account,
        )
        fused = fusion.fuse(layer_scores)
        level_counter[fused.risk_level] += 1

        # Accumulate per-layer score totals for the headline summary
        score_sum["rule"] += layer_scores.rule_score
        score_sum["graph"] += layer_scores.graph_score
        score_sum["supervised"] += layer_scores.supervised_score
        score_sum["anomaly"] += layer_scores.anomaly_score
        score_sum["tgn"] += layer_scores.tgn_score

        alert_obj = alerts.process(
            fused,
            rule_explanations=list(rule_out.rule_explanations),
            anomaly_drivers=list(anom_out.anomaly_drivers) if anom_out is not None else [],
            structural_anomaly_explanations=(
                list(tgn_out.temporal_graph_explanations) if tgn_out is not None else []
            ),
            community_id=int(graph_vec.sender_community_id) if graph_vec is not None else -1,
        )
        if alert_obj is not None:
            n_alerts += 1
        n_processed = i + 1

        if args.progress_every and n_processed % args.progress_every == 0:
            dt = time.time() - t0
            log.info(
                "  ... %d/%d (%.1f ev/s)  alerts=%d  levels=%s",
                n_processed, limit, n_processed / max(dt, 1e-9),
                n_alerts, dict(level_counter),
            )

    dt = time.time() - t0

    # ------------------------------------------------------------------ #
    # Export alerts CSV (P2 -> System 3)
    # ------------------------------------------------------------------ #
    args.alerts_csv.parent.mkdir(parents=True, exist_ok=True)
    alerts.export_csv(args.alerts_csv)

    # ------------------------------------------------------------------ #
    # Headline summary
    # ------------------------------------------------------------------ #
    n = max(n_processed, 1)
    summary = {
        "events_processed": n_processed,
        "elapsed_s": round(dt, 2),
        "events_per_s": round(n_processed / max(dt, 1e-9), 2),
        "alerts_persisted": int(alerts.count()),
        "alerts_new_this_run": n_alerts,
        "alerts_csv": str(args.alerts_csv),
        "level_distribution": dict(level_counter),
        "mean_layer_score": {k: round(v / n, 2) for k, v in score_sum.items()},
        "rule_firing_counts": dict(sorted(rule_counter.items())),
    }
    log.info("")
    log.info("=" * 64)
    log.info("  run_pipeline_full -- DONE")
    log.info("=" * 64)
    log.info("  processed:        %d events in %.1fs (%.1f ev/s)",
             n_processed, dt, n_processed / max(dt, 1e-9))
    log.info("  alerts (this run): %d", n_alerts)
    log.info("  alerts (db total): %d", alerts.count())
    log.info("  level dist:        %s", dict(level_counter))
    log.info("  mean layer score:  rule=%.1f  graph=%.1f  sup=%.1f  anom=%.1f  tgn=%.1f",
             *(summary["mean_layer_score"][k] for k in ("rule", "graph", "supervised", "anomaly", "tgn")))
    log.info("  alerts CSV ->      %s", args.alerts_csv)
    log.info("=" * 64)
    return summary


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_layer(name: str, cls, store: ArtifactStore, strict: bool):
    """Load a layer inferencer, raising loud in strict mode if missing."""
    if hasattr(cls, "all_artifacts_present"):
        present = cls.all_artifacts_present(store)
    elif hasattr(cls, "any_artifacts_present"):
        present = cls.any_artifacts_present(store)
    else:
        present = True
    if not present:
        msg = f"[{name}] inference artifacts not all present"
        if strict:
            raise RuntimeError(msg + " (strict mode)")
        log.warning("%s -- continuing with this layer disabled", msg)
        return None
    try:
        inferencer = cls(store=store)
    except Exception as exc:
        if strict:
            raise
        log.warning("[%s] failed to load inferencer: %s -- disabled", name, exc)
        return None
    log.info("[%s] inferencer loaded", name)
    return inferencer


def _safe_score(inferencer, name: str, call):
    if inferencer is None:
        return None
    try:
        return call()
    except Exception as exc:
        log.debug("[%s] scoring failed: %s", name, exc)
        return None


def main() -> int:
    args = _parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
