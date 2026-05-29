"""Dry-run the D0 → D1 → D2 → L1 → L3 chain on the first N stream events.

Loads any available offline artifacts (transaction_multigraph.pkl,
feature_scaler.pkl, community_profiles.parquet, supervised_*.pkl) and prints a
one-line summary per event with the most operationally useful fields plus the
Layer 1 rule_score + the Layer 3 supervised_score when available.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

from src.system2_detection.layer1_rules.rule_engine import RuleEngine
from src.system2_detection.layer3_supervised.d5b_inference import SupervisedInferencer
from src.system2_detection.shared.d0_replay_driver import ReplayDriver
from src.system2_detection.shared.d1_feature_updater import LiveFeatureUpdater
from src.system2_detection.shared.d2_graph_updater import LiveGraphUpdater
from src.system2_detection.shared.s3_artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("dry-run")


def _safe_load(store: ArtifactStore, name: str):
    try:
        return store.load(name)
    except ArtifactNotFoundError:
        log.info("  %s not present — running without it", name)
        return None


def run(limit: int = 100, warm_days: int = 0) -> None:
    store = ArtifactStore()

    # Load whatever's available
    log.info("[load] artifact store ...")
    graph = _safe_load(store, "transaction_multigraph.pkl")
    scaler = _safe_load(store, "feature_scaler.pkl")
    community_profiles = _safe_load(store, "community_profiles.parquet")
    graph_features = _safe_load(store, "graph_features.parquet")

    log.info(
        "[load] graph nodes=%s edges=%s scaler=%s",
        graph.number_of_nodes() if graph is not None else "n/a",
        graph.number_of_edges() if graph is not None else "n/a",
        "yes" if scaler is not None else "no",
    )

    # Wire components
    driver = ReplayDriver()
    log.info("[D0] stream events available: %d", driver.count)

    fu = LiveFeatureUpdater(scaler=scaler)
    if warm_days > 0:
        log.info("[D1] warm-starting last %d days of history ...", warm_days)
        n = fu.warm_start(DATA_DIR / "historical_transactions.csv", only_last_days=warm_days)
        log.info("[D1] ingested %d historical events", n)

    gu = LiveGraphUpdater(
        graph=graph,
        community_profiles=community_profiles,
        graph_features=graph_features,
    )

    rule_engine = RuleEngine()
    log.info(
        "[L1] %d rules loaded; max_possible_raw=%.2f",
        len(rule_engine.rules),
        rule_engine.max_possible_raw,
    )

    inferencer = None
    if SupervisedInferencer.all_artifacts_present(store):
        try:
            inferencer = SupervisedInferencer(store=store)
            log.info("[L3] supervised inferencer loaded (%d features)", len(inferencer.feature_names))
        except Exception as exc:
            log.warning("[L3] failed to load supervised inferencer: %s", exc)
    else:
        log.info("[L3] supervised artifacts not all present; skipping")

    # Iterate
    log.info("[run] processing first %d events ...\n", limit)
    t0 = time.time()
    rule_counter: Counter[str] = Counter()
    high_rule_count = 0
    sup_score_acc: list[float] = []
    events_iter = driver.events()
    for i, event in enumerate(events_iter):
        if i >= limit:
            break
        feat = fu.process_event(event)
        graph_vec = gu.process_event(event)
        rule_out = rule_engine.evaluate(feat, graph_vec, event)
        rule_counter.update(rule_out.triggered_rules)
        if rule_out.rule_score >= 50:
            high_rule_count += 1

        sup_score = None
        sup_top: list[str] = []
        if inferencer is not None:
            try:
                sup_out = inferencer.score(feat, graph_vec, rule_out)
                sup_score = sup_out.supervised_score
                sup_top = sup_out.top_features[:3]
                sup_score_acc.append(sup_score)
            except Exception as exc:
                log.warning("supervised scoring failed at event %d: %s", i, exc)

        log.info(
            "%03d  %s  %s -> %s  amt=%-10.0f  rule=%5.1f  sup=%s  rules=[%s]  top=[%s]",
            i,
            event.transaction_id[:8],
            event.sender_account[:8],
            event.receiver_account[:8],
            event.amount,
            rule_out.rule_score,
            f"{sup_score:5.1f}" if sup_score is not None else "  n/a",
            ",".join(rule_out.triggered_rules) if rule_out.triggered_rules else "-",
            ",".join(sup_top) if sup_top else "-",
        )
    dt = time.time() - t0
    log.info("\n[done] processed %d events in %.2fs (%.1f ev/s)", limit, dt, limit / max(dt, 1e-9))
    log.info("[stats] rule-score >=50: %d / %d", high_rule_count, limit)
    if sup_score_acc:
        import statistics as _stats
        log.info(
            "[stats] supervised_score mean=%.1f, p95=%.1f, max=%.1f, >=60: %d",
            _stats.mean(sup_score_acc),
            _stats.quantiles(sup_score_acc, n=20)[-1] if len(sup_score_acc) >= 20 else max(sup_score_acc),
            max(sup_score_acc),
            sum(1 for s in sup_score_acc if s >= 60),
        )
    log.info("[stats] rule firing counts:")
    for rule_id, count in sorted(rule_counter.items()):
        log.info("        %s: %d", rule_id, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run the D0/D1/D2 chain")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warm-days", type=int, default=0,
                        help="Warm-start D1 with the last N days of history (0 = skip)")
    args = parser.parse_args()
    run(limit=args.limit, warm_days=args.warm_days)


if __name__ == "__main__":
    main()
