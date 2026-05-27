"""Dry-run the D0 → D1 → D2 chain on the first 100 stream events.

Loads any available offline artifacts (transaction_multigraph.pkl,
feature_scaler.pkl, community_profiles.parquet) and prints a one-line summary
per event with the most operationally useful fields from both vectors.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

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

    # Iterate
    log.info("[run] processing first %d events ...\n", limit)
    t0 = time.time()
    events_iter = driver.events()
    for i, event in enumerate(events_iter):
        if i >= limit:
            break
        feat = fu.process_event(event)
        graph_vec = gu.process_event(event)
        log.info(
            "%03d  %s  %s -> %s  amt=%.0f  vel1h=%.0f  zsc=%+.2f  "
            "newDev=%s  cycle=%s(L=%d)  R14u24h=%d  CV=%.2f  "
            "relay=%s  inflow=%.0f  gap=%.0fs",
            i,
            event.transaction_id[:8],
            event.sender_account[:8],
            event.receiver_account[:8],
            event.amount,
            feat.tx_velocity_1h,
            feat.amount_zscore,
            int(feat.new_device_flag),
            int(graph_vec.edge_creates_cycle),
            graph_vec.cycle_length,
            graph_vec.receiver_in_degree_unique_24h,
            graph_vec.receiver_inflow_amount_cv,
            int(graph_vec.sender_is_relay_node),
            graph_vec.sender_last_inflow_amount,
            min(graph_vec.sender_last_inflow_gap_seconds, 99999.0),
        )
    dt = time.time() - t0
    log.info("\n[done] processed %d events in %.2fs (%.1f ev/s)", limit, dt, limit / max(dt, 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run the D0/D1/D2 chain")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warm-days", type=int, default=0,
                        help="Warm-start D1 with the last N days of history (0 = skip)")
    args = parser.parse_args()
    run(limit=args.limit, warm_days=args.warm_days)


if __name__ == "__main__":
    main()
