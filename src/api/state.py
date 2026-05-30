"""Shared application state — components loaded once at startup.

Loaded by ``main.py`` lifespan. Keeps the heavy artifacts (multigraph,
scaler, supervised models, autoencoder, TGN) in memory for the whole API
lifetime so the per-event pipeline doesn't pay the load cost twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.system2_detection.layer1_rules.rule_engine import RuleEngine
from src.system2_detection.layer3_supervised.d5b_inference import SupervisedInferencer
from src.system2_detection.layer4_anomaly.d6b_inference import BehavioralAnomalyInferencer
from src.system2_detection.layer5_gnn.d7b_inference import TGNInferencer
from src.system2_detection.post_detection.d8_fusion import RiskFusionEngine
from src.system2_detection.post_detection.p2_alerts import AlertManager
from src.system2_detection.shared.d0_replay_driver import ReplayDriver
from src.system2_detection.shared.d1_feature_updater import LiveFeatureUpdater
from src.system2_detection.shared.d2_graph_updater import LiveGraphUpdater
from src.system2_detection.shared.s3_artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _safe_load(store: ArtifactStore, name: str):
    try:
        return store.load(name)
    except ArtifactNotFoundError:
        return None


@dataclass
class AppState:
    """Container the routers/generator pull dependencies from."""

    artifact_store: ArtifactStore = field(default_factory=ArtifactStore)
    driver: ReplayDriver | None = None
    feature_updater: LiveFeatureUpdater | None = None
    graph_updater: LiveGraphUpdater | None = None
    rule_engine: RuleEngine | None = None
    supervised: SupervisedInferencer | None = None
    anomaly: BehavioralAnomalyInferencer | None = None
    tgn: TGNInferencer | None = None
    fusion: RiskFusionEngine | None = None
    alerts: AlertManager | None = None

    # Generator-task related shared resources
    generator_lock = None
    generator_task = None
    generator_status: dict[str, Any] = field(default_factory=lambda: {
        "running": False,
        "events_emitted": 0,
        "alerts_emitted": 0,
        "started_at": None,
        "stopped_at": None,
        "config": {"min_amount": 100, "max_amount": 5_000_000, "interval_seconds": 1.5},
    })

    # In-memory buffers the API exposes to the frontend
    recent_events: list[dict] = field(default_factory=list)        # last N TransactionEvents+scores
    recent_alerts: list[dict] = field(default_factory=list)        # last N persisted Alerts
    subscribers: set = field(default_factory=set)                  # SSE queue.Queue per connection

    # Scenario Studio: queue of injected events the generator drains
    # IN FRONT OF the D0 stream, plus a labelled export buffer.
    studio_injection_queue: list = field(default_factory=list)
    studio_export_buffer: list[dict] = field(default_factory=list)
    studio_activity_log: list[dict] = field(default_factory=list)

    @property
    def driver_count(self) -> int:
        return self.driver.count if self.driver is not None else 0


def build_app_state() -> AppState:
    """Heavy: loads every available offline artifact + sets up all layers."""
    state = AppState()
    store = state.artifact_store

    logger.info("[api] loading driver + accounts ...")
    try:
        state.driver = ReplayDriver()
    except FileNotFoundError as exc:
        logger.warning("[api] stream_transactions.csv missing: %s", exc)

    logger.info("[api] loading offline artifacts ...")
    graph = _safe_load(store, "transaction_multigraph.pkl")
    scaler = _safe_load(store, "feature_scaler.pkl")
    community_profiles = _safe_load(store, "community_profiles.parquet")
    graph_features = _safe_load(store, "graph_features.parquet")

    state.feature_updater = LiveFeatureUpdater(scaler=scaler)
    state.graph_updater = LiveGraphUpdater(
        graph=graph,
        community_profiles=community_profiles,
        graph_features=graph_features,
    )
    state.rule_engine = RuleEngine()

    if SupervisedInferencer.all_artifacts_present(store):
        try:
            state.supervised = SupervisedInferencer(store=store)
        except Exception as exc:
            logger.warning("[api] supervised inferencer failed: %s", exc)
    if BehavioralAnomalyInferencer.all_artifacts_present(store):
        try:
            state.anomaly = BehavioralAnomalyInferencer(store=store)
        except Exception as exc:
            logger.warning("[api] anomaly inferencer failed: %s", exc)
    if TGNInferencer.any_artifacts_present(store):
        try:
            state.tgn = TGNInferencer(store=store)
        except Exception as exc:
            logger.warning("[api] TGN inferencer failed: %s", exc)

    state.fusion = RiskFusionEngine(fusion_mode="static_weights")
    state.alerts = AlertManager()
    return state
