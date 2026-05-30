"""E1 — Alert / Ground-Truth Joiner.

Merges three CSVs (and, when available, the live alerts DB) into a single
``joined_evaluation_frame`` parquet that every downstream evaluator
consumes:

    stream_transactions.csv   ← G4 (universe of transactions actually replayed)
    generated_alerts.csv      ← P2 (alerts emitted by the detector)
    hidden_ground_truth.csv   ← G4 [SEALED] (per-transaction labels)
    logs/alerts.db (optional) ← P2 (full Alert rows incl. score_breakdown)

Spec processing (architecture.md §E1):

* Left-join all stream transactions against alerts on transaction_id.
* Join the ground-truth labels on transaction_id.
* ``y_true``  = 1 if ``suspicious_flag == "Suspicious"`` else 0
* ``y_pred``  = 1 if ``alert_status ∈ {Open, Investigating, SAR_Filed}`` else 0
* ``y_score`` = ``transaction_risk_score / 100`` (defaults to 0 when no alert)

5-layer additions, attached when available:

* ``layer_*_score`` (rule / graph / supervised / anomaly / tgn) — pulled from
  the alerts DB's ``score_breakdown.scores`` JSON. These columns power E2's
  ``per_layer_detection_credit`` analysis.
* ``layer_*_contribution`` — weighted contribution from
  ``score_breakdown.weighted_contributions``.
* ``alert_latency_ms`` — ``alert.created_at - stream.timestamp`` in
  milliseconds. Powers E4's latency block.

The joiner is forgiving about missing inputs (returns an empty/partial
frame with the expected schema) so the evaluator can be smoke-tested on
synthetic fixtures without a full pipeline run.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_ALERTS_DB = PROJECT_ROOT / "logs" / "alerts.db"

# Alerts whose status counts as a positive detection (architecture.md §E1).
POSITIVE_STATUSES: frozenset[str] = frozenset({"Open", "Investigating", "SAR_Filed"})

# The five layer score columns we pull from score_breakdown.
LAYER_KEYS: tuple[str, ...] = (
    "rule_score",
    "graph_score",
    "supervised_score",
    "anomaly_score",
    "tgn_score",
)


# --------------------------------------------------------------------------- #
# Frame schema — published as a module-level constant so downstream evaluators
# (and tests) can assert column-level contracts.
# --------------------------------------------------------------------------- #


JOINED_FRAME_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "y_true",
    "y_pred",
    "y_score",
    "fraud_ring_id",
    "community_id_alert",
    "suspicious_cluster_id",
    "synthetic_pattern_type",
    "scenario_severity",
    "alert_status",
    "triggered_patterns",
    "transaction_risk_score",
    "group_risk_score",
    "risk_level",
    "alert_created_at",
    "stream_timestamp",
    "alert_latency_ms",
    "sender_account",
    # Per-layer score breakdown (optional — NaN when not available)
    "layer_rule_score",
    "layer_graph_score",
    "layer_supervised_score",
    "layer_anomaly_score",
    "layer_tgn_score",
    "layer_rule_contribution",
    "layer_graph_contribution",
    "layer_supervised_contribution",
    "layer_anomaly_contribution",
    "layer_tgn_contribution",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parse_triggered_patterns(value: Any) -> list[str]:
    """``triggered_patterns`` is JSON-encoded in the CSV but might already
    arrive as a Python list when called from a test fixture.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return [str(value)]
    s = value.strip()
    if not s:
        return []
    # Tolerate both JSON arrays and Python repr lists.
    try:
        parsed = json.loads(s)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return [str(parsed)]


def _load_breakdowns(db_path: Path | None) -> pd.DataFrame:
    """Load ``transaction_id → score_breakdown`` from the alerts SQLite DB,
    if present. Returns an empty frame when the DB is missing or unreadable
    so the join degrades gracefully.
    """
    if db_path is None or not Path(db_path).exists():
        return pd.DataFrame(columns=["transaction_id", "score_breakdown"])
    try:
        from sqlalchemy import create_engine, select

        from src.system2_detection.post_detection.p2_alerts.models import Alert

        engine = create_engine(f"sqlite:///{Path(db_path).as_posix()}", future=True)
        with engine.connect() as conn:
            rows = conn.execute(select(Alert.transaction_id, Alert.score_breakdown)).all()
    except Exception as exc:  # pragma: no cover — never crash a dev evaluation
        logger.warning("could not read score_breakdown from %s: %s", db_path, exc)
        return pd.DataFrame(columns=["transaction_id", "score_breakdown"])

    if not rows:
        return pd.DataFrame(columns=["transaction_id", "score_breakdown"])
    return pd.DataFrame(rows, columns=["transaction_id", "score_breakdown"])


def _extract_layer_columns(breakdown: Any) -> dict[str, float]:
    """Pull ``scores`` + ``weighted_contributions`` for the five layers out
    of a score_breakdown dict. Returns NaN for missing fields so downstream
    analyses can skip them.
    """
    out: dict[str, float] = {f"layer_{k}": float("nan") for k in LAYER_KEYS}
    out.update({f"layer_{k.replace('_score', '_contribution')}": float("nan") for k in LAYER_KEYS})
    if not isinstance(breakdown, dict):
        return out
    scores = breakdown.get("scores") or {}
    contributions = breakdown.get("weighted_contributions") or {}
    for key in LAYER_KEYS:
        if key in scores:
            try:
                out[f"layer_{key}"] = float(scores[key])
            except (TypeError, ValueError):
                pass
        if key in contributions:
            try:
                out[f"layer_{key.replace('_score', '_contribution')}"] = float(contributions[key])
            except (TypeError, ValueError):
                pass
    return out


# --------------------------------------------------------------------------- #
# Joiner
# --------------------------------------------------------------------------- #


@dataclass
class E1Joiner:
    """Build the joined evaluation frame.

    Args:
        stream_path: stream_transactions.csv (transaction universe + timestamps).
        alerts_path: generated_alerts.csv from P2.
        ground_truth_path: hidden_ground_truth.csv (SEALED, from G4).
        alerts_db_path: optional logs/alerts.db. When present, we attach the
            per-layer score_breakdown to every alerted transaction.
        output_path: where to write the joined frame (parquet).
    """

    stream_path: Path = DATA_DIR / "stream_transactions.csv"
    alerts_path: Path = DATA_DIR / "generated_alerts.csv"
    ground_truth_path: Path = DATA_DIR / "hidden_ground_truth.csv"
    alerts_db_path: Path | None = DEFAULT_ALERTS_DB
    output_path: Path = ARTIFACT_DIR / "joined_evaluation_frame.parquet"

    def run(self) -> pd.DataFrame:
        stream = self._load_stream()
        alerts = self._load_alerts()
        ground_truth = self._load_ground_truth()

        breakdowns = _load_breakdowns(self.alerts_db_path)
        if not breakdowns.empty:
            alerts = alerts.merge(breakdowns, on="transaction_id", how="left")
        else:
            alerts["score_breakdown"] = None

        # LEFT join: every stream tx gets a row, alerts/labels fill in when present
        frame = stream.merge(alerts, on="transaction_id", how="left")
        frame = frame.merge(ground_truth, on="transaction_id", how="left")

        # Derive y_true / y_pred / y_score per spec §E1
        frame["y_true"] = (frame["suspicious_flag"].fillna("") == "Suspicious").astype(int)
        frame["y_pred"] = (frame["alert_status"].fillna("").isin(POSITIVE_STATUSES)).astype(int)
        frame["y_score"] = (
            pd.to_numeric(frame["transaction_risk_score"], errors="coerce").fillna(0.0) / 100.0
        ).clip(0.0, 1.0)

        # Latency in ms (alert.created_at - stream.timestamp). NaN when no alert.
        stream_ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        alert_ts = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
        latency_ms = (alert_ts - stream_ts).dt.total_seconds() * 1000.0
        frame["alert_latency_ms"] = latency_ms

        # Expand triggered_patterns into a clean python list
        if "triggered_patterns" in frame.columns:
            frame["triggered_patterns"] = frame["triggered_patterns"].apply(_parse_triggered_patterns)
        else:
            frame["triggered_patterns"] = [[] for _ in range(len(frame))]

        # Extract per-layer scores + contributions from score_breakdown
        layer_rows = frame["score_breakdown"].apply(_extract_layer_columns).tolist()
        layer_df = pd.DataFrame(layer_rows, index=frame.index)
        for col in layer_df.columns:
            frame[col] = layer_df[col]

        # Rename a few columns to the spec's joined-frame names
        frame = frame.rename(columns={
            "community_id": "community_id_alert",
            "created_at": "alert_created_at",
            "timestamp": "stream_timestamp",
        })

        # Make sure the spec's column set is present (fill missing with NaN/empty)
        for col in JOINED_FRAME_COLUMNS:
            if col not in frame.columns:
                frame[col] = pd.NA

        # Reduce to the published schema
        frame = frame[list(JOINED_FRAME_COLUMNS)].copy()

        # Persist
        self._persist(frame)
        logger.info(
            "[E1] joined frame: rows=%d alerts=%d suspicious=%d",
            len(frame),
            int(frame["y_pred"].sum()),
            int(frame["y_true"].sum()),
        )
        return frame

    # ------------------------------ loaders -------------------------------- #

    def _load_stream(self) -> pd.DataFrame:
        if not Path(self.stream_path).exists():
            raise FileNotFoundError(f"stream_transactions.csv not found at {self.stream_path}")
        df = pd.read_csv(self.stream_path, usecols=["transaction_id", "timestamp", "sender_account"])
        return df.drop_duplicates("transaction_id")

    def _load_alerts(self) -> pd.DataFrame:
        if not Path(self.alerts_path).exists():
            logger.warning("[E1] alerts CSV missing — proceeding with zero alerts")
            return pd.DataFrame(columns=[
                "transaction_id", "transaction_risk_score", "group_risk_score",
                "risk_level", "alert_status", "triggered_patterns", "created_at",
                "community_id",
            ])
        df = pd.read_csv(self.alerts_path)
        # The CSV may legitimately not include sender_account — drop our duplicate
        if "sender_account" in df.columns:
            df = df.drop(columns=["sender_account"])
        return df

    def _load_ground_truth(self) -> pd.DataFrame:
        if not Path(self.ground_truth_path).exists():
            raise FileNotFoundError(
                f"hidden_ground_truth.csv not found at {self.ground_truth_path}"
            )
        # The C parser's chunked ``usecols`` path has an off-by-one bug on
        # large multi-column files in some pandas releases; reading the full
        # frame and projecting is robust on every version we care about.
        df = pd.read_csv(self.ground_truth_path, low_memory=False)
        wanted = [
            "transaction_id", "suspicious_flag", "fraud_ring_id",
            "suspicious_cluster_id", "synthetic_pattern_type", "scenario_severity",
        ]
        present = [c for c in wanted if c in df.columns]
        df = df.loc[:, present]
        return df.drop_duplicates("transaction_id")

    # ------------------------------ persistence ---------------------------- #

    def _persist(self, frame: pd.DataFrame) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Pyarrow can't serialise plain python lists in some pandas versions —
        # cast to JSON strings for the parquet write, then keep a list version
        # in-memory for the caller.
        out = frame.copy()
        out["triggered_patterns"] = out["triggered_patterns"].apply(json.dumps)
        try:
            out.to_parquet(self.output_path, index=False)
        except (ImportError, ValueError) as exc:  # pyarrow missing or schema issue
            logger.warning("[E1] parquet write failed (%s) — falling back to CSV", exc)
            out.to_csv(self.output_path.with_suffix(".csv"), index=False)


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #


def joined_evaluation_frame(
    *,
    stream_path: Path | None = None,
    alerts_path: Path | None = None,
    ground_truth_path: Path | None = None,
    alerts_db_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Functional wrapper around ``E1Joiner.run``."""
    joiner = E1Joiner(
        stream_path=stream_path or DATA_DIR / "stream_transactions.csv",
        alerts_path=alerts_path or DATA_DIR / "generated_alerts.csv",
        ground_truth_path=ground_truth_path or DATA_DIR / "hidden_ground_truth.csv",
        alerts_db_path=alerts_db_path if alerts_db_path is not None else DEFAULT_ALERTS_DB,
        output_path=output_path or ARTIFACT_DIR / "joined_evaluation_frame.parquet",
    )
    return joiner.run()
