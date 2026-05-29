"""AlertManager — the high-level facade the rest of the system uses.

Responsibilities:
    * Decide whether a FusedRiskOutput should become an Alert (trigger
      condition: risk_level OR risk_level_group in {High, Critical}).
    * Persist new alerts to the configured DB (SQLite by default, Postgres
      via DATABASE_URL).
    * List / filter / update alert status.
    * Export the spec's ``generated_alerts.csv`` for System 3.

The CSV column set is locked to the spec — see ``CSV_COLUMNS`` below.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Alert,
    AlertStatus,
    create_engine_from_url,
    init_db,
    make_session_factory,
)

logger = logging.getLogger(__name__)


# The spec's generated_alerts.csv column order — preserved verbatim.
CSV_COLUMNS: tuple[str, ...] = (
    "alert_id",
    "transaction_id",
    "sender_account",
    "community_id",
    "transaction_risk_score",
    "group_risk_score",
    "risk_level",
    "triggered_patterns",        # JSON-encoded list per spec
    "alert_status",
    "created_at",
)

TRIGGER_LEVELS: frozenset[str] = frozenset({"High", "Critical"})


# --------------------------------------------------------------------------- #
# AlertManager
# --------------------------------------------------------------------------- #


class AlertManager:
    """Persistence + lifecycle for Alerts.

    Args:
        engine: A pre-built SQLAlchemy Engine. If ``None``, one is created from
            ``database_url`` (or DATABASE_URL env, or the SQLite default).
        database_url: Override the connection URL.
        init: Run ``init_db`` (create tables) at construction. Default True so
            the dev pipeline boots cleanly.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        database_url: str | None = None,
        init: bool = True,
    ) -> None:
        self.engine: Engine = engine or create_engine_from_url(database_url)
        if init:
            init_db(self.engine)
        self._SessionLocal: sessionmaker = make_session_factory(self.engine)

    # ------------------------------ trigger / create ----------------------- #

    @staticmethod
    def should_trigger(fused) -> bool:
        return (
            getattr(fused, "risk_level", "") in TRIGGER_LEVELS
            or getattr(fused, "risk_level_group", "") in TRIGGER_LEVELS
        )

    def process(
        self,
        fused,
        *,
        rule_explanations: list[str] | None = None,
        anomaly_drivers: list[str] | None = None,
        structural_anomaly_explanations: list[str] | None = None,
        community_id: int | None = None,
    ) -> Alert | None:
        """Persist a new alert if the trigger condition is met. Returns the
        Alert row on success, ``None`` when the event doesn't qualify.
        """
        if not self.should_trigger(fused):
            return None

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            transaction_id=fused.transaction_id,
            sender_account=fused.sender_account,
            community_id=int(community_id) if community_id is not None else -1,
            transaction_risk_score=float(fused.transaction_risk_score),
            group_risk_score=float(fused.group_risk_score),
            risk_level=fused.risk_level,
            risk_level_group=fused.risk_level_group,
            triggered_patterns=list(fused.triggered_patterns),
            rule_explanations=list(rule_explanations or []),
            anomaly_drivers=list(anomaly_drivers or []),
            structural_anomaly_explanations=list(structural_anomaly_explanations or []),
            score_breakdown=dict(fused.score_breakdown),
            top_shap_features=list(fused.top_shap_features),
            explanation=fused.explanation,
            alert_status=AlertStatus.OPEN.value,
            assigned_to=None,
        )
        # Sanity: score_breakdown must carry all five layer scores.
        _validate_breakdown(alert.score_breakdown)

        with self._SessionLocal() as session:
            session.add(alert)
            session.commit()
            session.refresh(alert)
        return alert

    # ------------------------------ queries -------------------------------- #

    def list(
        self,
        *,
        risk_level: str | None = None,
        alert_status: str | None = None,
        sender_account: str | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]:
        stmt = select(Alert).order_by(Alert.created_at.desc())
        if risk_level:
            stmt = stmt.where(Alert.risk_level == risk_level)
        if alert_status:
            stmt = stmt.where(Alert.alert_status == alert_status)
        if sender_account:
            stmt = stmt.where(Alert.sender_account == sender_account)
        if limit:
            stmt = stmt.limit(int(limit))
        with self._SessionLocal() as session:
            rows = session.scalars(stmt).all()
            return [r.to_dict() for r in rows]

    def get(self, alert_id: str) -> dict[str, Any] | None:
        with self._SessionLocal() as session:
            obj = session.get(Alert, alert_id)
            return obj.to_dict() if obj is not None else None

    def update_status(
        self,
        alert_id: str,
        new_status: str,
        assigned_to: str | None = None,
    ) -> dict[str, Any] | None:
        """Update status + assignee. Returns the updated Alert dict or None."""
        try:
            AlertStatus(new_status)
        except ValueError as exc:
            raise ValueError(
                f"unknown alert_status {new_status!r}; expected one of "
                f"{[s.value for s in AlertStatus]}"
            ) from exc

        with self._SessionLocal() as session:
            obj = session.get(Alert, alert_id)
            if obj is None:
                return None
            obj.alert_status = new_status
            if assigned_to is not None:
                obj.assigned_to = assigned_to
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj.to_dict()

    def count(self) -> int:
        with self._SessionLocal() as session:
            return session.query(Alert).count()

    # ------------------------------ export --------------------------------- #

    def export_csv(self, out_path: Path) -> Path:
        """Dump all alerts to ``out_path`` using the spec's column set."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with self._SessionLocal() as session:
            rows = session.scalars(select(Alert).order_by(Alert.created_at)).all()

        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(list(CSV_COLUMNS))
            for row in rows:
                writer.writerow([
                    row.alert_id,
                    row.transaction_id,
                    row.sender_account,
                    int(row.community_id),
                    f"{float(row.transaction_risk_score):.4f}",
                    f"{float(row.group_risk_score):.4f}",
                    row.risk_level,
                    json.dumps(list(row.triggered_patterns or [])),
                    row.alert_status,
                    (row.created_at.isoformat() if row.created_at else ""),
                ])
        return out_path

    # ------------------------------ admin ---------------------------------- #

    def clear(self) -> int:
        """Delete every alert. Returns the number removed. Intended for tests
        and for re-running the dry pipeline cleanly."""
        with self._SessionLocal() as session:
            n = session.query(Alert).delete()
            session.commit()
        return int(n)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_REQUIRED_BREAKDOWN_SCORES = (
    "rule_score", "graph_score", "supervised_score", "anomaly_score", "tgn_score",
)


def _validate_breakdown(breakdown: dict[str, Any]) -> None:
    scores = breakdown.get("scores", {}) if isinstance(breakdown, dict) else {}
    missing = [k for k in _REQUIRED_BREAKDOWN_SCORES if k not in scores]
    if missing:
        raise ValueError(
            f"Alert.score_breakdown.scores missing required layer keys: {missing}"
        )
