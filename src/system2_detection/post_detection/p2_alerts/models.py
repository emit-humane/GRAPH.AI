"""SQLAlchemy 2.0 schema for the P2 Alert table.

Defaults to a local SQLite file at ``logs/alerts.db`` so the dev pipeline runs
out of the box. Set ``DATABASE_URL`` (e.g. ``postgresql+asyncpg://…``) to point
at Postgres in production. JSON columns hold the structured per-layer payloads
exactly as the v2 build spec requires.
"""

from __future__ import annotations

import enum
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, DateTime, Engine, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_SQLITE_PATH = LOGS_DIR / "alerts.db"


class AlertStatus(str, enum.Enum):
    OPEN = "Open"
    INVESTIGATING = "Investigating"
    SAR_FILED = "SAR_Filed"
    CLOSED = "Closed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Alert(Base):
    """Persisted alert record. Mirrors the spec's Alert schema field for field.

    JSON columns (``triggered_patterns``, ``rule_explanations``,
    ``anomaly_drivers``, ``structural_anomaly_explanations``,
    ``score_breakdown``, ``top_shap_features``) keep the structured payload
    from each layer intact so the dashboard can drill in without re-running
    inference. ``score_breakdown`` MUST carry all five layer scores
    (rule, graph, supervised, anomaly, tgn) plus the matching weights — this
    is the auditable record of how the final number was reached.
    """

    __tablename__ = "alerts"

    alert_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True)
    sender_account: Mapped[str] = mapped_column(String(80), index=True)
    community_id: Mapped[int] = mapped_column(Integer, default=-1)
    transaction_risk_score: Mapped[float] = mapped_column(Float)
    group_risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    risk_level_group: Mapped[str] = mapped_column(String(16), index=True)

    triggered_patterns: Mapped[list[str]] = mapped_column(JSON, default=list)
    rule_explanations: Mapped[list[str]] = mapped_column(JSON, default=list)
    anomaly_drivers: Mapped[list[str]] = mapped_column(JSON, default=list)
    structural_anomaly_explanations: Mapped[list[str]] = mapped_column(JSON, default=list)
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    top_shap_features: Mapped[list[str]] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(String, default="")

    alert_status: Mapped[str] = mapped_column(String(20), default=AlertStatus.OPEN.value, index=True)
    assigned_to: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "transaction_id": self.transaction_id,
            "sender_account": self.sender_account,
            "community_id": int(self.community_id),
            "transaction_risk_score": float(self.transaction_risk_score),
            "group_risk_score": float(self.group_risk_score),
            "risk_level": self.risk_level,
            "risk_level_group": self.risk_level_group,
            "triggered_patterns": list(self.triggered_patterns or []),
            "rule_explanations": list(self.rule_explanations or []),
            "anomaly_drivers": list(self.anomaly_drivers or []),
            "structural_anomaly_explanations": list(self.structural_anomaly_explanations or []),
            "score_breakdown": dict(self.score_breakdown or {}),
            "top_shap_features": list(self.top_shap_features or []),
            "explanation": self.explanation,
            "alert_status": self.alert_status,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# --------------------------------------------------------------------------- #
# Engine / session factory
# --------------------------------------------------------------------------- #


def default_database_url(sqlite_path: Path | None = None) -> str:
    """Return DATABASE_URL if set, else SQLite at logs/alerts.db."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    target = sqlite_path or DEFAULT_SQLITE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{target.as_posix()}"


def create_engine_from_url(url: str | None = None) -> Engine:
    return create_engine(url or default_database_url(), future=True)


def init_db(engine: Engine) -> None:
    """Create the alerts table if it doesn't exist."""
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


__all__ = [
    "Base",
    "Alert",
    "AlertStatus",
    "default_database_url",
    "create_engine_from_url",
    "init_db",
    "make_session_factory",
    "DEFAULT_SQLITE_PATH",
    "LOGS_DIR",
]
