"""FastAPI router for the P2 Alert Management System.

Endpoints:
    GET    /alerts              — paged list with filters (risk_level, status,
                                  sender_account, limit)
    GET    /alerts/{alert_id}   — single alert detail (full JSON dict)
    PATCH  /alerts/{alert_id}/status
                                — update status + optional assignee
    POST   /alerts/export-csv   — write generated_alerts.csv to disk and
                                  report the path / row count
    WS     /alerts/ws           — push live alert dicts as they're created
                                  (subscribers receive each new alert as JSON)

Mount with::

    from fastapi import FastAPI
    from src.system2_detection.post_detection.p2_alerts.api import build_router

    app = FastAPI()
    app.include_router(build_router(alert_manager))
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from .alert_manager import CSV_COLUMNS, AlertManager
from .models import AlertStatus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #


class StatusUpdate(BaseModel):
    alert_status: str
    assigned_to: str | None = None


class ExportRequest(BaseModel):
    out_path: str | None = None


class ExportResponse(BaseModel):
    path: str
    rows: int
    columns: list[str]


# --------------------------------------------------------------------------- #
# WebSocket broadcaster
# --------------------------------------------------------------------------- #


class AlertBroadcaster:
    """Tracks live WebSocket subscribers and fans out new alerts to them."""

    def __init__(self) -> None:
        self._subscribers: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._subscribers.add(ws)

    async def unsubscribe(self, ws: WebSocket) -> None:
        async with self._lock:
            self._subscribers.discard(ws)

    async def publish(self, alert_dict: dict[str, Any]) -> None:
        message = json.dumps(alert_dict, default=str)
        async with self._lock:
            stale: list[WebSocket] = []
            for ws in self._subscribers:
                try:
                    await ws.send_text(message)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._subscribers.discard(ws)

    def subscriber_count(self) -> int:
        return len(self._subscribers)


# --------------------------------------------------------------------------- #
# Router factory
# --------------------------------------------------------------------------- #


def build_router(
    manager: AlertManager,
    *,
    default_export_path: Path | None = None,
    broadcaster: AlertBroadcaster | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/alerts", tags=["alerts"])
    if default_export_path is None:
        from ...post_detection.p2_alerts.models import PROJECT_ROOT
        default_export_path = PROJECT_ROOT / "data" / "generated_alerts.csv"
    broadcaster = broadcaster or AlertBroadcaster()

    # ------------------------------ queries -------------------------------- #

    @router.get("")
    def list_alerts(
        risk_level: str | None = Query(default=None),
        alert_status: str | None = Query(default=None),
        sender_account: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=10_000),
    ) -> dict[str, Any]:
        items = manager.list(
            risk_level=risk_level,
            alert_status=alert_status,
            sender_account=sender_account,
            limit=limit,
        )
        return {"count": len(items), "items": items}

    @router.get("/{alert_id}")
    def get_alert(alert_id: str) -> dict[str, Any]:
        item = manager.get(alert_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
        return item

    # ------------------------------ status update -------------------------- #

    @router.patch("/{alert_id}/status")
    def patch_status(alert_id: str, body: StatusUpdate = Body(...)) -> dict[str, Any]:
        try:
            updated = manager.update_status(
                alert_id, body.alert_status, assigned_to=body.assigned_to
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
        return updated

    # ------------------------------ export --------------------------------- #

    @router.post("/export-csv", response_model=ExportResponse)
    def export_csv(body: ExportRequest = Body(default_factory=ExportRequest)) -> ExportResponse:
        target = Path(body.out_path) if body.out_path else Path(default_export_path)
        written = manager.export_csv(target)
        # Count rows we just wrote (cheap inline)
        with open(written, "r", encoding="utf-8") as fh:
            row_count = max(0, sum(1 for _ in fh) - 1)
        return ExportResponse(path=str(written), rows=row_count, columns=list(CSV_COLUMNS))

    # ------------------------------ status enum ---------------------------- #

    @router.get("/_meta/statuses")
    def list_statuses() -> list[str]:
        return [s.value for s in AlertStatus]

    # ------------------------------ websocket ------------------------------ #

    @router.websocket("/ws")
    async def alerts_ws(ws: WebSocket) -> None:
        await broadcaster.subscribe(ws)
        try:
            while True:
                # We're push-only; treat any inbound message as a heartbeat.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            await broadcaster.unsubscribe(ws)

    router.broadcaster = broadcaster  # type: ignore[attr-defined]
    return router


__all__ = [
    "build_router",
    "AlertBroadcaster",
    "StatusUpdate",
    "ExportRequest",
    "ExportResponse",
]
