"""FastAPI router for the Scenario Studio.

Three endpoints:
    GET  /studio/catalog                — list typologies + param schemas
    GET  /studio/preview                — plan (no injection, no export)
    POST /studio/inject                 — queue events into D0 stream
    POST /studio/export                 — append scenarios to the export buffer
    POST /studio/export/download        — drain the export buffer to a CSV pair
    GET  /studio/activity               — recent studio activity log
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.system1_generator.scenario_engine import (
    ScenarioEngine,
    default_params,
    typologies_catalog,
    TYPOLOGIES,
)

from .state import AppState

logger = logging.getLogger(__name__)


class PlanRequest(BaseModel):
    typology: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    count: int = 1


class ExportScenario(BaseModel):
    typology: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    count: int = 1


class ExportRequest(BaseModel):
    scenarios: list[ExportScenario] = Field(default_factory=list)


def _log(state: AppState, msg: str, kind: str = "info") -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "msg": msg,
        "kind": kind,
    }
    state.studio_activity_log.insert(0, entry)
    del state.studio_activity_log[60:]


def build_studio_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/studio", tags=["studio"])
    engine = ScenarioEngine()

    @r.get("/catalog")
    def catalog() -> dict[str, Any]:
        return {"typologies": typologies_catalog()}

    @r.get("/preview")
    def preview(
        typology: str = Query(...),
        seed: int = Query(default=42),
        count: int = Query(default=1, ge=1, le=20),
        params: str | None = Query(default=None,
                                   description="JSON-encoded param overrides"),
    ) -> dict[str, Any]:
        if typology not in TYPOLOGIES:
            raise HTTPException(status_code=404, detail=f"unknown typology {typology}")
        p: dict[str, Any] = {}
        if params:
            try:
                p = json.loads(params)
                if not isinstance(p, dict):
                    raise ValueError
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"bad params JSON: {exc}")
        try:
            plans = engine.build_plan(typology, params=p, seed=seed, count=count)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {
            "typology": typology,
            "params": p or default_params(typology),
            "plans": [plan.to_dict() for plan in plans],
        }

    @r.post("/inject")
    def inject(body: PlanRequest) -> dict[str, Any]:
        if not state.generator_status.get("running"):
            # Allowed — but warn so the operator knows nothing will flow yet
            _log(state, f"injected {body.count}× {body.typology} (generator stopped)", "warn")
        try:
            plans = engine.build_plan(body.typology, body.params, body.seed, body.count)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        events = engine.events_from_plans(plans)
        state.studio_injection_queue.extend(events)
        n_edges = sum(len(p.edges) for p in plans)
        total = sum(p.total_value() for p in plans)
        _log(
            state,
            f"Injected {body.count}× {body.typology} "
            f"({n_edges} edges, ₹{total:,.0f} total)",
            "inject",
        )
        return {
            "queued_events": len(events),
            "plans": [plan.to_dict() for plan in plans],
            "queue_depth": len(state.studio_injection_queue),
        }

    @r.post("/export")
    def export_append(body: ExportRequest) -> dict[str, Any]:
        if not body.scenarios:
            raise HTTPException(status_code=400, detail="scenarios list is empty")
        appended = 0
        for sc in body.scenarios:
            try:
                plans = engine.build_plan(sc.typology, sc.params, sc.seed, sc.count)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            events = engine.events_from_plans(plans)
            # Flatten + label per the export schema
            for ev in events:
                state.studio_export_buffer.append({
                    "event": _event_to_csv_row(ev),
                    "ground_truth": {
                        "transaction_id": ev.transaction_id,
                        "suspicious_flag": "Suspicious",
                        "synthetic_pattern_type": sc.typology,
                        "ring_id": plans[0].ring_id,
                    },
                })
                appended += 1
        _log(state, f"Buffered {appended} labelled events ({len(body.scenarios)} scenarios)", "export")
        return {
            "appended": appended,
            "buffer_size": len(state.studio_export_buffer),
        }

    @r.post("/export/download")
    def export_download() -> StreamingResponse:
        if not state.studio_export_buffer:
            raise HTTPException(status_code=400, detail="export buffer is empty")
        # Write a single ZIP with stream_transactions.csv + ground_truth.csv
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            stream_csv = io.StringIO()
            sw = csv.writer(stream_csv)
            sw.writerow(_STREAM_HEADER)
            gt_csv = io.StringIO()
            gw = csv.writer(gt_csv)
            gw.writerow(_GT_HEADER)
            for item in state.studio_export_buffer:
                sw.writerow([item["event"].get(c, "") for c in _STREAM_HEADER])
                gw.writerow([item["ground_truth"].get(c, "") for c in _GT_HEADER])
            zf.writestr("stream_transactions.csv", stream_csv.getvalue())
            zf.writestr("ground_truth.csv", gt_csv.getvalue())
            zf.writestr(
                "manifest.json",
                json.dumps({
                    "row_count": len(state.studio_export_buffer),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "schema": {"stream": _STREAM_HEADER, "ground_truth": _GT_HEADER},
                }, indent=2),
            )
        _log(
            state,
            f"Downloaded export pair: {len(state.studio_export_buffer)} rows",
            "download",
        )
        # Drain the buffer once handed off
        state.studio_export_buffer.clear()
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="studio_export.zip"'},
        )

    @r.delete("/export")
    def clear_export() -> dict[str, Any]:
        n = len(state.studio_export_buffer)
        state.studio_export_buffer.clear()
        _log(state, f"Cleared export buffer ({n} rows)", "info")
        return {"cleared": n}

    @r.get("/activity")
    def activity(limit: int = Query(default=40, ge=1, le=200)) -> dict[str, Any]:
        return {
            "items": state.studio_activity_log[:limit],
            "queue_depth": len(state.studio_injection_queue),
            "export_buffer": len(state.studio_export_buffer),
        }

    return r


# --------------------------------------------------------------------------- #
# CSV schema for the export
# --------------------------------------------------------------------------- #

_STREAM_HEADER = [
    "transaction_id", "timestamp",
    "sender_account", "receiver_account",
    "sender_bank", "receiver_bank",
    "sender_country", "receiver_country",
    "amount", "currency",
    "transaction_type", "payment_channel",
    "device_id", "ip_address",
    "geo_latitude", "geo_longitude",
    "merchant_category", "transaction_status",
    "is_international", "amount_leading_digit",
]

_GT_HEADER = [
    "transaction_id", "suspicious_flag", "synthetic_pattern_type", "ring_id",
]


def _event_to_csv_row(ev) -> dict[str, Any]:
    return {
        "transaction_id": ev.transaction_id,
        "timestamp": ev.timestamp.isoformat() if hasattr(ev.timestamp, "isoformat") else str(ev.timestamp),
        "sender_account": ev.sender_account,
        "receiver_account": ev.receiver_account,
        "sender_bank": ev.sender_bank,
        "receiver_bank": ev.receiver_bank,
        "sender_country": ev.sender_country,
        "receiver_country": ev.receiver_country,
        "amount": float(ev.amount),
        "currency": ev.currency,
        "transaction_type": ev.transaction_type,
        "payment_channel": ev.payment_channel,
        "device_id": ev.device_id,
        "ip_address": ev.ip_address,
        "geo_latitude": ev.geo_latitude,
        "geo_longitude": ev.geo_longitude,
        "merchant_category": ev.merchant_category,
        "transaction_status": ev.transaction_status,
        "is_international": bool(ev.is_international),
        "amount_leading_digit": int(ev.amount_leading_digit),
    }
