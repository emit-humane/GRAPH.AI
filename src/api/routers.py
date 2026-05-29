"""All HTTP + SSE routers for the GRAPH.AI backend.

Routers in this module:
    * generator     — start / stop / status
    * stream        — SSE /stream/events
    * alerts        — list / detail (UI-shaped wrapping of P2)
    * graph         — /graph/case/{txId} (Cytoscape chain), /graph/subgraph
    * geo           — /geo/case/{txId} (Leaflet)
    * report        — JSON, PDF, ZIP
    * stats         — /stats/evaluation
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import generator as gen_module
from .state import AppState, DATA_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Generator router
# --------------------------------------------------------------------------- #


class GeneratorConfigUpdate(BaseModel):
    min_amount: float | None = None
    max_amount: float | None = None
    interval_seconds: float | None = None


def build_generator_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/generator", tags=["generator"])

    @r.post("/start")
    async def start(config: GeneratorConfigUpdate | None = Body(default=None)) -> dict:
        if config is not None:
            cfg = state.generator_status["config"]
            for k in ("min_amount", "max_amount", "interval_seconds"):
                v = getattr(config, k)
                if v is not None:
                    cfg[k] = float(v)
        return await gen_module.start_generator(state)

    @r.post("/stop")
    async def stop() -> dict:
        return await gen_module.stop_generator(state)

    @r.get("/status")
    def status() -> dict:
        out = dict(state.generator_status)
        out["events_buffered"] = len(state.recent_events)
        out["alerts_buffered"] = len(state.recent_alerts)
        out["driver_event_count"] = state.driver_count
        out["layers_available"] = {
            "supervised": state.supervised is not None,
            "anomaly": state.anomaly is not None,
            "tgn": state.tgn is not None,
            "tgn_mode": getattr(state.tgn, "tgn_mode", None) if state.tgn is not None else None,
        }
        return out

    return r


# --------------------------------------------------------------------------- #
# SSE stream
# --------------------------------------------------------------------------- #


def build_stream_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/stream", tags=["stream"])

    @r.get("/events")
    async def stream_events(request: Request) -> StreamingResponse:
        """Server-Sent Events: each event published by the generator becomes
        a ``data: {…}\\n\\n`` chunk on this stream. Heartbeats every 15s.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        state.subscribers.add(queue)

        async def event_source():
            try:
                # Replay the latest snapshot so a freshly-connected UI sees
                # something immediately.
                for payload in list(state.recent_events[:20])[::-1]:
                    yield f"event: transaction\ndata: {json.dumps(payload, default=str)}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"event: transaction\ndata: {json.dumps(payload, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                state.subscribers.discard(queue)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return r


# --------------------------------------------------------------------------- #
# Alerts router — UI-shaped, wraps the P2 AlertManager
# --------------------------------------------------------------------------- #


def build_alerts_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/alerts", tags=["alerts"])

    def _ui_shape(alert: dict[str, Any]) -> dict[str, Any]:
        """Massage a P2 alert dict into the shape the UI cards expect."""
        bd = alert.get("score_breakdown") or {}
        scores = bd.get("scores", {})
        return {
            "id": alert["alert_id"],
            "alert_id": alert["alert_id"],
            "transaction_id": alert["transaction_id"],
            "sender_account": alert["sender_account"],
            "community_id": alert.get("community_id", -1),
            "transaction_risk_score": alert["transaction_risk_score"],
            "group_risk_score": alert["group_risk_score"],
            "risk_level": alert["risk_level"],
            "risk_level_group": alert.get("risk_level_group", alert["risk_level"]),
            "severity": alert["risk_level"],
            "triggered_patterns": alert.get("triggered_patterns", []),
            "rule_explanations": alert.get("rule_explanations", []),
            "anomaly_drivers": alert.get("anomaly_drivers", []),
            "structural_anomaly_explanations": alert.get("structural_anomaly_explanations", []),
            "top_shap_features": alert.get("top_shap_features", []),
            "explanation": alert.get("explanation", ""),
            "alert_status": alert.get("alert_status", "Open"),
            "assigned_to": alert.get("assigned_to"),
            "created_at": alert.get("created_at"),
            "updated_at": alert.get("updated_at"),
            "score_breakdown": bd,
            "layer_scores": {
                "rule": scores.get("rule_score", 0),
                "graph": scores.get("graph_score", 0),
                "supervised": scores.get("supervised_score", 0),
                "anomaly": scores.get("anomaly_score", 0),
                "tgn": scores.get("tgn_score", 0),
            },
            "layer_weights": bd.get("weights", {}),
            "weighted_contributions": bd.get("weighted_contributions", {}),
        }

    @r.get("")
    def list_alerts(
        risk_level: str | None = Query(default=None),
        alert_status: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=10_000),
    ) -> dict[str, Any]:
        items = state.alerts.list(
            risk_level=risk_level, alert_status=alert_status, limit=limit,
        )
        return {"count": len(items), "items": [_ui_shape(a) for a in items]}

    @r.get("/{alert_id}")
    def get_alert(alert_id: str) -> dict[str, Any]:
        alert = state.alerts.get(alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
        return _ui_shape(alert)

    @r.patch("/{alert_id}/status")
    def patch_status(alert_id: str, body: dict = Body(...)) -> dict[str, Any]:
        try:
            updated = state.alerts.update_status(
                alert_id,
                body.get("alert_status"),
                assigned_to=body.get("assigned_to"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
        return _ui_shape(updated)

    return r


# --------------------------------------------------------------------------- #
# Graph (Cytoscape + subgraph)
# --------------------------------------------------------------------------- #


def build_graph_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/graph", tags=["graph"])

    def _payload_for_tx(tx_id: str) -> dict[str, Any]:
        # Look up the event payload we already buffered
        for pay in state.recent_events:
            if pay.get("transaction_id") == tx_id:
                return pay
        # Otherwise try the alerts table
        try:
            for alert in state.alerts.list(limit=2000):
                if alert.get("transaction_id") == tx_id:
                    bd = alert.get("score_breakdown") or {}
                    return {
                        "transaction_id": tx_id,
                        "sender_account": alert["sender_account"],
                        "receiver_account": "",
                        "score_breakdown": bd,
                        "subgraph_evidence": {},
                    }
        except Exception:
            pass
        raise HTTPException(status_code=404, detail=f"transaction {tx_id} not in recent buffer")

    @r.get("/case/{tx_id}")
    def case_graph(tx_id: str) -> dict[str, Any]:
        """Return a Cytoscape-friendly chain for the case.

        Pulls the implicated 2-hop neighborhood from the live multigraph
        + the two-hop edge list the TGN inferencer captured at scoring time.
        """
        payload = _payload_for_tx(tx_id)
        evidence = payload.get("subgraph_evidence") or {}
        sender = payload.get("sender_account") or ""
        receiver = payload.get("receiver_account") or ""

        # Primary chain — the sender → receiver edge, plus any two-hop neighbours
        neighborhood = list(evidence.get("two_hop_neighborhood") or [])
        edge_samples = list(evidence.get("two_hop_edge_list_sample") or [])

        nodes_set: set[str] = {sender, receiver}
        nodes_set.update(neighborhood[:20])
        nodes_set.discard("")

        nodes = [
            {
                "data": {
                    "id": n,
                    "label": n[:8] if isinstance(n, str) else str(n),
                    "is_focal": n in (sender, receiver),
                    "is_sender": n == sender,
                    "is_receiver": n == receiver,
                },
            }
            for n in nodes_set
        ]

        edges: list[dict] = []
        if sender and receiver:
            edges.append({
                "data": {
                    "id": f"e-focal-{tx_id}",
                    "source": sender, "target": receiver,
                    "amount": payload.get("amount"),
                    "transaction_id": tx_id,
                    "is_focal": True,
                },
            })
        for i, edge in enumerate(edge_samples[:40]):
            try:
                u, v, ts, amt = edge[0], edge[1], edge[2], edge[3]
            except Exception:
                continue
            edges.append({
                "data": {
                    "id": f"e-{i}-{u}-{v}",
                    "source": str(u), "target": str(v),
                    "amount": float(amt),
                    "timestamp": str(ts),
                    "is_focal": False,
                },
            })

        return {
            "transaction_id": tx_id,
            "focal": {"sender": sender, "receiver": receiver},
            "elements": {"nodes": nodes, "edges": edges},
            "metadata": {
                "edge_creates_cycle": bool(evidence.get("edge_creates_cycle", False)),
                "cycle_length": int(evidence.get("cycle_length", 0) or 0),
                "sender_community_id": int(evidence.get("sender_community_id", -1) or -1),
                "receiver_community_id": int(evidence.get("receiver_community_id", -1) or -1),
                "shared_community": bool(evidence.get("shared_community", False)),
            },
        }

    @r.get("/subgraph")
    def subgraph(
        node_id: str = Query(...),
        hops: int = Query(default=2, ge=1, le=3),
    ) -> dict[str, Any]:
        """Return a generic k-hop subgraph around any account from the live D2 graph."""
        if state.graph_updater is None or state.graph_updater.G is None:
            raise HTTPException(status_code=503, detail="multigraph not loaded")
        G = state.graph_updater.G
        if not G.has_node(node_id):
            raise HTTPException(status_code=404, detail=f"node {node_id} not in live multigraph")

        # BFS up to `hops`
        frontier = {node_id}
        visited = {node_id}
        for _ in range(hops):
            nxt: set[str] = set()
            for n in frontier:
                nxt.update(G.successors(n))
                nxt.update(G.predecessors(n))
            new = nxt - visited
            visited.update(new)
            frontier = new
            if len(visited) > 80:
                break

        # Build Cytoscape elements (cap edges to avoid runaway responses)
        nodes = [{"data": {"id": n, "label": n[:10]}} for n in visited]
        edges: list[dict] = []
        for u, v, key, data in G.subgraph(visited).edges(keys=True, data=True):
            edges.append({
                "data": {
                    "id": f"e-{key}", "source": u, "target": v,
                    "amount": float(data.get("amount", 0.0)),
                    "transaction_id": str(key),
                },
            })
            if len(edges) >= 250:
                break

        return {"node_id": node_id, "hops": hops, "elements": {"nodes": nodes, "edges": edges}}

    return r


# --------------------------------------------------------------------------- #
# Geo router
# --------------------------------------------------------------------------- #


COUNTRY_CENTROIDS = {
    "IN": (20.59, 78.96),
    "US": (37.09, -95.71),
    "AE": (23.42, 53.85),
    "SG": (1.35, 103.81),
    "GB": (55.37, -3.43),
    "CN": (35.86, 104.19),
    "MU": (-20.35, 57.55),
    "NG": (9.08, 8.67),
    "PK": (30.37, 69.34),
    "CH": (46.81, 8.22),
}

# A small set of well-known Indian cities — used to project sender/receiver
# accounts onto a Leaflet map when the live event provides a country=IN.
INDIAN_CITY_SPLAY = [
    ("New Delhi", 28.61, 77.21),
    ("Mumbai", 19.08, 72.88),
    ("Bengaluru", 12.97, 77.59),
    ("Chennai", 13.08, 80.27),
    ("Kolkata", 22.57, 88.36),
    ("Hyderabad", 17.38, 78.48),
    ("Pune", 18.52, 73.85),
    ("Ahmedabad", 23.03, 72.58),
    ("Surat", 21.17, 72.83),
    ("Jaipur", 26.92, 75.79),
]


def _city_for_account(account_id: str) -> tuple[str, float, float]:
    h = abs(hash(str(account_id))) % len(INDIAN_CITY_SPLAY)
    name, lat, lng = INDIAN_CITY_SPLAY[h]
    return name, lat, lng


def _country_coord(country: str | None, account_id: str = "") -> tuple[str, float, float]:
    if country and country.upper() == "IN":
        return _city_for_account(account_id)
    if country and country.upper() in COUNTRY_CENTROIDS:
        lat, lng = COUNTRY_CENTROIDS[country.upper()]
        return country.upper(), lat, lng
    # Fallback
    name, lat, lng = INDIAN_CITY_SPLAY[abs(hash(account_id)) % len(INDIAN_CITY_SPLAY)]
    return name, lat, lng


def build_geo_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/geo", tags=["geo"])

    @r.get("/case/{tx_id}")
    def case_geo(tx_id: str) -> dict[str, Any]:
        # Find the buffered event for this transaction
        payload = None
        for ev in state.recent_events:
            if ev.get("transaction_id") == tx_id:
                payload = ev
                break
        if payload is None:
            raise HTTPException(status_code=404, detail=f"transaction {tx_id} not in recent buffer")

        sender = payload["sender_account"]
        receiver = payload["receiver_account"]
        s_country = payload.get("sender_country")
        r_country = payload.get("receiver_country")
        s_city, s_lat, s_lng = _country_coord(s_country, sender)
        r_city, r_lat, r_lng = _country_coord(r_country, receiver)

        # 2-hop neighbourhood from the TGN evidence, projected the same way
        evidence = payload.get("subgraph_evidence") or {}
        neighborhood = list(evidence.get("two_hop_neighborhood") or [])

        # Build nodes
        seen: set[str] = set()
        nodes: list[dict] = []
        def _add(acc: str, lat: float, lng: float, city: str, is_focal: bool):
            if acc in seen or not acc:
                return
            seen.add(acc)
            nodes.append({
                "account_id": acc,
                "city": city,
                "lat": lat, "lng": lng,
                "is_focal": is_focal,
            })

        _add(sender, s_lat, s_lng, s_city, True)
        _add(receiver, r_lat, r_lng, r_city, True)
        for n in neighborhood[:16]:
            city, lat, lng = _city_for_account(n)
            _add(n, lat, lng, city, False)

        edges = [{
            "source": sender, "target": receiver,
            "amount": payload.get("amount"),
            "is_focal": True,
        }]

        return {
            "transaction_id": tx_id,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "active_accounts": len(state.graph_updater.G.nodes()) if state.graph_updater is not None else 0,
                "fraud_accounts": 2,
                "fraud_edges": 1,
            },
        }

    return r


# --------------------------------------------------------------------------- #
# Report router (JSON + PDF + ZIP)
# --------------------------------------------------------------------------- #


def build_report_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/report", tags=["report"])

    def _find_payload(tx_id: str) -> dict[str, Any]:
        for pay in state.recent_events:
            if pay.get("transaction_id") == tx_id:
                return pay
        # Try alerts
        for alert in state.alerts.list(limit=2000):
            if alert.get("transaction_id") == tx_id:
                return alert
        raise HTTPException(status_code=404, detail=f"transaction {tx_id} not found")

    @r.get("/{tx_id}")
    def report_json(tx_id: str) -> dict[str, Any]:
        payload = _find_payload(tx_id)
        return {
            "case_id": f"FRAUD_{tx_id[:10]}",
            "transaction": payload,
            "filing_notes": _filing_notes(payload),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @r.get("/{tx_id}/pdf")
    def report_pdf(tx_id: str) -> StreamingResponse:
        payload = _find_payload(tx_id)
        pdf_bytes = _render_pdf(payload, _filing_notes(payload))
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="report_{tx_id}.pdf"'},
        )

    @r.get("/{tx_id}/zip")
    def report_zip(tx_id: str) -> StreamingResponse:
        payload = _find_payload(tx_id)
        notes = _filing_notes(payload)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"report_{tx_id}.json", json.dumps({
                "case_id": f"FRAUD_{tx_id[:10]}",
                "transaction": payload,
                "filing_notes": notes,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }, default=str, indent=2))
            zf.writestr(f"report_{tx_id}.pdf", _render_pdf(payload, notes))
            zf.writestr(
                "explanation.txt",
                f"Case ID: FRAUD_{tx_id[:10]}\n\n{payload.get('explanation', '')}\n",
            )
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="evidence_{tx_id}.zip"'},
        )

    return r


def _filing_notes(payload: dict[str, Any]) -> str:
    rules = payload.get("triggered_rules") or []
    patterns = payload.get("triggered_patterns") or []
    return (
        f"Transaction {payload.get('transaction_id')} moved INR "
        f"{float(payload.get('amount', 0)):,.0f} from {payload.get('sender_account', '')} "
        f"to {payload.get('receiver_account', '')} on {payload.get('timestamp', '')}. "
        f"Triggered rules: {', '.join(rules) if rules else 'none'}. "
        f"Cross-layer indicators: {', '.join(patterns[:6])}. "
        f"Fused risk score = {float(payload.get('transaction_risk_score', 0)):.1f} "
        f"({payload.get('risk_level', 'Low')}). "
        f"Recommendation: continue tracing downstream beneficiaries and review the "
        f"community-level pattern before SAR escalation."
    )


def _render_pdf(payload: dict[str, Any], notes: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"GRAPH.AI Report {payload.get('transaction_id', '')}",
    )
    styles = getSampleStyleSheet()
    body = []
    body.append(Paragraph("<b>GRAPH.AI — FIU Filing Preview</b>", styles["Title"]))
    body.append(Spacer(1, 4 * mm))
    body.append(Paragraph(
        f"<b>Case ID</b>: FRAUD_{payload.get('transaction_id', '')[:10]}",
        styles["Normal"],
    ))
    body.append(Paragraph(
        f"<b>Transaction</b>: {payload.get('transaction_id', '')}", styles["Normal"]
    ))
    body.append(Paragraph(
        f"<b>Generated</b>: {datetime.now(timezone.utc).isoformat()}", styles["Normal"]
    ))
    body.append(Spacer(1, 4 * mm))

    breakdown = payload.get("score_breakdown") or {}
    scores = breakdown.get("scores", {})
    weights = breakdown.get("weights", {})
    rows = [["Layer", "Score (0-100)", "Weight"]]
    for k in ("rule_score", "graph_score", "supervised_score", "anomaly_score", "tgn_score"):
        rows.append([k, f"{float(scores.get(k, 0)):.2f}", f"{float(weights.get(k, 0)):.2f}"])
    rows.append(["transaction_risk_score", f"{float(payload.get('transaction_risk_score', 0)):.2f}", "—"])
    rows.append(["group_risk_score", f"{float(payload.get('group_risk_score', 0)):.2f}", "—"])
    tbl = Table(rows, colWidths=[55 * mm, 40 * mm, 30 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2940")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    body.append(tbl)
    body.append(Spacer(1, 4 * mm))

    body.append(Paragraph("<b>Triggered patterns</b>", styles["Heading3"]))
    patterns = payload.get("triggered_patterns") or []
    body.append(Paragraph(
        ", ".join(patterns) if patterns else "—", styles["Normal"],
    ))
    body.append(Spacer(1, 3 * mm))

    body.append(Paragraph("<b>Top SHAP features</b>", styles["Heading3"]))
    shap = payload.get("top_shap_features") or []
    body.append(Paragraph(", ".join(shap) if shap else "—", styles["Normal"]))
    body.append(Spacer(1, 4 * mm))

    body.append(Paragraph("<b>Explanation</b>", styles["Heading3"]))
    explanation = payload.get("explanation", "") or "—"
    for line in explanation.split("\n"):
        body.append(Paragraph(line.replace("•", "&bull;"), styles["Normal"]))
    body.append(Spacer(1, 4 * mm))

    body.append(Paragraph("<b>Suggested Filing Notes</b>", styles["Heading3"]))
    body.append(Paragraph(notes, styles["Normal"]))

    doc.build(body)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Stats router
# --------------------------------------------------------------------------- #


def build_stats_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/stats", tags=["stats"])

    @r.get("/evaluation")
    def evaluation_stats() -> dict[str, Any]:
        """Live aggregate over the in-memory event buffer + persisted alerts."""
        events = state.recent_events
        n = len(events)
        risk_distribution: dict[str, int] = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        pattern_counts: dict[str, int] = {}
        rule_counts: dict[str, int] = {}
        for e in events:
            level = e.get("risk_level", "Low")
            risk_distribution[level] = risk_distribution.get(level, 0) + 1
            for r_ in e.get("triggered_rules") or []:
                rule_counts[r_] = rule_counts.get(r_, 0) + 1
            for p in e.get("triggered_patterns") or []:
                pattern_counts[p] = pattern_counts.get(p, 0) + 1

        scores = [float(e.get("transaction_risk_score", 0)) for e in events]
        mean_risk = (sum(scores) / n) if n else 0.0
        return {
            "events_seen": n,
            "alerts_total": state.alerts.count() if state.alerts is not None else 0,
            "mean_risk_score": round(mean_risk, 2),
            "max_risk_score": round(max(scores) if scores else 0.0, 2),
            "risk_distribution": risk_distribution,
            "pattern_counts": dict(sorted(pattern_counts.items(), key=lambda kv: -kv[1])[:20]),
            "rule_counts": dict(sorted(rule_counts.items(), key=lambda kv: -kv[1])),
            "layers_available": {
                "supervised": state.supervised is not None,
                "anomaly": state.anomaly is not None,
                "tgn": state.tgn is not None,
            },
        }

    return r
