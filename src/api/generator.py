"""Generator background task — runs the FULL 5-layer detection pipeline.

Spawned by ``POST /generator/start``. Each tick:
    1. pulls the next ``TransactionEvent`` from D0 (or wraps around)
    2. runs D1 (feature updater), D2 (graph updater), L1 rules
    3. runs the heavy layers (L3 supervised, L4 anomaly, L5 TGN) in a
       thread-pool so the asyncio loop stays responsive within the
       1-10s cadence the UI expects
    4. fuses via D8 and persists the alert via P2 when triggered
    5. publishes the result to every SSE subscriber queue

When the offline artifacts aren't yet present the layers gracefully drop
out — the dry-run UI still surfaces rule + graph + fused scores so the
operator sees a heartbeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from src.system2_detection.post_detection.d8_fusion import collect_layer_scores

from .state import AppState

logger = logging.getLogger(__name__)

MAX_BUFFER = 200  # max events / alerts retained in memory for the API


# --------------------------------------------------------------------------- #
# Per-event pipeline
# --------------------------------------------------------------------------- #


def _run_pipeline_for_event(state: AppState, event) -> dict[str, Any]:
    """Execute D1 -> D2 -> L1 -> L3 -> L4 -> L5 -> D8 -> P2 for one event."""
    feat = state.feature_updater.process_event(event)
    graph_vec = state.graph_updater.process_event(event)
    rule_out = state.rule_engine.evaluate(feat, graph_vec, event)

    sup_out = None
    if state.supervised is not None:
        try:
            sup_out = state.supervised.score(feat, graph_vec, rule_out)
        except Exception as exc:
            logger.warning("[gen] supervised failed: %s", exc)

    anom_out = None
    if state.anomaly is not None:
        try:
            anom_out = state.anomaly.score(feat, graph_vec)
        except Exception as exc:
            logger.warning("[gen] anomaly failed: %s", exc)

    tgn_out = None
    if state.tgn is not None:
        try:
            tgn_out = state.tgn.score(feat, graph_vec, event)
        except Exception as exc:
            logger.warning("[gen] tgn failed: %s", exc)

    layer_scores = collect_layer_scores(
        rule_out=rule_out,
        graph_vec=graph_vec,
        supervised_out=sup_out,
        anomaly_out=anom_out,
        tgn_out=tgn_out,
        transaction_id=event.transaction_id,
        sender_account=event.sender_account,
    )
    fused = state.fusion.fuse(layer_scores)

    alert_obj = state.alerts.process(
        fused,
        rule_explanations=list(rule_out.rule_explanations),
        anomaly_drivers=list(anom_out.anomaly_drivers) if anom_out is not None else [],
        structural_anomaly_explanations=list(tgn_out.temporal_graph_explanations) if tgn_out is not None else [],
        community_id=int(graph_vec.sender_community_id) if graph_vec is not None else -1,
    )

    payload = {
        "transaction_id": event.transaction_id,
        "sender_account": event.sender_account,
        "receiver_account": event.receiver_account,
        "amount": float(event.amount),
        "currency": event.currency,
        "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        "sender_country": event.sender_country,
        "receiver_country": event.receiver_country,
        "transaction_type": event.transaction_type,
        "rule_score": float(rule_out.rule_score),
        "supervised_score": float(sup_out.supervised_score) if sup_out is not None else None,
        "anomaly_score": float(anom_out.anomaly_score) if anom_out is not None else None,
        "tgn_score": float(tgn_out.tgn_score) if tgn_out is not None else None,
        "transaction_risk_score": float(fused.transaction_risk_score),
        "group_risk_score": float(fused.group_risk_score),
        "risk_level": fused.risk_level,
        "risk_level_group": fused.risk_level_group,
        "triggered_rules": list(rule_out.triggered_rules),
        "triggered_patterns": list(fused.triggered_patterns),
        "top_shap_features": list(fused.top_shap_features),
        "explanation": fused.explanation,
        "score_breakdown": fused.score_breakdown,
        "alert_id": alert_obj.alert_id if alert_obj is not None else None,
        "alerted": alert_obj is not None,
        "tgn_mode": tgn_out.tgn_mode if tgn_out is not None else None,
        "subgraph_evidence": tgn_out.subgraph_evidence if tgn_out is not None else None,
    }
    return payload


# --------------------------------------------------------------------------- #
# Background asyncio loop
# --------------------------------------------------------------------------- #


_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="graphai-pipeline")


async def _publish(state: AppState, payload: dict[str, Any]) -> None:
    """Push a payload to every SSE subscriber queue (non-blocking)."""
    dead: list = []
    for q in list(state.subscribers):
        try:
            q.put_nowait(payload)
        except Exception:
            dead.append(q)
    for q in dead:
        state.subscribers.discard(q)


async def generator_loop(state: AppState) -> None:
    """The actual background coroutine. Reads from D0, runs the pipeline,
    publishes to subscribers, and respects min/max amount filtering."""
    logger.info("[gen] generator loop starting")
    state.generator_status["running"] = True
    state.generator_status["started_at"] = datetime.now(timezone.utc).isoformat()

    if state.driver is None:
        logger.error("[gen] no D0 driver; aborting")
        state.generator_status["running"] = False
        return

    events_iter = iter(state.driver.events())
    loop = asyncio.get_running_loop()

    try:
        while state.generator_status["running"]:
            cfg = state.generator_status.get("config", {})
            interval = float(cfg.get("interval_seconds", 1.5))
            min_amt = float(cfg.get("min_amount", 0))
            max_amt = float(cfg.get("max_amount", 1e12))

            # Drain any Scenario-Studio-injected events FIRST, so the operator
            # sees what they just queued without competing with the D0 stream.
            if state.studio_injection_queue:
                event = state.studio_injection_queue.pop(0)
            else:
                # Pull next event from D0 (wrap around when exhausted)
                try:
                    event = next(events_iter)
                except StopIteration:
                    events_iter = iter(state.driver.events())
                    continue

                # Amount filter applies to D0 only — injected events bypass.
                if not (min_amt <= float(event.amount) <= max_amt):
                    continue

            t0 = time.time()
            try:
                payload = await loop.run_in_executor(
                    _executor, _run_pipeline_for_event, state, event
                )
            except Exception as exc:
                logger.warning("[gen] pipeline error on %s: %s", event.transaction_id, exc)
                continue
            elapsed = time.time() - t0

            state.generator_status["events_emitted"] += 1
            state.recent_events.insert(0, payload)
            del state.recent_events[MAX_BUFFER:]

            if payload.get("alerted"):
                state.generator_status["alerts_emitted"] += 1
                # Pull the persisted alert dict for the UI buffer
                alert_dict = state.alerts.get(payload["alert_id"])
                if alert_dict is not None:
                    state.recent_alerts.insert(0, alert_dict)
                    del state.recent_alerts[MAX_BUFFER:]

            await _publish(state, payload)

            # Respect cadence; allow the pipeline cost to count against the gap
            remaining = max(0.0, interval - elapsed)
            await asyncio.sleep(remaining)
    except asyncio.CancelledError:
        logger.info("[gen] generator loop cancelled")
    finally:
        state.generator_status["running"] = False
        state.generator_status["stopped_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("[gen] generator loop stopped (emitted %d events, %d alerts)",
                    state.generator_status["events_emitted"],
                    state.generator_status["alerts_emitted"])


async def start_generator(state: AppState) -> dict:
    if state.generator_task is not None and not state.generator_task.done():
        return state.generator_status
    state.generator_task = asyncio.create_task(generator_loop(state))
    return state.generator_status


async def stop_generator(state: AppState) -> dict:
    if state.generator_task is None or state.generator_task.done():
        state.generator_status["running"] = False
        return state.generator_status
    state.generator_status["running"] = False
    state.generator_task.cancel()
    try:
        await state.generator_task
    except (asyncio.CancelledError, Exception):
        pass
    return state.generator_status
