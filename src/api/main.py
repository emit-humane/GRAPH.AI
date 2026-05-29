"""GRAPH.AI FastAPI app — entry point at :8000.

Run via::

    uvicorn src.api.main:app --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import generator as gen_module
from .routers import (
    build_alerts_router,
    build_generator_router,
    build_geo_router,
    build_graph_router,
    build_report_router,
    build_stats_router,
    build_stream_router,
)
from .state import AppState, build_app_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Build state + app at import time so routers are mounted before TestClient
# --------------------------------------------------------------------------- #


logger.info("[api] importing — loading shared state ...")
_STATE: AppState = build_app_state()
logger.info(
    "[api] state ready: layers supervised=%s anomaly=%s tgn=%s; driver_events=%d",
    _STATE.supervised is not None,
    _STATE.anomaly is not None,
    _STATE.tgn is not None,
    _STATE.driver_count,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.appstate = _STATE
    _STATE.generator_lock = asyncio.Lock()
    yield
    logger.info("[api] shutdown — stopping generator ...")
    if _STATE.generator_task is not None and not _STATE.generator_task.done():
        _STATE.generator_status["running"] = False
        _STATE.generator_task.cancel()
        try:
            await _STATE.generator_task
        except Exception:
            pass


app = FastAPI(title="GRAPH.AI Backend", version="1.0.0", lifespan=lifespan)
app.state.appstate = _STATE

# CORS — open for local dev (frontend at :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "GRAPH.AI",
        "status": "ok",
        "endpoints": [
            "/generator/start", "/generator/stop", "/generator/status",
            "/stream/events",
            "/alerts", "/alerts/{id}", "/alerts/{id}/status",
            "/graph/case/{tx_id}", "/graph/subgraph",
            "/geo/case/{tx_id}",
            "/report/{tx_id}", "/report/{tx_id}/pdf", "/report/{tx_id}/zip",
            "/stats/evaluation",
        ],
    }


@app.get("/health")
def health() -> dict:
    state: AppState = app.state.appstate  # type: ignore[attr-defined]
    return {
        "status": "ok",
        "generator_running": state.generator_status["running"],
        "events_emitted": state.generator_status["events_emitted"],
        "alerts_emitted": state.generator_status["alerts_emitted"],
    }


# --------------------------------------------------------------------------- #
# Routers (mounted at import time so TestClient sees them immediately)
# --------------------------------------------------------------------------- #


app.include_router(build_generator_router(_STATE))
app.include_router(build_stream_router(_STATE))
app.include_router(build_alerts_router(_STATE))
app.include_router(build_graph_router(_STATE))
app.include_router(build_geo_router(_STATE))
app.include_router(build_report_router(_STATE))
app.include_router(build_stats_router(_STATE))
