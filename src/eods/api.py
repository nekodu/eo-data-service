from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import cube as cubelib
from . import ml
from .auth import Principal, require_scope
from .config import get_settings
from .metrics import CELLS_SERVED, CUBE_LATEST, CUBE_STEPS, LATENCY, MODEL_LOADED, REQUESTS

log = logging.getLogger("eods")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        out.update(getattr(record, "fields", {}))
        return json.dumps(out)


def _setup_logging() -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    log.handlers[:] = [h]
    log.setLevel(logging.INFO)
    log.propagate = False


class State:
    ds = None
    model: dict | None = None


state = State()


def load_resources() -> None:
    s = get_settings()
    state.ds = cubelib.open_cube(s.cube_path) if s.cube_path.exists() else None
    if state.ds is not None and state.ds.sizes["time"]:
        CUBE_STEPS.set(state.ds.sizes["time"])
        CUBE_LATEST.set(state.ds.time.values.max().astype("datetime64[s]").astype(int))
    state.model = ml.load(s.model_path) if s.model_path.exists() else None
    MODEL_LOADED.set(1 if state.model else 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    load_resources()
    log.info(
        "started",
        extra={"fields": {"cube": state.ds is not None, "model": state.model is not None}},
    )
    yield


app = FastAPI(
    title="EO data service", version="0.1.0", lifespan=lifespan, docs_url="/docs", redoc_url=None
)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    dur = time.perf_counter() - start
    REQUESTS.labels(path, request.method, str(response.status_code)).inc()
    LATENCY.labels(path).observe(dur)
    response.headers["x-request-id"] = rid
    if path not in ("/metrics", "/healthz", "/readyz"):
        log.info(
            "request",
            extra={
                "fields": {
                    "request_id": rid,
                    "route": path,
                    "method": request.method,
                    "status": response.status_code,
                    "ms": round(dur * 1000, 1),
                    "client": getattr(request.state, "client_id", None),
                }
            },
        )
    return response


def _cube():
    if state.ds is None:
        raise HTTPException(503, "cube not available")
    return state.ds


def _window(y0: int, y1: int, x0: int, x1: int) -> cubelib.Window:
    try:
        w = cubelib.Window(y0, y1, x0, x1)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    return w


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz():
    if state.ds is None:
        return JSONResponse({"status": "no cube"}, status_code=503)
    return {"status": "ready", "model": state.model is not None}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/cube")
def cube_info(_: Principal = Depends(require_scope("cube:read"))):
    info = cubelib.describe(_cube())
    info["synthetic"] = bool(info["attrs"].get("synthetic", 0))
    return info


@app.get("/v1/cube/cloud-fraction")
def cloud_fraction(
    y0: int = 0,
    y1: int = Query(..., gt=0),
    x0: int = 0,
    x1: int = Query(..., gt=0),
    start: datetime | None = None,
    end: datetime | None = None,
    _: Principal = Depends(require_scope("cube:read")),
):
    ds = _cube()
    w = _window(y0, y1, x0, x1)
    try:
        da = cubelib.select(ds, w, start, end)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if da.sizes["time"] * w.cells > 50 * get_settings().max_cells:
        raise HTTPException(413, "request too large; narrow the window or time range")
    CELLS_SERVED.labels("cloud-fraction").inc(da.size)
    return {"window": w.__dict__, "series": cubelib.cloud_fraction(da)}


@app.get("/v1/cube/window")
def window(
    time_: datetime = Query(..., alias="time"),
    y0: int = 0,
    y1: int = Query(..., gt=0),
    x0: int = 0,
    x1: int = Query(..., gt=0),
    _: Principal = Depends(require_scope("cube:read")),
):
    ds = _cube()
    w = _window(y0, y1, x0, x1)
    if w.cells > get_settings().max_cells:
        raise HTTPException(413, f"window larger than {get_settings().max_cells} cells")
    try:
        da = cubelib.select(ds, w).sel(
            time=np.datetime64(time_.replace(tzinfo=None)), method="nearest"
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    CELLS_SERVED.labels("window").inc(da.size)
    return {"time": str(da.time.values), "window": w.__dict__, "values": da.values.tolist()}


class NowcastRequest(BaseModel):
    y0: int = Field(0, ge=0)
    y1: int = Field(..., gt=0)
    x0: int = Field(0, ge=0)
    x1: int = Field(..., gt=0)
    time: datetime | None = Field(
        None, description="latest observed slot to start from; default newest"
    )
    threshold: float = Field(0.5, gt=0, lt=1)


@app.post("/v1/nowcast")
def nowcast(req: NowcastRequest, _: Principal = Depends(require_scope("nowcast:run"))):
    ds = _cube()
    if state.model is None:
        raise HTTPException(503, "model not loaded")
    w = _window(req.y0, req.y1, req.x0, req.x1)
    if w.cells > get_settings().max_cells:
        raise HTTPException(413, f"window larger than {get_settings().max_cells} cells")
    cfg = state.model["config"]
    try:
        da = cubelib.select(ds, w)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    if req.time is None:
        t_idx = da.sizes["time"] - 1
    else:
        t_idx = int(np.abs(da.time.values - np.datetime64(req.time.replace(tzinfo=None))).argmin())
    if t_idx < cfg["history"] - 1:
        raise HTTPException(422, f"need {cfg['history']} slots before the start time")
    past = da.isel(time=slice(t_idx - cfg["history"] + 1, t_idx + 1)).values
    prob = ml.predict_map(state.model, past)
    valid = ~np.isnan(prob)
    step = int(ds.attrs.get("step_minutes", 15))
    return {
        "from_time": str(da.time.values[t_idx]),
        "lead_minutes": cfg["horizon"] * step,
        "window": w.__dict__,
        "cloud_fraction_now": float((past[-1][valid] == cubelib.CLOUD).mean())
        if valid.any()
        else None,
        "cloud_fraction_forecast": float((prob[valid] >= req.threshold).mean())
        if valid.any()
        else None,
        "mean_probability": float(np.nanmean(prob)) if valid.any() else None,
        "synthetic": bool(ds.attrs.get("synthetic", 0)),
    }
