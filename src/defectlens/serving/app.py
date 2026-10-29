"""FastAPI serving surface for DefectLens (Phase 5).

Inference runs in a worker thread behind a bounded slot pool. A request that finds every slot busy
is refused with 503 straight away, so latency stays bounded under overload instead of queueing
without limit.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from defectlens.serving import metrics as m
from defectlens.serving.imaging import Decoded, decode_image
from defectlens.serving.limits import MAX_BATCH, SlidingWindowLimiter, client_key
from defectlens.serving.logs import ACCESS_LOGGER

_MODEL_STATE: dict[str, dict[str, Any]] = {}
_CASE_QUEUE: list[dict[str, Any]] = []
_SPOOL_DIR: Path | None = None
_REVIEW_LOG: Path | None = None
_UPLOAD_LIMIT_BYTES = 20 * 1024 * 1024
_LIMITER: SlidingWindowLimiter | None = None
_SLOTS = threading.BoundedSemaphore(2)

# "confirmed-good" is the only label that feeds a rebuild; the other two record rejections.
LABELS = frozenset({"confirmed-good", "confirmed-defect", "false-positive"})

RATE_LIMITED_ROUTES = frozenset({"/inspect", "/inspect/batch"})
# Multipart framing adds a little to the declared body size on top of the image bytes.
_MULTIPART_SLACK_BYTES = 64 * 1024
KNOWN_ROUTES = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/queue",
        "/review/{id}",
        "/review/{id}/image",
        "/inspect",
        "/inspect/batch",
    }
)
# Client-supplied IDs are echoed into logs and headers, so they are restricted to a safe alphabet.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_ACCESS_LOG = logging.getLogger(ACCESS_LOGGER)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@router.get("/ready")
def ready() -> Response:
    if not _MODEL_STATE:
        return JSONResponse({"ok": False, "reason": "no models loaded"}, status_code=503)
    return JSONResponse({"ok": True, "categories": sorted(_MODEL_STATE)})


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(m.REGISTRY), media_type=CONTENT_TYPE_LATEST)


@router.get("/queue")
def queue() -> list[dict[str, Any]]:
    return _CASE_QUEUE


@router.post("/review/{case_id}")
def review(case_id: str, label: str) -> dict[str, Any]:
    if label not in LABELS:
        raise HTTPException(status_code=422, detail=f"label must be one of {sorted(LABELS)}")
    for item in _CASE_QUEUE:
        if item["id"] == case_id:
            item["label"] = label
            _append_review(item)
            return dict(item)
    raise HTTPException(status_code=404, detail="case_id not found")


@router.get("/review/{case_id}/image")
def review_image(case_id: str) -> FileResponse:
    for item in _CASE_QUEUE:
        if item["id"] == case_id and item.get("image_path"):
            return FileResponse(item["image_path"], media_type="image/png")
    raise HTTPException(status_code=404, detail="no image for that case_id")


@router.post("/inspect")
async def inspect(
    request: Request,
    category: str,
    threshold: float | None = None,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    _charge(request, 1)
    _require_category(category)
    raw = await _read_upload(file)
    outcome = (await _score_request(category, [raw], threshold))[0]
    if isinstance(outcome, HTTPException):
        raise outcome
    return outcome


@router.post("/inspect/batch")
async def inspect_batch(
    request: Request,
    category: str,
    files: list[UploadFile] = File(...),
    threshold: float | None = None,
) -> dict[str, Any]:
    _charge(request, len(files))
    _require_category(category)
    if len(files) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"at most {MAX_BATCH} images per batch")
    raws = [await _read_upload(item) for item in files]
    outcomes = await _score_request(category, raws, threshold)
    results: list[dict[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        if isinstance(outcome, HTTPException):
            results.append(
                {"index": index, "status": outcome.status_code, "detail": outcome.detail}
            )
        else:
            results.append({"index": index, "status": 200, **outcome})
    return {"category": category, "count": len(results), "results": results}


def _require_category(category: str) -> None:
    if category not in _MODEL_STATE:
        raise HTTPException(status_code=404, detail=f"category {category!r} not loaded")


def _charge(request: Request, units: int) -> None:
    # Charged per image in the endpoint, not per request in the middleware, so a batch pays for
    # every image it carries. The client address keys the budget only and is never logged.
    if _LIMITER is None:
        return
    host = request.client.host if request.client else "unknown"
    if not _LIMITER.allow(client_key(host), cost=units):
        m.REJECTED.labels(reason="rate_limited").inc()
        raise HTTPException(
            status_code=429, detail="rate limit exceeded", headers={"Retry-After": "60"}
        )


async def _read_upload(file: UploadFile) -> bytes:
    # Read one byte past the limit so an oversized upload is detected without buffering all of it.
    raw = await file.read(_UPLOAD_LIMIT_BYTES + 1)
    if len(raw) > _UPLOAD_LIMIT_BYTES:
        m.REJECTED.labels(reason="too_large").inc()
        raise HTTPException(
            status_code=413, detail=f"upload exceeds {_UPLOAD_LIMIT_BYTES // (1024 * 1024)} MB"
        )
    return raw


async def _score_request(
    category: str, raws: list[bytes], threshold: float | None
) -> list[dict[str, Any] | HTTPException]:
    if not _SLOTS.acquire(blocking=False):
        m.REJECTED.labels(reason="busy").inc()
        raise HTTPException(
            status_code=503,
            detail="inference is at capacity; retry shortly",
            headers={"Retry-After": "1"},
        )
    try:
        return await run_in_threadpool(_score_images, category, raws, threshold)
    finally:
        _SLOTS.release()


def _score_images(
    category: str, raws: list[bytes], threshold: float | None
) -> list[dict[str, Any] | HTTPException]:
    state = _MODEL_STATE[category]
    decoded: list[Decoded | HTTPException] = []
    for raw in raws:
        try:
            decoded.append(decode_image(raw))
        except HTTPException as exc:
            m.REJECTED.labels(reason="invalid_image").inc()
            decoded.append(exc)

    valid = [item.tensor for item in decoded if isinstance(item, Decoded)]
    if not valid:
        return [item for item in decoded if isinstance(item, HTTPException)]

    t0 = time.perf_counter()
    with torch.no_grad():
        scores, maps = state["model"].predict(torch.cat(valid))
    per_image_ms = (time.perf_counter() - t0) * 1000.0 / len(valid)
    m.INFERENCE_SECONDS.labels(category).observe(per_image_ms / 1000.0)

    thresh = threshold if threshold is not None else state.get("threshold", 0.0)
    outcomes: list[dict[str, Any] | HTTPException] = []
    index = 0
    for raw, item in zip(raws, decoded, strict=True):
        if isinstance(item, HTTPException):
            outcomes.append(item)
            continue
        outcomes.append(
            _record(
                category=category,
                state=state,
                thresh=thresh,
                raw=raw,
                score=float(scores[index].item()),
                score_map=maps[index, 0].detach().cpu().numpy(),
                latency_ms=per_image_ms,
                reasons=item.reasons,
            )
        )
        index += 1
    return outcomes


def _record(
    *,
    category: str,
    state: dict[str, Any],
    thresh: float,
    raw: bytes,
    score: float,
    score_map: np.ndarray,
    latency_ms: float,
    reasons: tuple[str, ...],
) -> dict[str, Any]:
    decision = "anomaly" if score > thresh else "good"
    m.DECISIONS.labels(category, decision).inc()
    m.ANOMALY_SCORE.labels(category).observe(score)
    for reason in reasons:
        m.LOW_CONFIDENCE.labels(reason).inc()
    result: dict[str, Any] = {
        "category": category,
        "score": round(score, 4),
        "decision": decision,
        "threshold": thresh,
        "latency_ms": round(latency_ms, 1),
        "low_confidence": bool(reasons),
        "reasons": list(reasons),
        "heatmap_png": _heatmap_b64(score_map),
        "model": state.get("version", "unknown"),
    }
    if decision == "anomaly":
        case_id = str(uuid.uuid4())
        _CASE_QUEUE.append(
            {
                "id": case_id,
                "category": category,
                "score": score,
                "image_path": _spool(case_id, raw),
            }
        )
        result["case_id"] = case_id
    return result


def _heatmap_b64(score_map: np.ndarray) -> str:
    arr = (score_map - score_map.min()) / max(score_map.max() - score_map.min(), 1e-8)
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _spool(case_id: str, raw: bytes) -> str | None:
    """Keep the flagged image so a human can look at it and a refit can learn from it."""
    if _SPOOL_DIR is None:
        return None
    _SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    path = _SPOOL_DIR / f"{case_id}.png"
    path.write_bytes(raw)
    return str(path)


def _append_review(item: dict[str, Any]) -> None:
    """Append one labelled case so ``defectlens fit`` can rebuild the bank from the feedback."""
    if _REVIEW_LOG is None:
        return
    _REVIEW_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {k: v for k, v in item.items() if k != "id"} | {"case_id": item["id"]}
    with _REVIEW_LOG.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _route_label(path: str) -> str:
    # Collapse path parameters so the metric label set stays bounded.
    if path.startswith("/review/"):
        return "/review/{id}/image" if path.endswith("/image") else "/review/{id}"
    return path if path in KNOWN_ROUTES else "other"


def _client_host(request: Request) -> str:
    # The address keys the rate limit only. It is never logged.
    return request.client.host if request.client else "unknown"


def _pre_check(request: Request, route: str) -> Response | None:
    # Refuse cheaply before FastAPI parses the multipart body: a client already out of budget, or
    # a declared body larger than the route could ever legitimately accept.
    if _LIMITER is not None and _LIMITER.exhausted(client_key(_client_host(request))):
        m.REJECTED.labels(reason="rate_limited").inc()
        return JSONResponse(
            {"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"}
        )
    declared = request.headers.get("content-length", "")
    images = MAX_BATCH if route == "/inspect/batch" else 1
    if declared.isdigit() and int(declared) > images * _UPLOAD_LIMIT_BYTES + _MULTIPART_SLACK_BYTES:
        m.REJECTED.labels(reason="too_large").inc()
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    return None


def _incoming_request_id(value: str | None) -> str:
    if value and _REQUEST_ID.match(value):
        return value
    return uuid.uuid4().hex


async def _observe(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    started = time.perf_counter()
    request_id = _incoming_request_id(request.headers.get("x-request-id"))
    route = _route_label(request.url.path)
    rejection = (
        _pre_check(request, route)
        if request.method == "POST" and route in RATE_LIMITED_ROUTES
        else None
    )
    response = rejection if rejection is not None else await call_next(request)

    duration = time.perf_counter() - started
    m.REQUESTS.labels(route, str(response.status_code)).inc()
    m.REQUEST_SECONDS.labels(route).observe(duration)
    response.headers["X-Request-ID"] = request_id
    _ACCESS_LOG.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "route": route,
            "status": response.status_code,
            "duration_ms": round(duration * 1000.0, 1),
        },
    )
    return response


def create_app(
    models: dict[str, dict[str, Any]] | None = None,
    *,
    spool_dir: Path | None = None,
    review_log: Path | None = None,
    max_upload_mb: int = 20,
    max_concurrent_inference: int = 2,
    rate_limit_per_minute: int = 60,
) -> FastAPI:
    _MODEL_STATE.clear()
    _MODEL_STATE.update(models or {})
    _CASE_QUEUE.clear()
    global _SPOOL_DIR, _REVIEW_LOG, _UPLOAD_LIMIT_BYTES, _LIMITER, _SLOTS
    _SPOOL_DIR, _REVIEW_LOG = spool_dir, review_log
    _UPLOAD_LIMIT_BYTES = max_upload_mb * 1024 * 1024
    _LIMITER = SlidingWindowLimiter(rate_limit_per_minute, window_seconds=60.0)
    _SLOTS = threading.BoundedSemaphore(max_concurrent_inference)
    m.MODELS_LOADED.set_function(lambda: len(_MODEL_STATE))
    m.REVIEW_PENDING.set_function(lambda: len(_CASE_QUEUE))

    app = FastAPI(title="DefectLens", version="0.1.0")
    app.include_router(router)
    app.middleware("http")(_observe)
    return app
