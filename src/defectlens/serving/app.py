"""FastAPI serving surface for DefectLens (Phase 5)."""

from __future__ import annotations

import base64
import io
import time
import uuid
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

_MODEL_STATE: dict[str, dict[str, Any]] = {}
_CASE_QUEUE: list[dict[str, Any]] = []

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
    lines = [
        "# HELP defectlens_models_loaded Number of ready models",
        "# TYPE defectlens_models_loaded gauge",
        f"defectlens_models_loaded {len(_MODEL_STATE)}",
        "# HELP review_cases_pending Pending reviews",
        "# TYPE review_cases_pending gauge",
        f"review_cases_pending {len(_CASE_QUEUE)}",
    ]
    return Response("\n".join(lines), media_type="text/plain")


@router.get("/queue")
def queue() -> list[dict[str, Any]]:
    return _CASE_QUEUE


@router.post("/review/{case_id}")
def review(case_id: str, label: str) -> dict[str, Any]:
    for item in _CASE_QUEUE:
        if item["id"] == case_id:
            item["label"] = label
            return dict(item)
    raise HTTPException(status_code=404, detail="case_id not found")


@router.post("/inspect")
async def inspect(
    category: str,
    threshold: float | None = None,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if category not in _MODEL_STATE:
        raise HTTPException(status_code=404, detail=f"category {category!r} not loaded")
    state = _MODEL_STATE[category]
    tensor = _preprocess(io.BytesIO(await file.read()))
    t0 = time.perf_counter()
    with torch.no_grad():
        scores, maps = state["model"].predict(tensor)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    score = float(scores[0].item())
    thresh = threshold if threshold is not None else state.get("threshold", 0.0)
    decision = "anomaly" if score > thresh else "good"
    result: dict[str, Any] = {
        "category": category,
        "score": round(score, 4),
        "decision": decision,
        "threshold": thresh,
        "latency_ms": round(latency_ms, 1),
        "heatmap_png": _heatmap_b64(maps[0, 0].detach().cpu().numpy()),
        "model": state.get("version", "unknown"),
    }
    if decision == "anomaly":
        case_id = str(uuid.uuid4())
        _CASE_QUEUE.append({"id": case_id, "category": category, "score": score})
        result["case_id"] = case_id
    return result


def _preprocess(image: io.BytesIO, size: int = 224) -> torch.Tensor:
    from torchvision.transforms.functional import center_crop, normalize, to_tensor

    try:
        img = Image.open(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="unsupported or corrupt image") from exc
    img = center_crop(img.resize((256, 256), Image.Resampling.BILINEAR), size)
    tensor = to_tensor(img)
    batch = normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    assert isinstance(batch, torch.Tensor)
    return batch


def _heatmap_b64(score_map: np.ndarray) -> str:
    arr = (score_map - score_map.min()) / max(score_map.max() - score_map.min(), 1e-8)
    buf = io.BytesIO()
    Image.fromarray((arr * 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(models: dict[str, dict[str, Any]] | None = None) -> FastAPI:
    _MODEL_STATE.clear()
    _MODEL_STATE.update(models or {})
    _CASE_QUEUE.clear()
    app = FastAPI(title="DefectLens", version="0.1.0")
    app.include_router(router)
    return app
