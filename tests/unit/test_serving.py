"""Smoke tests for the FastAPI serving shell."""

from __future__ import annotations

import io
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from defectlens.serving.app import create_app


class _FakeModel:
    """Returns a fixed score so assertions do not depend on model internals."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = tensor.shape[0]
        return torch.full((n,), self._score), torch.zeros(n, 1, 224, 224)


def _png(color: str = "gray") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buf, format="PNG")
    return buf.getvalue()


def _client(**models: Any) -> TestClient:
    return TestClient(create_app(models=models))


def test_ready_is_503_without_models() -> None:
    assert _client().get("/ready").status_code == 503


def test_health_and_ready_with_model() -> None:
    client = _client(bottle={"model": _FakeModel(), "threshold": 0.6})
    assert client.get("/health").json() == {"ok": True}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["categories"] == ["bottle"]


def test_inspect_good_decision() -> None:
    client = _client(bottle={"model": _FakeModel(0.5), "threshold": 0.6, "version": "v1"})
    r = client.post("/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 0.5
    assert body["decision"] == "good"
    assert body["model"] == "v1"
    assert body["heatmap_png"]
    assert "case_id" not in body


def test_inspect_anomaly_queues_review_case() -> None:
    client = _client(bottle={"model": _FakeModel(0.9), "threshold": 0.6})
    r = client.post("/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")})
    body = r.json()
    assert body["decision"] == "anomaly"

    pending = client.get("/queue").json()
    assert [c["id"] for c in pending] == [body["case_id"]]

    reviewed = client.post(f"/review/{body['case_id']}?label=confirmed-good").json()
    assert reviewed["label"] == "confirmed-good"


def test_review_unknown_case_is_404() -> None:
    client = _client()
    assert client.post("/review/nope?label=good").status_code == 404


def test_inspect_unknown_category_is_404() -> None:
    client = _client()
    r = client.post("/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")})
    assert r.status_code == 404


def test_inspect_threshold_override() -> None:
    client = _client(bottle={"model": _FakeModel(0.7), "threshold": 0.9})
    r = client.post(
        "/inspect?category=bottle&threshold=0.1", files={"file": ("t.png", _png(), "image/png")}
    )
    assert r.json()["decision"] == "anomaly"


def test_metrics_exposes_gauges() -> None:
    client = _client(bottle={"model": _FakeModel(0.9), "threshold": 0.6})
    client.post("/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")})
    body = client.get("/metrics").text
    assert "defectlens_models_loaded 1" in body
    assert "review_cases_pending 1" in body


@pytest.mark.parametrize("payload", [b"", b"not-an-image"])
def test_inspect_rejects_corrupt_image(payload: bytes) -> None:
    client = _client(bottle={"model": _FakeModel(), "threshold": 0.6})
    r = client.post(
        "/inspect?category=bottle",
        files={"file": ("t.png", payload, "image/png")},
        headers={"accept": "application/json"},
    )
    assert r.status_code >= 400
