"""Smoke tests for the FastAPI serving shell."""

from __future__ import annotations

import io
import json
from pathlib import Path
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


def _spooling_client(tmp_path: Path, score: float = 0.9, **kwargs: Any) -> TestClient:
    return TestClient(
        create_app(
            models={"bottle": {"model": _FakeModel(score), "threshold": 0.6}},
            spool_dir=tmp_path / "queue",
            review_log=tmp_path / "reviews.jsonl",
            **kwargs,
        )
    )


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
    assert client.post("/review/nope?label=confirmed-good").status_code == 404


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


def test_review_loop_spools_image_and_appends_log(tmp_path: Path) -> None:
    client = _spooling_client(tmp_path)
    case_id = client.post(
        "/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")}
    ).json()["case_id"]

    image = client.get(f"/review/{case_id}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"

    client.post(f"/review/{case_id}?label=confirmed-good")
    logged = json.loads((tmp_path / "reviews.jsonl").read_text().splitlines()[0])
    assert logged["label"] == "confirmed-good"
    assert logged["category"] == "bottle"
    assert logged["image_path"] == str(tmp_path / "queue" / f"{case_id}.png")


def test_review_rejects_unknown_label(tmp_path: Path) -> None:
    client = _spooling_client(tmp_path)
    case_id = client.post(
        "/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")}
    ).json()["case_id"]
    assert client.post(f"/review/{case_id}?label=whatever").status_code == 422
    assert not (tmp_path / "reviews.jsonl").exists()


def test_image_route_404_without_spool(tmp_path: Path) -> None:
    client = _client(bottle={"model": _FakeModel(0.9), "threshold": 0.6})
    case_id = client.post(
        "/inspect?category=bottle", files={"file": ("t.png", _png(), "image/png")}
    ).json()["case_id"]
    assert client.get(f"/review/{case_id}/image").status_code == 404
