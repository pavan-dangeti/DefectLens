"""Failure-mode suite from design §15: every case must give a clear error or an explicit flag.

Images are synthesised in memory and the model is a fake, so these tests check the serving layer's
behaviour on bad input, not the detector's accuracy. Rotation is the one case the layer cannot
flag yet; it is kept as a strict xfail so the gap stays visible in every run.
"""

from __future__ import annotations

import io
import json
import logging
import struct
import threading
import zlib
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image, ImageFilter

from defectlens.serving.app import create_app
from defectlens.serving.logs import JsonFormatter


class _FakeModel:
    def __init__(self, score: float = 0.5) -> None:
        self._score = score

    def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = tensor.shape[0]
        return torch.full((n,), self._score), torch.zeros(n, 1, 224, 224)


def _client(score: float = 0.5, **kwargs: Any) -> TestClient:
    models = {"bottle": {"model": _FakeModel(score), "threshold": 0.6}}
    return TestClient(create_app(models=models, **kwargs))


def _encode(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _upload(client: TestClient, raw: bytes, category: str = "bottle") -> Any:
    return client.post(
        "/inspect", params={"category": category}, files={"file": ("x.png", raw, "image/png")}
    )


def _huge_png_header(width: int, height: int) -> bytes:
    # A valid IHDR and IDAT with a tiny payload: the header claims a huge image, so only the
    # size check stands between it and a multi-gigabyte decode.
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# --- corrupt and malformed files -------------------------------------------------------------


def test_corrupt_bytes_are_400() -> None:
    response = _upload(_client(), b"definitely not an image")
    assert response.status_code == 400
    assert "corrupt" in response.json()["detail"]


def test_empty_body_is_400() -> None:
    assert _upload(_client(), b"").status_code == 400


def test_truncated_png_is_400() -> None:
    raw = _encode(Image.new("RGB", (64, 64), "gray"))
    assert _upload(_client(), raw[: len(raw) // 2]).status_code == 400


@pytest.mark.parametrize("size", [(1, 1), (1, 20000)])
def test_degenerate_dimensions_are_400(size: tuple[int, int]) -> None:
    response = _upload(_client(), _encode(Image.new("RGB", size, "gray")))
    assert response.status_code == 400
    assert "at least" in response.json()["detail"]


def test_decompression_bomb_is_400_without_decoding() -> None:
    # 30000 x 30000 is 900 MP: refused from the header, never decoded.
    response = _upload(_client(), _huge_png_header(30000, 30000))
    assert response.status_code == 400


@pytest.mark.parametrize(
    ("image", "fmt"),
    [
        (Image.new("L", (224, 224), 128), "PNG"),
        (Image.new("CMYK", (224, 224), (0, 0, 0, 128)), "TIFF"),  # PNG cannot store CMYK
        (Image.new("P", (224, 224)), "PNG"),
    ],
    ids=["grayscale", "cmyk", "palette"],
)
def test_non_rgb_modes_are_accepted(image: Image.Image, fmt: str) -> None:
    assert _upload(_client(), _encode(image, fmt)).status_code == 200


def test_oversized_upload_is_413() -> None:
    client = _client(max_upload_mb=1)
    assert _upload(client, b"\x00" * (1024 * 1024 + 1)).status_code == 413


def test_unknown_category_is_404() -> None:
    response = _upload(_client(), _encode(Image.new("RGB", (224, 224))), category="nope")
    assert response.status_code == 404


def test_missing_model_artefact_is_unready_and_refuses_inspection() -> None:
    client = TestClient(create_app(models={}))
    assert client.get("/ready").status_code == 503
    raw = _encode(Image.new("RGB", (224, 224)))
    assert _upload(client, raw).status_code == 404


# --- lighting, blur and rotation: decision plus an explicit flag ------------------------------


@pytest.mark.parametrize("value", [0, 255], ids=["near_black", "blown_out"])
def test_extreme_lighting_is_flagged_not_rejected(value: int) -> None:
    response = _upload(_client(), _encode(Image.new("RGB", (224, 224), (value,) * 3)))
    body = response.json()
    assert response.status_code == 200
    assert body["low_confidence"] is True
    assert "extreme_lighting" in body["reasons"]
    assert body["decision"] in {"good", "anomaly"}


def _checkerboard(size: int = 256, square: int = 8) -> Image.Image:
    img = Image.new("L", (size, size))
    pixels = img.load()
    assert pixels is not None
    for y in range(size):
        for x in range(size):
            pixels[x, y] = 255 if ((x // square) + (y // square)) % 2 == 0 else 0
    return img.convert("RGB")


def test_blur_is_flagged() -> None:
    blurred = _checkerboard().filter(ImageFilter.GaussianBlur(radius=6))
    body = _upload(_client(), _encode(blurred)).json()
    assert "blur" in body["reasons"]
    assert body["low_confidence"] is True


def test_sharp_image_is_not_flagged() -> None:
    body = _upload(_client(), _encode(_checkerboard())).json()
    assert body["reasons"] == []
    assert body["low_confidence"] is False


@pytest.mark.xfail(
    strict=True,
    reason="rotation outside the training distribution is not detected yet; needs a "
    "feature-distance check (step 4 drift monitoring)",
)
def test_rotation_out_of_distribution_is_flagged() -> None:
    rotated = _checkerboard().rotate(37, resample=Image.Resampling.BILINEAR)
    body = _upload(_client(), _encode(rotated)).json()
    assert body["low_confidence"] is True


# --- batch, concurrency and traffic limits ---------------------------------------------------


def test_batch_reports_per_item_status() -> None:
    files = [
        ("files", ("a.png", _encode(Image.new("RGB", (224, 224))), "image/png")),
        ("files", ("b.png", b"junk", "image/png")),
        ("files", ("c.png", _encode(Image.new("RGB", (224, 224))), "image/png")),
    ]
    body = _client().post("/inspect/batch", params={"category": "bottle"}, files=files).json()
    assert [item["status"] for item in body["results"]] == [200, 400, 200]
    assert body["results"][0]["decision"] in {"good", "anomaly"}


def test_batch_over_limit_is_413() -> None:
    files = [
        ("files", (f"{i}.png", _encode(Image.new("RGB", (224, 224))), "image/png"))
        for i in range(17)
    ]
    response = _client().post("/inspect/batch", params={"category": "bottle"}, files=files)
    assert response.status_code == 413


def test_busy_inference_is_503_immediately() -> None:
    gate = threading.Event()
    entered = threading.Event()

    class _BlockingModel(_FakeModel):
        def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            entered.set()
            gate.wait(timeout=5)
            return super().predict(tensor)

    models = {"bottle": {"model": _BlockingModel(), "threshold": 0.6}}
    client = TestClient(create_app(models=models, max_concurrent_inference=1))
    raw = _encode(Image.new("RGB", (224, 224)))
    results: list[int] = []

    def slow() -> None:
        results.append(_upload(client, raw).status_code)

    worker = threading.Thread(target=slow)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        busy = _upload(client, raw)
        assert busy.status_code == 503
        assert busy.headers["Retry-After"] == "1"
    finally:
        gate.set()
        worker.join(timeout=5)
    assert results == [200]


def test_batch_is_charged_per_image_not_per_request() -> None:
    client = _client(rate_limit_per_minute=3)
    files = [
        ("files", (f"{i}.png", _encode(Image.new("RGB", (224, 224))), "image/png"))
        for i in range(2)
    ]
    first = client.post("/inspect/batch", params={"category": "bottle"}, files=files)
    second = client.post("/inspect/batch", params={"category": "bottle"}, files=files)
    assert first.status_code == 200
    assert second.status_code == 429


def test_exhausted_client_is_refused_before_the_batch_is_read() -> None:
    client = _client(rate_limit_per_minute=1)
    raw = _encode(Image.new("RGB", (224, 224)))
    assert _upload(client, raw).status_code == 200
    files = [("files", ("a.png", raw, "image/png"))]
    refused = client.post("/inspect/batch", params={"category": "bottle"}, files=files)
    assert refused.status_code == 429
    assert refused.headers["Retry-After"] == "60"


def test_rate_limit_is_429_with_retry_after() -> None:
    client = _client(rate_limit_per_minute=2)
    raw = _encode(Image.new("RGB", (224, 224)))
    statuses = [_upload(client, raw).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert _upload(client, raw).headers["Retry-After"] == "60"


# --- observability ---------------------------------------------------------------------------


def test_request_id_is_echoed_or_replaced() -> None:
    client = _client()
    kept = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert kept.headers["X-Request-ID"] == "abc-123"
    replaced = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert replaced.headers["X-Request-ID"] != "bad id with spaces"
    assert len(replaced.headers["X-Request-ID"]) == 32


def test_access_log_carries_request_id_and_no_client_address(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="defectlens.access")
    _client().get("/health", headers={"X-Request-ID": "trace-1"})
    record = next(r for r in caplog.records if getattr(r, "request_id", None) == "trace-1")
    line = json.loads(JsonFormatter().format(record))
    assert line["route"] == "/health"
    assert line["status"] == 200
    assert "client" not in line and "ip" not in line


def test_unknown_path_is_logged_without_query_or_client(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="defectlens.access")
    _client().get("/wp-login.php?x=1")
    record = next(r for r in caplog.records if getattr(r, "route", None) == "other")
    line = json.loads(JsonFormatter().format(record))
    assert line["path"] == "/wp-login.php"
    assert "?" not in line["path"] and "client" not in line and "ip" not in line


def test_metrics_expose_request_and_decision_counters() -> None:
    client = _client()
    _upload(client, _encode(Image.new("RGB", (224, 224))))
    text = client.get("/metrics").text
    assert 'defectlens_requests_total{route="/inspect",status="200"}' in text
    assert 'defectlens_decisions_total{category="bottle",decision=' in text
    assert "defectlens_models_loaded 1.0" in text
    assert "review_cases_pending" in text


def test_review_queue_gauge_tracks_flagged_cases(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            models={"bottle": {"model": _FakeModel(0.9), "threshold": 0.6}},
            spool_dir=tmp_path / "queue",
        )
    )
    _upload(client, _encode(Image.new("RGB", (224, 224))))
    assert "review_cases_pending 1.0" in client.get("/metrics").text
