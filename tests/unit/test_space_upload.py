"""The Space's upload path refuses a huge-pixel header before any pixel is decoded, with the service's error.

hf/space/app.py cannot be imported here: it needs gradio and downloads its manifest at import time. What it
does with an upload is read the file's bytes and pass them to decode_image, so these tests exercise that
function on the same bytes, and check the same refusal the service returns.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import ImageFile

from defectlens.serving.app import create_app
from defectlens.serving.imaging import decode_image
from defectlens.serving.limits import MAX_PIXELS


def _huge_png(width: int, height: int) -> bytes:
    """A valid PNG header and a tiny IDAT that claims a huge image: the size check must refuse it."""

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


class _FakeModel:
    def predict(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = tensor.shape[0]
        return torch.full((n,), 0.5), torch.zeros(n, 1, 224, 224)


@pytest.fixture
def upload(tmp_path: Path) -> bytes:
    # 10000 x 10000 is 100 MP: over the 25 MP cap, in a file of a few hundred bytes.
    path = tmp_path / "upload.png"
    path.write_bytes(_huge_png(10000, 10000))
    assert path.stat().st_size < 1024
    return path.read_bytes()


def test_space_upload_is_refused_before_any_pixel_is_decoded(
    upload: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded_pixels: list[bool] = []

    def spy(self: ImageFile.ImageFile, *args: object, **kwargs: object) -> None:
        decoded_pixels.append(True)
        raise AssertionError("pixels were decoded before the size check")

    monkeypatch.setattr(ImageFile.ImageFile, "load", spy)
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as refused:
        decode_image(upload)
    assert refused.value.status_code == 400
    assert refused.value.detail == f"image exceeds {MAX_PIXELS} pixels"
    assert decoded_pixels == []


def test_service_returns_the_same_refusal_for_the_same_bytes(upload: bytes) -> None:
    client = TestClient(create_app(models={"bottle": {"model": _FakeModel(), "threshold": 0.6}}))
    response = client.post(
        "/inspect", params={"category": "bottle"}, files={"file": ("x.png", upload, "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == f"image exceeds {MAX_PIXELS} pixels"
