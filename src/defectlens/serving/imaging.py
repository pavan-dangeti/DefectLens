"""Decode and validate uploads, and flag inputs the model has no basis to score with confidence.

Validation runs before any tensor is built, so a bad upload is refused with a specific 400 instead
of reaching the backbone. The low-confidence checks do not reject the image: the design requires a
decision plus an explicit flag, so a dark or blurred part still gets scored and labelled.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import center_crop, normalize, to_tensor

from defectlens.serving.limits import MAX_PIXELS, MIN_SIDE

# Heuristic bounds on the 0-255 luminance scale, measured on the 256-pixel working image. They
# catch inputs the bank was never fitted on (lights off, blown out, out of focus). They are not a
# calibrated out-of-distribution test, and rotation is not covered by them.
LIGHTING_RANGE = (15.0, 240.0)
SHARPNESS_FLOOR = 10.0

_CORRUPT = "unsupported or corrupt image"


@dataclass(frozen=True)
class Decoded:
    tensor: torch.Tensor
    reasons: tuple[str, ...]


def decode_image(raw: bytes) -> Decoded:
    try:
        probe = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=_CORRUPT) from exc
    # The size is read from the header, before any pixel is decoded, so a bomb is refused cheaply.
    width, height = probe.size
    if width * height > MAX_PIXELS:
        raise HTTPException(status_code=400, detail=f"image exceeds {MAX_PIXELS} pixels")
    if min(width, height) < MIN_SIDE:
        raise HTTPException(status_code=400, detail=f"image sides must be at least {MIN_SIDE}px")
    try:
        probe.load()
        rgb = probe.convert("RGB")
    except (OSError, ValueError) as exc:  # truncated stream or invalid palette
        raise HTTPException(status_code=400, detail=_CORRUPT) from exc

    working = rgb.resize((256, 256), Image.Resampling.BILINEAR)
    reasons = _low_confidence_reasons(working)
    tensor = normalize(
        to_tensor(center_crop(working, 224)),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).unsqueeze(0)
    assert isinstance(tensor, torch.Tensor)
    return Decoded(tensor=tensor, reasons=reasons)


def _low_confidence_reasons(image: Image.Image) -> tuple[str, ...]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    reasons: list[str] = []
    mean = float(gray.mean())
    if not LIGHTING_RANGE[0] <= mean <= LIGHTING_RANGE[1]:
        reasons.append("extreme_lighting")
    # Variance of the 4-neighbour Laplacian: near zero when edges are soft or absent.
    lap = (
        4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    if float(lap.var()) < SHARPNESS_FLOOR:
        reasons.append("blur")
    return tuple(reasons)
