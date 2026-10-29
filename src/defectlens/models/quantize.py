"""Static INT8 quantisation of the frozen backbone, calibrated on nominal images.

Unlike the dynamic quantisation in exporters.py, static quantisation also fixes activation ranges
from a calibration set, so convolutions are quantised as well. It runs eagerly, through FX graph
mode, on the platform's default quantised engine (qnnpack on Apple Silicon, fbgemm on x86). The
calibration images must be nominal and must not be test images.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx


def _select_engine() -> str:
    """fbgemm on x86, qnnpack on arm64: the first supported engine, set as the active one."""
    for name in ("fbgemm", "qnnpack"):
        if name in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError("no quantised engine (fbgemm or qnnpack) is available in this PyTorch build")


def quantize_backbone_static(
    model: nn.Module, calibration: torch.Tensor, batch_size: int = 8
) -> nn.Module:
    """Return an INT8 copy of ``model``, with activation ranges observed on ``calibration``.

    ``calibration`` is a (N, 3, H, W) batch of normalised images at the serving input size.
    """
    if calibration.shape[0] == 0:
        raise ValueError("static quantisation needs at least one calibration image")
    engine = _select_engine()
    # torch does not annotate get_default_qconfig, so mypy --strict needs this one call exempted.
    qmap = QConfigMapping().set_global(get_default_qconfig(engine))  # type: ignore[no-untyped-call]
    prepared = prepare_fx(model, qmap, example_inputs=(calibration[:1],))
    with torch.no_grad():
        for start in range(0, calibration.shape[0], batch_size):
            prepared(calibration[start : start + batch_size])
    quantized = convert_fx(prepared)
    # FeatureExtractor reads the layer names from this attribute, which the converted graph drops.
    quantized.feature_info = model.feature_info
    return quantized
