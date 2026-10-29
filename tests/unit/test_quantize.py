from __future__ import annotations

import pytest
import torch
from torch import nn

from defectlens.models.quantize import quantize_backbone_static


class _TinyBackbone(nn.Module):
    """Two conv stages returning both feature maps, shaped like a features_only timm model.

    Convolutions only, with no residual add: QNNPACK rejects some add scales that a randomly
    initialised ResNet produces, and those are not what this test is about.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
        self.stage2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
        self.feature_info = [{"module": "stage1"}, {"module": "stage2"}]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        first = self.stage1(x)
        return [first, self.stage2(first)]


def test_static_int8_keeps_feature_layout_and_shapes() -> None:
    model = _TinyBackbone().eval()
    calibration = torch.randn(4, 3, 32, 32)
    quantized = quantize_backbone_static(model, calibration, batch_size=2)
    with torch.no_grad():
        fp32 = model(calibration[:1])
        int8 = quantized(calibration[:1])
    assert [t.shape for t in int8] == [t.shape for t in fp32]
    assert all(torch.isfinite(t).all() for t in int8)
    assert quantized.feature_info is model.feature_info


def test_static_int8_needs_calibration_images() -> None:
    with pytest.raises(ValueError, match="calibration"):
        quantize_backbone_static(_TinyBackbone().eval(), torch.empty(0, 3, 32, 32))
