"""Phase 4: ONNX export + INT8 quantization wrappers around the frozen backbone.

The FeatureExtractor network end is exported as a single-call ONNX graph, and
 temporarily using dynamic quantization on that graph gives a first-pass INT8
 binary. Accuracy and latency deltas are measured by the benchmark harness
 driven by results/optimisation.py (committed alongside).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from defectlens.models.features import FeatureExtractor


class _ExtractorGraph(nn.Module):
    def __init__(self, extractor: FeatureExtractor, interp: str = "bilinear") -> None:
        super().__init__()
        self.extractor = extractor
        self.interp = interp

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extractor.embeddings(images, interp=self.interp)


def export_extractor_onnx(
    extractor: FeatureExtractor, out_path: Path, interp: str = "bilinear", input_size: int = 224
) -> Path:
    graph = _ExtractorGraph(extractor, interp=interp).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    onnx_export = torch.onnx.export(
        graph,
        (dummy,),
        str(out_path),
        opset_version=17,
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
    )
    del onnx_export
    return out_path


def quantize_dynamic_int8(onnx_path: Path, out_path: Path) -> Path:
    from onnxruntime import quantization

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantization.quantize_dynamic(
        str(onnx_path), str(out_path), weight_type=quantization.QuantType.QInt8
    )
    return out_path
