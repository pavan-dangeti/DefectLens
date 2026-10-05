"""Export WRN50 feature extractor to ONNX/Int8 and measure CPU-latency parity."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from defectlens.models.exporters import export_extractor_onnx, quantize_dynamic_int8
from defectlens.models.features import FeatureExtractor

OUT = Path("results/optimisation")
OUT.mkdir(parents=True, exist_ok=True)


def time_run(fn, batch, n=10) -> float:
    # warmup
    for _ in range(3):
        fn(batch)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(batch)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples) * 1000.0)  # median ms


def main() -> None:
    extractor = FeatureExtractor("wide_resnet50_2", ("layer2", "layer3"))
    images = torch.zeros(1, 3, 224, 224)

    onnx_path = OUT / "extractor_fp32.onnx"
    export_extractor_onnx(extractor, onnx_path)
    int8_path = OUT / "extractor_int8_dynamic.onnx"
    quantize_dynamic_int8(onnx_path, int8_path)

    # Measure eager vs ONNX runtime vs int8 runtime
    import onnxruntime as rt

    sess_fp32 = rt.InferenceSession(str(onnx_path))
    sess_int8 = rt.InferenceSession(str(int8_path))

    name = sess_fp32.get_inputs()[0].name
    with torch.no_grad():
        emb = extractor.embeddings(images)
    emb_onnx = sess_fp32.run(None, {name: images.numpy()})[0]
    emb_int8 = sess_int8.run(None, {name: images.numpy()})[0]

    def cos_sim(a: torch.Tensor, b: np.ndarray) -> float:
        a_flat = a.flatten().to(torch.float64)
        b_flat = torch.from_numpy(b).flatten().to(torch.float64)
        return float((a_flat @ b_flat) / (a_flat.norm() * b_flat.norm()))

    cos_sim_fp32 = cos_sim(emb, emb_onnx)
    cos_sim_int8 = cos_sim(emb, emb_int8)

    latency = {
        "eager_ms": time_run(lambda b: extractor.embeddings(b), images),
        "onnx_fp32_ms": time_run(lambda b: sess_fp32.run(None, {name: b.numpy()})[0], images),
        "onnx_int8_ms": time_run(lambda b: sess_int8.run(None, {name: b.numpy()})[0], images),
    }
    onnx_size = onnx_path.stat().st_size
    int8_size = int8_path.stat().st_size
    results = {
        "backbone": "wide_resnet50_2",
        "input": "1x3x224x224",
        "fusion": "layer2+layer3",
        "latency": latency,
        "cosine_similarity": {"fp32": cos_sim_fp32, "int8": cos_sim_int8},
        "file_sizes_bytes": {"fp32": onnx_size, "int8": int8_size},
        "int8_dynamic": True,
    }
    # Hardening: write summary
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
