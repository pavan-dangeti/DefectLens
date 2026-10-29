"""Static INT8 backbone against fp32: accuracy, feature agreement and CPU latency.

For each category the backbone is quantised once, calibrated on the first 32 nominal fit images of seed
0 (never test images). The accuracy run is the same ``evaluate_category`` protocol as the benchmark, with
the quantised backbone swapped in, so the metrics are directly comparable with the committed fp32 raw
results at coreset 0.01, seed 0. Latency is measured after all accuracy runs, on an otherwise idle CPU.

    uv run python results/optimisation/int8_static.py
"""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import defectlens.eval.benchmark as bench
from defectlens.models.quantize import quantize_backbone_static

ROOTS = {"mvtec": Path("data"), "visa": Path("data/visa")}
CATEGORIES = [("mvtec", "screw"), ("mvtec", "toothbrush"), ("visa", "capsules")]
CALIBRATION_IMAGES = 32
RAW = Path("results/evaluation/raw")
OUT = Path("results/optimisation/int8_static")


def _raw_fp32(dataset: str, category: str) -> dict[str, Any]:
    record = json.loads((RAW / f"{dataset}_patchcore_{category}_seed0.json").read_text())
    return record["metrics"]


def _quantised_backbone(root: Path, category: str) -> torch.nn.Module:
    split = bench._split(root, category)
    fit, _, _ = bench._fit_calibration_splits(split["train_good"], seed=0)
    calibration = bench._to_tensors(fit[:CALIBRATION_IMAGES])
    fp32_model = bench._make_model("patchcore", 0.01, 0, pretrained=True)
    return quantize_backbone_static(fp32_model.extractor.model, calibration)


def _accuracy(root: Path, category: str, backbone: torch.nn.Module) -> dict[str, Any]:
    original = bench._make_model

    def with_backbone(method: str, coreset_ratio: float, seed: int, pretrained: bool = True) -> Any:
        model = original(method, coreset_ratio, seed, pretrained=pretrained)
        model.extractor.model = backbone
        return model

    bench._make_model = with_backbone  # type: ignore[assignment]
    try:
        metrics, _ = bench.evaluate_category(root, category, coreset_ratio=0.01, seed=0)
    finally:
        bench._make_model = original  # type: ignore[assignment]
    return metrics


def _median_latency_ms(model: Any, image: torch.Tensor, warmup: int = 5, runs: int = 30) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model.extractor.embeddings(image)
        samples = []
        for _ in range(runs):
            start = time.perf_counter()
            model.extractor.embeddings(image)
            samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _embedding_agreement(fp32: Any, int8: Any, images: torch.Tensor) -> float:
    with torch.no_grad():
        a = fp32.extractor.embeddings(images)
        b = int8.extractor.embeddings(images)
    return float(F.cosine_similarity(a, b, dim=1).mean())


def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    backbones: list[tuple[str, str, torch.nn.Module]] = []
    for dataset, category in CATEGORIES:
        root = ROOTS[dataset]
        backbone = _quantised_backbone(root, category)
        started = time.monotonic()
        int8 = _accuracy(root, category, backbone)
        record = {
            "dataset": dataset,
            "category": category,
            "calibration_images": CALIBRATION_IMAGES,
            "engine": torch.backends.quantized.engine,
            "fp32_committed": _raw_fp32(dataset, category),
            "int8": int8,
            "wall_seconds_accuracy": round(time.monotonic() - started, 1),
            "git_commit": _git_commit(),
        }
        (OUT / f"{dataset}_{category}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"accuracy done: {dataset}/{category}", flush=True)
        backbones.append((dataset, category, backbone))

    # Latency and feature agreement only after every accuracy run, so nothing else loads the CPU.
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "engine": torch.backends.quantized.engine,
        "categories": [],
    }
    for dataset, category, backbone in backbones:
        root = ROOTS[dataset]
        split = bench._split(root, category)
        fp32_model = bench._make_model("patchcore", 0.01, 0, pretrained=True)
        int8_model = bench._make_model("patchcore", 0.01, 0, pretrained=True)
        int8_model.extractor.model = backbone
        image = bench._to_tensors(split["test_good"][:1])
        images = bench._to_tensors(split["test_good"][:16])
        entry = {
            "dataset": dataset,
            "category": category,
            "latency_ms_fp32_median": round(_median_latency_ms(fp32_model, image), 2),
            "latency_ms_int8_median": round(_median_latency_ms(int8_model, image), 2),
            "embedding_cosine_mean_16_images": round(
                _embedding_agreement(fp32_model, int8_model, images), 4
            ),
        }
        summary["categories"].append(entry)
        print(json.dumps(entry), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
