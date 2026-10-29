"""Cost of the coreset ratio: fit time, peak memory, artefact size and per-image latency.

Each (ratio, category) runs in its own process, so the peak resident memory measured for one fit is not
inherited from an earlier one. The fit set and the calibration split are the seed-0 split used by the
benchmark. Latency is the median of 30 single-image predictions on nominal test images, after warm-up,
on an otherwise idle CPU.

    uv run python results/evaluation/coreset_cost.py
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

import defectlens.eval.benchmark as bench
from defectlens.models.patchcore import PatchCore

ROOT = Path("data/visa")
CATEGORIES = ["capsules", "macaroni1", "macaroni2", "candle"]
RATIOS = [0.01, 0.1]
OUT = Path("results/evaluation/coreset_cost")
LATENCY_RUNS = 30


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def worker(ratio: float, category: str) -> dict[str, Any]:
    split = bench._split(ROOT, category)
    fit_paths, _, _ = bench._fit_calibration_splits(split["train_good"], seed=0)
    fit_tensor = bench._to_tensors(fit_paths)
    test_tensor = bench._to_tensors(split["test_good"][:LATENCY_RUNS])

    model = PatchCore(backbone="wide_resnet50_2", coreset_ratio=ratio, seed=0, pretrained=True)
    started = time.perf_counter()
    model.fit(fit_tensor)
    fit_seconds = time.perf_counter() - started
    coreset = model._coreset
    assert coreset is not None

    with tempfile.TemporaryDirectory() as tmp:
        artefact = Path(tmp) / f"{category}.pt"
        model.save(artefact, category=category, threshold=0.0)
        artefact_mb = artefact.stat().st_size / (1024 * 1024)

    with torch.no_grad():
        for i in range(3):  # warm-up
            model.predict(test_tensor[i : i + 1])
        samples = []
        for i in range(LATENCY_RUNS):
            t0 = time.perf_counter()
            model.predict(test_tensor[i : i + 1])
            samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "ratio": ratio,
        "category": category,
        "n_fit_images": len(fit_paths),
        "coreset_entries": int(coreset.shape[0]),
        "coreset_dims": int(coreset.shape[1]),
        "artefact_mb": round(artefact_mb, 1),
        "fit_seconds": round(fit_seconds, 1),
        "predict_ms_median": round(statistics.median(samples), 1),
        "predict_ms_p95": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 1),
        "peak_rss_mb": round(_peak_rss_mb(), 0),
    }


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Category | Coreset ratio | Bank entries | Artefact (MB) | Predict median (ms) | Predict p95 (ms) "
        "| Fit time (s) | Peak RSS while fitting (MB) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["category"], r["ratio"])):
        lines.append(
            f"| {row['category']} | {row['ratio']} | {row['coreset_entries']} | {row['artefact_mb']} "
            f"| {row['predict_ms_median']} | {row['predict_ms_p95']} | {row['fit_seconds']} "
            f"| {row['peak_rss_mb']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--table", action="store_true", help="render coreset_cost.md from summary.json"
    )
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--category")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.ratio, args.category)), flush=True)
        return
    if args.table:
        rows = json.loads((OUT / "summary.json").read_text())
        table = render_table(rows)
        (OUT / "coreset_cost.md").write_text(table)
        print(table)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for category in CATEGORIES:
        for ratio in RATIOS:
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--ratio",
                    str(ratio),
                    "--category",
                    category,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            row = json.loads(result.stdout.strip().splitlines()[-1])
            (OUT / f"{category}_{ratio}.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
