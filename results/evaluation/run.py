"""Run the full-category evaluation protocol and write one raw JSON per (dataset, method, category, seed).

Resumable: a run whose output file already exists is skipped, so an interrupted sweep continues
where it stopped. Each file records the coreset ratio, environment and wall time next to the
metrics, so a number can always be traced to the exact configuration that produced it.

    uv run python results/evaluation/run.py --dataset visa --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from defectlens import __version__
from defectlens.eval.benchmark import evaluate_category

ROOTS = {"mvtec": Path("data"), "visa": Path("data/visa")}
CATEGORIES = {
    "mvtec": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
    "visa": [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ],
}
OUT = Path("results/evaluation/raw")


def _command_output(*argv: str) -> str:
    out = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return out.stdout.strip()


def _environment() -> dict[str, str]:
    return {
        "defectlens": __version__,
        "git_commit": _command_output("git", "rev-parse", "--short", "HEAD") or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": _command_output("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def run_one(
    dataset: str, method: str, category: str, seed: int, coreset_ratio: float, out_dir: Path
) -> Path:
    path = out_dir / f"{dataset}_{method}_{category}_seed{seed}.json"
    if path.exists():
        print(f"skip {path.name} (exists)", flush=True)
        return path
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    metrics, _ = evaluate_category(
        ROOTS[dataset],
        category,
        method=method,
        coreset_ratio=coreset_ratio,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "category": category,
        "seed": seed,
        "coreset_ratio": coreset_ratio,
        "started_utc": started.isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "environment": _environment(),
        "metrics": metrics,
    }
    _write_atomic(path, payload)
    print(
        f"done {path.name} image_auroc={metrics['image_auroc']:.4f} "
        f"pixel_auroc={metrics['pixel_auroc']:.4f} aupro={metrics['aupro']:.4f} "
        f"wall={payload['wall_seconds']}s",
        flush=True,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(ROOTS), required=True)
    parser.add_argument("--method", choices=["patchcore", "padim"], default="patchcore")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    categories = args.categories or CATEGORIES[args.dataset]
    for seed in args.seeds:
        for category in categories:
            run_one(args.dataset, args.method, category, seed, args.coreset_ratio, args.out_dir)


if __name__ == "__main__":
    main()
