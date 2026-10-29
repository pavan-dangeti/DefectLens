"""Resident memory of a serving process: load the registry, then score nominal images.

This is what SLO-3 measures. A serving process never builds the full memory bank, so its peak is the
backbone, the Python and torch runtime, and the saved coreset. Run it on one artefact per process:

    uv run defectlens fit --root data/visa --category candle --out <registry> --coreset-ratio 0.01
    uv run python results/evaluation/serving_rss.py --registry <registry> --root data/visa --category candle

The output JSON records the artefact size and the peak resident memory after the registry is loaded and
after 30 predictions. Images are read from the local dataset and are not committed.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
from pathlib import Path
from typing import Any

import torch

import defectlens.eval.benchmark as bench
from defectlens.serving.registry import load_registry

RUNS = 30


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    before = _peak_rss_mb()
    registry = load_registry(args.registry)
    loaded = _peak_rss_mb()
    model = registry[args.category]["model"]
    artefact = args.registry / f"{args.category}.pt"

    split = bench._split(args.root, args.category)
    images = bench._to_tensors(split["test_good"][:RUNS])
    with torch.no_grad():
        for i in range(RUNS):
            model.predict(images[i : i + 1])
    scored = _peak_rss_mb()

    record: dict[str, Any] = {
        "category": args.category,
        "artefact_mb": round(artefact.stat().st_size / (1024 * 1024), 1),
        "coreset_entries": int(model._coreset.shape[0]) if model._coreset is not None else None,
        "rss_before_load_mb": round(before, 0),
        "rss_after_load_mb": round(loaded, 0),
        "rss_after_predict_mb": round(scored, 0),
        "predictions": RUNS,
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
