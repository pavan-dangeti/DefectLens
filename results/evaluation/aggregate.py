"""Aggregate raw per-seed evaluation runs into summary.json with 95% confidence intervals.

Per category, each metric is the mean over seeds with a Student-t 95% CI. The dataset average
takes the mean over categories within each seed first, so the CI reflects seed-to-seed variation
of the whole benchmark, not the spread between categories.

    uv run python results/evaluation/aggregate.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RAW = Path("results/evaluation/raw")
OUT = Path("results/evaluation/summary.json")
METRICS = ("image_auroc", "pixel_auroc", "aupro", "escape_rate_at_target_frr")


def _ci(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    if n < 2:
        return {"mean": mean, "ci95_half_width": float("nan"), "n_seeds": n}
    sem = float(arr.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(0.975, df=n - 1) * sem)
    return {"mean": mean, "ci95_half_width": half, "n_seeds": n}


def load_runs() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RAW.glob("*.json"))]


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        by_category[(run["dataset"], run["method"], run["category"])].append(run)
        by_seed[(run["dataset"], run["method"])][run["seed"]].append(run)

    categories: dict[str, Any] = {}
    for (dataset, method, category), group in sorted(by_category.items()):
        entry: dict[str, Any] = {"coreset_ratio": group[0]["coreset_ratio"]}
        for metric in METRICS:
            entry[metric] = _ci([float(r["metrics"][metric]) for r in group])
        few_shot_keys = group[0]["metrics"]["few_shot_image_auroc"].keys()
        entry["few_shot_image_auroc"] = {
            k: _ci([float(r["metrics"]["few_shot_image_auroc"][k]) for r in group])
            for k in few_shot_keys
        }
        categories[f"{dataset}/{method}/{category}"] = entry

    datasets: dict[str, Any] = {}
    for (dataset, method), seeds_map in sorted(by_seed.items()):
        seeds = sorted(seeds_map)
        metrics_summary: dict[str, Any] = {}
        for metric in METRICS:
            per_seed_mean = [
                float(np.mean([float(r["metrics"][metric]) for r in seeds_map[s]])) for s in seeds
            ]
            metrics_summary[metric] = _ci(per_seed_mean)
        datasets[f"{dataset}/{method}"] = {
            "n_categories": len(seeds_map[seeds[0]]),
            "seeds": seeds,
            "mean_over_categories": metrics_summary,
        }

    return {"datasets": datasets, "categories": categories}


def main() -> None:
    summary = aggregate(load_runs())
    OUT.write_text(json.dumps(summary, indent=2))
    for name, info in summary["datasets"].items():
        print(f"{name}: {info['n_categories']} categories, seeds {info['seeds']}")
        for metric, ci in info["mean_over_categories"].items():
            print(
                f"  {metric:28s} {ci['mean']:.4f} ± {ci['ci95_half_width']:.4f} (n={ci['n_seeds']})"
            )


if __name__ == "__main__":
    main()
