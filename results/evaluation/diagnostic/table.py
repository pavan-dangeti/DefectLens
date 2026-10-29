"""Build the anomalib-default diagnostic table from committed results.

Baseline is our configuration (coreset 0.01, fit on 80% of train-good, 224 crop) from the committed
three-seed raw results. Each variant is the seed-0 run in this directory. Because the variants are
single-seed, the table also reports the seed-to-seed range of the baseline, so a change can be
judged against the noise it sits inside.

    uv run python results/evaluation/diagnostic/table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/evaluation/raw"
HERE = Path(__file__).resolve().parent
CATEGORIES = ["leather", "screw", "toothbrush"]
VARIANTS = [
    ("coreset 0.01 -> 0.1", "coreset_0.1"),
    ("fit on all train-good (100%, not 80%)", "fit_all"),
    ("no crop (256 instead of 224)", "no_crop"),
    ("anomalib default (all three)", "anomalib"),
]


def _metrics(path: Path) -> dict[str, float]:
    metrics = json.loads(path.read_text())["metrics"]
    return {"image_auroc": metrics["image_auroc"], "aupro": metrics["aupro"]}


def main() -> None:
    baseline = {c: _metrics(RAW / f"mvtec_patchcore_{c}_seed0.json") for c in CATEGORIES}
    seed_range = {}
    for c in CATEGORIES:
        seeds = [
            _metrics(RAW / f"mvtec_patchcore_{c}_seed{s}.json")["image_auroc"] for s in (0, 1, 2)
        ]
        seed_range[c] = {"image_auroc_min": min(seeds), "image_auroc_max": max(seeds)}

    lines = [
        "| Setting changed | " + " | ".join(f"{c}: image AUROC / AUPRO" for c in CATEGORIES) + " |",
        "|---|" + "---|" * len(CATEGORIES),
        "| ours (baseline, seed 0) | "
        + " | ".join(
            f"{baseline[c]['image_auroc']:.4f} / {baseline[c]['aupro']:.4f}" for c in CATEGORIES
        )
        + " |",
        "| ours (baseline, image AUROC range over seeds 0-2) | "
        + " | ".join(
            f"{seed_range[c]['image_auroc_min']:.4f} to {seed_range[c]['image_auroc_max']:.4f}"
            for c in CATEGORIES
        )
        + " |",
    ]
    variants: dict[str, dict[str, dict[str, float]]] = {}
    for label, name in VARIANTS:
        row = []
        variants[name] = {}
        for c in CATEGORIES:
            v = _metrics(HERE / name / f"mvtec_patchcore_{c}_seed0.json")
            variants[name][c] = v
            d_img = v["image_auroc"] - baseline[c]["image_auroc"]
            d_pro = v["aupro"] - baseline[c]["aupro"]
            row.append(f"{v['image_auroc']:.4f} ({d_img:+.4f}) / {v['aupro']:.4f} ({d_pro:+.4f})")
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    table = "\n".join(lines) + "\n"
    (HERE / "table.md").write_text(table)
    summary = {"baseline_seed0": baseline, "baseline_seed_range": seed_range, "variants": variants}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table)


if __name__ == "__main__":
    main()
