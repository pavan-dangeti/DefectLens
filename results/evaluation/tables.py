"""Render the README result tables from the committed three-seed summary.

Reads results/evaluation/summary.json (written by aggregate.py) and writes results/evaluation/tables.md.
Every number in the README's results section is copied from that file, so it can be traced to the raw
runs under results/evaluation/raw/.

    uv run python results/evaluation/tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "summary.json"
OUT = HERE / "tables.md"
METRICS = [
    ("image_auroc", "Image AUROC"),
    ("pixel_auroc", "Pixel AUROC"),
    ("aupro", "AUPRO"),
    ("escape_rate_at_target_frr", "Escape rate at 1% FRR"),
]


def _cell(stat: dict[str, Any]) -> str:
    return f"{stat['mean']:.3f} ± {stat['ci95_half_width']:.3f}"


def _dataset_table(summary: dict[str, Any]) -> list[str]:
    lines = ["| Dataset | Categories | " + " | ".join(label for _, label in METRICS) + " |"]
    lines.append("|---|---|" + "---|" * len(METRICS))
    for key, label in (("mvtec/patchcore", "MVTec AD"), ("visa/patchcore", "VisA")):
        block = summary["datasets"][key]
        means = block["mean_over_categories"]
        cells = [_cell(means[name]) for name, _ in METRICS]
        lines.append(f"| {label} | {block['n_categories']} | " + " | ".join(cells) + " |")
    return lines


def _category_table(summary: dict[str, Any], prefix: str) -> list[str]:
    rows = sorted(
        (key.split("/")[-1], value)
        for key, value in summary["categories"].items()
        if key.startswith(prefix)
    )
    lines = ["| Category | Image AUROC | AUPRO | Escape rate at 1% FRR |", "|---|---|---|---|"]
    for name, value in rows:
        lines.append(
            f"| {name} | {_cell(value['image_auroc'])} | {_cell(value['aupro'])} "
            f"| {_cell(value['escape_rate_at_target_frr'])} |"
        )
    return lines


def _few_shot_table(summary: dict[str, Any], prefix: str) -> list[str]:
    counts = ["1", "5", "10", "50"]
    per_category = [
        value["few_shot_image_auroc"]
        for key, value in summary["categories"].items()
        if key.startswith(prefix)
    ]
    cells = []
    for k in counts:
        means = [category[k]["mean"] for category in per_category if k in category]
        cells.append(f"{sum(means) / len(means):.3f}")
    return [
        "| Good images used to fit | " + " | ".join(counts) + " |",
        "|---|" + "---|" * len(counts),
        "| Mean image AUROC over categories | " + " | ".join(cells) + " |",
    ]


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    sections = [
        "## Dataset means (3 seeds, 95% t-interval across seeds)",
        "",
        *_dataset_table(summary),
        "",
        "## MVTec AD by category",
        "",
        *_category_table(summary, "mvtec/"),
        "",
        "## VisA by category",
        "",
        *_category_table(summary, "visa/"),
        "",
        "## Few-shot: MVTec AD",
        "",
        *_few_shot_table(summary, "mvtec/"),
        "",
        "## Few-shot: VisA",
        "",
        *_few_shot_table(summary, "visa/"),
    ]
    text = "\n".join(sections) + "\n"
    OUT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
