"""Render the static INT8 comparison from the committed results of int8_static.py.

Accuracy is compared against the committed fp32 raw results at coreset 0.01, seed 0, which come from the
same evaluate_category protocol. Latency and embedding agreement come from summary.json.

    uv run python results/optimisation/int8_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "int8_static"


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text())
    speed = {(c["dataset"], c["category"]): c for c in summary["categories"]}
    lines = [
        "| Category | Image AUROC fp32 → INT8 | AUPRO fp32 → INT8 | Pixel AUROC fp32 → INT8 | "
        "Escape at 1% FRR fp32 → INT8 | Backbone latency fp32 → INT8 (ms) | Feature cosine |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, category in [("mvtec", "screw"), ("mvtec", "toothbrush"), ("visa", "capsules")]:
        record = json.loads((RESULTS / f"{dataset}_{category}.json").read_text())
        fp32, int8 = record["fp32_committed"], record["int8"]
        timing = speed[(dataset, category)]
        lines.append(
            f"| {dataset}/{category} "
            f"| {fp32['image_auroc']:.3f} → {int8['image_auroc']:.3f} "
            f"| {fp32['aupro']:.3f} → {int8['aupro']:.3f} "
            f"| {fp32['pixel_auroc']:.3f} → {int8['pixel_auroc']:.3f} "
            f"| {fp32['escape_rate_at_target_frr']:.2f} → {int8['escape_rate_at_target_frr']:.2f} "
            f"| {timing['latency_ms_fp32_median']:.1f} → {timing['latency_ms_int8_median']:.1f} "
            f"| {timing['embedding_cosine_mean_16_images']:.3f} |"
        )
    table = "\n".join(lines) + "\n"
    (HERE / "int8_table.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
