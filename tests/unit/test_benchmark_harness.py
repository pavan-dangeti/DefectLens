from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from defectlens.eval.benchmark import evaluate_category


def test_evaluate_category_canonical_layout(tmp_path: Path) -> None:
    """Synthetic MVTec-AD-style layout; harness must execute and report all keys."""
    base = tmp_path / "bottle"
    (base / "train" / "good").mkdir(parents=True)
    (base / "test" / "good").mkdir(parents=True)
    (base / "test" / "broken").mkdir(parents=True)
    (base / "ground_truth" / "broken").mkdir(parents=True)

    rng = np.random.default_rng(0)
    for i in range(8):
        img = rng.integers(60, 200, size=(96, 96, 3), dtype=np.uint8)
        Image.fromarray(img).save(base / "train" / "good" / f"{i}.png")
    for i in range(3):
        Image.fromarray(rng.integers(60, 200, size=(96, 96, 3), dtype=np.uint8)).save(
            base / "test" / "good" / f"{i}.png"
        )
    for i in range(2):
        broken = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
        Image.fromarray(broken).save(base / "test" / "broken" / f"{i}.png")
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        Image.fromarray(mask).save(base / "ground_truth" / "broken" / f"{i}_mask.png")

    result, _samples = evaluate_category(
        tmp_path,
        "bottle",
        method="padim",
        few_shot_counts=(),
        seed=0,
        pretrained=False,
    )
    assert set(result) >= {
        "category",
        "image_auroc",
        "pixel_auroc",
        "aupro",
        "escape_rate_at_target_frr",
        "frr_at_calibration_target",
        "few_shot_image_auroc",
    }
    assert result["category"] == "bottle"
    assert result["n_fit"] + result["n_cal"] + result["n_holdout"] == 8
    assert 0.0 <= result["image_auroc"] <= 1.0
    assert 0.0 <= result["escape_rate_at_target_frr"] <= 1.0
