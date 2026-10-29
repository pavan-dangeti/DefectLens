"""Diagnostic: which PatchCore setting explains the gap to the anomalib default on MVTec AD.

Each variant changes settings relative to our configuration (coreset 0.01, fit on 80% of train-good,
resize 256 then centre crop 224). The one-at-a-time variants change a single setting; the anomalib
variant changes all of them to match anomalib's PatchCore defaults.

Fitting on all train-good images leaves no held-out slice to calibrate a threshold on, so these
variants report threshold-free metrics only (image AUROC, pixel AUROC, AUPRO).

    uv run python results/evaluation/diagnostic/run_diagnostic.py --category screw --variant anomalib
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from defectlens.eval.benchmark import (
    _aupro,
    _fit_calibration_splits,
    _image_auroc,
    _make_model,
    _pixelwise_auroc,
    _split,
)

ROOT = Path("data")
OUT = Path("results/evaluation/diagnostic")

# coreset: greedy k-centre ratio. fit_all: fit on every train-good image rather than 80% of them.
# crop: centre-crop size after the resize to `resize`; None means no crop.
VARIANTS: dict[str, dict[str, Any]] = {
    "coreset_0.1": {"coreset_ratio": 0.1, "fit_all": False, "resize": 256, "crop": 224},
    "fit_all": {"coreset_ratio": 0.01, "fit_all": True, "resize": 256, "crop": 224},
    "no_crop": {"coreset_ratio": 0.01, "fit_all": False, "resize": 256, "crop": None},
    "anomalib": {"coreset_ratio": 0.1, "fit_all": True, "resize": 256, "crop": None},
}


def _images(paths: list[Path], resize: int, crop: int | None) -> torch.Tensor:
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
        if crop is not None:
            img = TF.center_crop(img, crop)
        t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        tensors.append(TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return torch.stack(tensors)


def _masks(paths: list[Path], resize: int, crop: int | None) -> torch.Tensor:
    masks = []
    for p in paths:
        img = Image.open(p).resize((resize, resize), Image.Resampling.NEAREST)
        if crop is not None:
            img = TF.center_crop(img, crop)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append(torch.from_numpy((arr > 0).astype(np.uint8)))
    return torch.stack(masks)


def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def run(category: str, variant: str, seed: int) -> Path:
    cfg = VARIANTS[variant]
    path = OUT / variant / f"mvtec_patchcore_{category}_seed{seed}.json"
    if path.exists():
        print(f"skip {path.name} (exists)", flush=True)
        return path
    split = _split(ROOT, category)
    if cfg["fit_all"]:
        fit = split["train_good"]
    else:
        fit, _, _ = _fit_calibration_splits(split["train_good"], seed=seed)
    resize, crop = cfg["resize"], cfg["crop"]

    started = time.monotonic()
    model = _make_model("patchcore", cfg["coreset_ratio"], seed, pretrained=True)
    model.fit(_images(fit, resize, crop))
    good_scores, _ = model.predict(_images(split["test_good"], resize, crop))
    bad_scores, bad_maps = model.predict(_images(split["test_bad"], resize, crop))
    gt = _masks(split["test_gt"], resize, crop)

    metrics = {
        "image_auroc": _image_auroc(good_scores.detach().numpy(), bad_scores.detach().numpy()),
        "pixel_auroc": _pixelwise_auroc(gt.numpy(), bad_maps[:, 0].detach().numpy()),
        "aupro": _aupro(gt, bad_maps),
    }
    record = {
        "category": category,
        "variant": variant,
        "config": {
            **cfg,
            "seed": seed,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
        },
        "n_fit": len(fit),
        "n_test_good": len(split["test_good"]),
        "n_test_bad": len(split["test_bad"]),
        "metrics": metrics,
        "wall_seconds": round(time.monotonic() - started, 1),
        "git_commit": _git_commit(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["metrics"]), f"{variant} {category}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--category", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.category, args.variant, args.seed)


if __name__ == "__main__":
    main()
