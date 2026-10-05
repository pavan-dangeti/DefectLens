"""Phase 3: benchmark harness — full-category MVTec / VisA evaluation.

Canonical layout assumed:
    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect>/*.png
    <root>/<category>/ground_truth/<defect>/*.png
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score

VALID_EXT = {".png", ".jpg", ".jpeg"}


def _split(root: Path, category: str) -> dict[str, list[Path]]:
    base = root / category
    train_good = sorted(
        p for p in (base / "train" / "good").iterdir() if p.suffix.lower() in VALID_EXT
    )
    test_good = sorted(
        p for p in (base / "test" / "good").iterdir() if p.suffix.lower() in VALID_EXT
    )
    defects = sorted(p.name for p in (base / "test").iterdir() if p.is_dir() and p.name != "good")
    test_bad: list[Path] = []
    test_gt: list[Path] = []
    for d in defects:
        test_bad.extend(
            sorted(p for p in (base / "test" / d).iterdir() if p.suffix.lower() in VALID_EXT)
        )
        gt_dir = base / "ground_truth" / d
        if gt_dir.is_dir():
            test_gt.extend(sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in VALID_EXT))
    return {
        "train_good": train_good,
        "test_good": test_good,
        "test_bad": test_bad,
        "test_gt": test_gt,
    }


def _fit_calibration_splits(
    train_good: list[Path], seed: int = 0
) -> tuple[list[Path], list[Path], list[Path]]:
    idx = list(range(len(train_good)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_fit = max(int(0.8 * n), 1)
    n_cal = max(int(0.1 * n), 1)
    return (
        [train_good[idx[i]] for i in range(n_fit)],
        [train_good[idx[i]] for i in range(n_fit, n_fit + n_cal)],
        [train_good[idx[i]] for i in range(n_fit + n_cal, n)],
    )


def _preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    img = TF.center_crop(img, 224)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    assert isinstance(t, torch.Tensor)
    return t


def _to_tensors(paths: list[Path]) -> torch.Tensor:
    return torch.stack([_preprocess(Image.open(p)) for p in paths])


def _to_masks(paths: list[Path]) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).resize((256, 256), Image.Resampling.NEAREST)
        img = TF.center_crop(img, 224)
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[..., 0]
        masks.append(torch.from_numpy((arr > 0).astype(np.uint8)))
    return torch.stack(masks)


def _image_auroc(good_scores: np.ndarray, bad_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(good_scores), dtype=np.uint8), np.ones(len(bad_scores), dtype=np.uint8)]
    )
    scores = np.concatenate([good_scores, bad_scores])
    return float(roc_auc_score(labels, scores))


def _pixelwise_auroc(gt: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(gt.ravel() > 0, scores.ravel()))


def _aupro(
    gt: torch.Tensor | np.ndarray,
    maps: torch.Tensor | np.ndarray,
    fpr_limit: float = 0.3,
    n_thresholds: int = 50,
) -> float:
    """Area under the per-region overlap curve, FP-limited to ``fpr_limit`` (Bergmann et al.,
    "Uninformed Students" CVPR 2020). Mirrors anomalib.metrics.AUPRO but avoids the
    Batch-wrapper ceremony and imports kornia's connected components via scipy."""
    import scipy.ndimage

    gt_arr = np.asarray(gt, dtype=np.uint8).astype(bool)
    score_arr = np.asarray(maps, dtype=np.float64)
    if score_arr.ndim == 4:
        score_arr = score_arr[:, 0]
    gt_arr = gt_arr.astype(np.uint8)  # label regions on binary ground truth

    region_labels = np.zeros_like(gt_arr, dtype=np.int64)
    r_id = 0
    for i in range(gt_arr.shape[0]):
        labeled, _ = scipy.ndimage.label(gt_arr[i])
        for r in range(1, int(labeled.max()) + 1):
            r_id += 1
            region_labels[i][labeled == r] = r_id
    num_regions = max(int(region_labels.max()), 0)
    num_bg = int((region_labels == 0).sum())
    if num_regions == 0 or num_bg == 0:
        return float("nan")

    region_sizes = (
        torch.bincount(torch.from_numpy(region_labels.ravel()).long(), minlength=num_regions + 1)
        .numpy()
        .astype(np.float64)
    )
    scores = np.unique(score_arr)
    thresholds = np.linspace(scores.max(), scores.min(), n_thresholds)
    pts: list[tuple[float, float]] = []
    for t in thresholds:
        binary = score_arr >= t
        fp = int((binary & (region_labels == 0)).sum())
        fpr = fp / max(num_bg, 1)
        if fpr > fpr_limit and pts:
            break
        if fpr > fpr_limit:
            continue
        covered_labels = region_labels[binary & (region_labels > 0)].ravel()
        covered = np.bincount(covered_labels, minlength=num_regions + 1).astype(np.float64)[1:]
        per_region_overlap = covered / np.maximum(region_sizes[1:], 1.0)
        pts.append((fpr, float(per_region_overlap.mean())))
    pts.sort(key=lambda p: p[0])
    if not pts:
        return float("nan")
    fprs = [p[0] for p in pts]
    pros = [p[1] for p in pts]
    grid = np.linspace(0.0, fpr_limit, 64)
    pro = np.interp(grid, fprs, pros)
    return float(np.trapezoid(pro, grid) / fpr_limit)


class _Fittable(Protocol):
    def fit(self, images: torch.Tensor) -> None: ...
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


def _make_model(method: str, coreset_ratio: float, seed: int, pretrained: bool = True) -> _Fittable:
    if method == "patchcore":
        from defectlens.models.patchcore import PatchCore

        return PatchCore(
            backbone="wide_resnet50_2",
            coreset_ratio=coreset_ratio,
            num_neighbors=9,
            seed=seed,
            pretrained=pretrained,
        )
    if method == "padim":
        from defectlens.models.padim import PaDiM

        return PaDiM(backbone="resnet18", seed=seed, pretrained=pretrained)
    raise ValueError(method)


def evaluate_category(
    root: Path,
    category: str,
    method: str = "patchcore",
    coreset_ratio: float = 0.01,
    target_frr: float = 0.01,
    few_shot_counts: tuple[int, ...] = (1, 5, 10, 50),
    seed: int = 0,
    pretrained: bool = True,
) -> tuple[dict[str, object], list[float]]:
    split = _split(root, category)
    fit, cal, holdout = _fit_calibration_splits(split["train_good"], seed=seed)

    fit_tensor = _to_tensors(fit)
    cal_tensor = _to_tensors(cal)
    holdout_tensor = _to_tensors(holdout)
    test_good_tensor = _to_tensors(split["test_good"])
    test_bad_tensor = _to_tensors(split["test_bad"])
    gt_masks = _to_masks(split["test_gt"])

    model = _make_model(method, coreset_ratio, seed, pretrained=pretrained)
    model.fit(fit_tensor)

    cal_scores, _ = model.predict(cal_tensor)
    tau = float(np.quantile(cal_scores.detach().numpy().astype(np.float64), 1.0 - target_frr))

    holdout_scores, _ = model.predict(holdout_tensor)
    frr_holdout = float((holdout_scores.detach().numpy() > tau).mean())

    good_scores, _ = model.predict(test_good_tensor)
    bad_scores, bad_maps = model.predict(test_bad_tensor)

    image_auroc = _image_auroc(good_scores.detach().numpy(), bad_scores.detach().numpy())
    pixel_auroc = _pixelwise_auroc(gt_masks.numpy(), bad_maps[:, 0].detach().numpy())
    aupro = _aupro(gt_masks, bad_maps)
    escape = float((bad_scores.detach().numpy() <= tau).mean())

    few_shot: dict[int, float] = {}
    for k in few_shot_counts:
        if k > len(fit):
            continue
        sub = random.Random(seed).sample(fit, k)
        m = _make_model(method, coreset_ratio, seed, pretrained=pretrained)
        m.fit(_to_tensors(sub))
        g, _ = m.predict(test_good_tensor)
        b, _ = m.predict(test_bad_tensor)
        few_shot[k] = _image_auroc(g.detach().numpy(), b.detach().numpy())

    result = {
        "category": category,
        "method": method,
        "n_fit": len(fit),
        "n_cal": len(cal),
        "n_holdout": len(holdout),
        "n_test_good": len(split["test_good"]),
        "n_test_bad": len(split["test_bad"]),
        "threshold": tau,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "aupro": aupro,
        "escape_rate_at_target_frr": escape,
        "frr_at_calibration_target": frr_holdout,
        "few_shot_image_auroc": few_shot,
        "seed": seed,
    }
    auc_samples = [image_auroc]
    return result, auc_samples
