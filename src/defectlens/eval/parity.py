"""Parity gate (SLO-7): my PatchCore must agree with anomalib's PatchCore on the same images.

Three checks, in increasing rigour of my implementation rather than of anomalib's:

1. Scoring math — anomalib's `memory_bank` scored by my `score_with_bank` must be
   numerically allclose to anomalib's own `pred_score`. Isolates my NN + reweighting code;
   any coreset-selection difference is excluded by construction.
2. Full pipeline — anomalib's scores vs my `PatchCore.fit`+`predict` end to end. Must agree
   to within coreset-selection noise (different random sparse projection + start index).
3. Decision quality — both should classify defect vs good consistently (AUROC delta small).

Uses skimage data so the check is self-contained and does not need the 1.9 GB VisA tarball on
each developer machine. The tarball is exercised on Kaggle + Oracle, not on every laptop.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def _preprocess(pil_image: Image.Image) -> torch.Tensor:
    tf = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = tf(pil_image)
    assert isinstance(tensor, torch.Tensor)
    return tensor


def _make_anomaly(image: torch.Tensor, patch_size: int = 24) -> torch.Tensor:
    """Square grey patch, a simple but genuine defect."""
    out = image.clone()
    h = (out.shape[-2] - patch_size) // 2
    w = (out.shape[-1] - patch_size) // 2
    out[:, h : h + patch_size, w : w + patch_size] = 1.5
    return out


def _load_skimage_images() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Returns (good, anomalous) preprocessed tensors with matching anomaly count."""
    from skimage import data

    names = ["brick", "chelsea", "coffee", "camera", "astronaut", "moon", "coins", "cat"]
    good: list[torch.Tensor] = []
    anomalous: list[torch.Tensor] = []
    for name in names:
        raw = getattr(data, name)()
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        pic = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
        tensor = _preprocess(pic)
        good.append(tensor)
        anomalous.append(_make_anomaly(tensor))
    return good, anomalous


def run_parity(
    backbone: str = "wide_resnet50_2",
    num_neighbors: int = 9,
    coreset_ratio: float = 0.1,
    pre_fit_images: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    from defectlens.models.patchcore import PatchCore

    good, anomalous = _load_skimage_images()
    fit_images = pre_fit_images if pre_fit_images is not None else good[:4]
    good_test_images = torch.stack(good[4:])
    anomaly_test_images = torch.stack(anomalous[4:])

    torch.manual_seed(0)
    anomalib_model = PatchcoreModel(
        layers=["layer2", "layer3"],
        backbone=backbone,
        pre_trained=True,
        num_neighbors=num_neighbors,
    )
    anomalib_model.train()
    with torch.no_grad():
        for img in fit_images:
            anomalib_model(img.unsqueeze(0))
    anomalib_model.subsample_embedding(sampling_ratio=coreset_ratio)
    anomalib_model.eval()
    with torch.no_grad():
        good_scores_a = anomalib_model(good_test_images).pred_score.detach().cpu()
        bad_scores_a = anomalib_model(anomaly_test_images).pred_score.detach().cpu()

    my_model = PatchCore(
        backbone=backbone, num_neighbors=num_neighbors, coreset_ratio=coreset_ratio, seed=0
    )
    my_model.fit(torch.stack(fit_images))
    with torch.no_grad():
        good_scores_m = my_model.predict(good_test_images)[0].detach().cpu()
        bad_scores_m = my_model.predict(anomaly_test_images)[0].detach().cpu()

    anomalib_bank = anomalib_model.memory_bank
    with torch.no_grad():
        bank_scores = (
            my_model.score_with_bank(
                anomalib_bank, torch.cat([good_test_images, anomaly_test_images], dim=0)
            )[0]
            .detach()
            .cpu()
        )

    labels = torch.tensor([0] * len(good_test_images) + [1] * len(anomaly_test_images))
    auroc_a = _binary_auroc(torch.cat([good_scores_a, bad_scores_a]), labels)
    auroc_m = _binary_auroc(torch.cat([good_scores_m, bad_scores_m]), labels)
    auroc_bank = _binary_auroc(bank_scores, labels)

    reference = torch.cat([good_scores_a, bad_scores_a])
    return {
        "anomalib_score_auroc": float(auroc_a),
        "mine_score_auroc": float(auroc_m),
        "mine_on_anomalib_bank_auroc": float(auroc_bank),
        "max_abs_score_diff_vs_anomalib": float((bank_scores - reference).abs().max()),
        "mean_abs_score_diff_vs_anomalib": float((bank_scores - reference).abs().mean()),
    }


def _binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney AUROC; higher score must mean more anomalous."""
    n = scores.numel()
    order = torch.argsort(scores)  # ascending
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float32)
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    u = ranks[labels == 1].sum() - positives * (positives + 1) / 2
    return float(u / (positives * negatives))
