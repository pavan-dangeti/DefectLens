from __future__ import annotations

import numpy as np
import torch

from defectlens.eval.benchmark import _aupro, _image_auroc, _pixelwise_auroc


def test_aupro_perfect_detection_is_high() -> None:
    gt = np.zeros((1, 32, 32), dtype=np.uint8)
    gt[0, 4:12, 4:12] = 1
    maps = np.zeros((1, 32, 32), dtype=np.float64)
    maps[0, 4:12, 4:12] = 0.9
    score = _aupro(torch.from_numpy(gt), torch.from_numpy(maps))
    assert 0.7 < score <= 1.0


def test_aupro_near_zero_detection_is_low() -> None:
    gt = np.zeros((1, 32, 32), dtype=np.uint8)
    gt[0, 4:12, 4:12] = 1
    maps = np.full((1, 32, 32), 0.5)
    maps[0, 0:4, 0:4] = 0.9  # high in the wrong place
    score = _aupro(torch.from_numpy(gt), torch.from_numpy(maps))
    assert score < 0.3


def test_image_auroc_perfect_task() -> None:
    assert _image_auroc(np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.9, 1.0])) == 1.0


def test_pixelwise_auroc_perfect_task() -> None:
    gt = np.zeros((2, 32, 32), dtype=np.uint8)
    gt[0, 0:4, 0:4] = 1
    scores = np.zeros((2, 32, 32), dtype=np.float64)
    scores[0, 0:4, 0:4] = 0.9
    assert _pixelwise_auroc(gt, scores) == 1.0
