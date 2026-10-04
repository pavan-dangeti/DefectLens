"""PaDiM baseline (ADR-0003). Paper: arXiv:2011.08785.

Per-patch-position multivariate Gaussians in the concatenated multi-layer embedding space,
covariance shrunk by ε·I (paper default 0.01). Score is the Mahalanobis distance at each
patch position.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from defectlens.models.features import FeatureExtractor

_PADIM_DEFAULTS = {"resnet18": 100, "wide_resnet50_2": 550}
_PADIM_LAYERS = {
    "resnet18": ("layer1", "layer2", "layer3"),
    "wide_resnet50_2": ("layer1", "layer2", "layer3"),
}


class PaDiM(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        n_features: int | None = None,
        epsilon: float = 0.01,
        pretrained: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if backbone not in _PADIM_LAYERS:
            raise ValueError(f"unknown backbone {backbone!r}")
        self.layers = _PADIM_LAYERS[backbone]
        self.extractor = FeatureExtractor(backbone, self.layers, pretrained=pretrained, pool=False)
        self.n_features = n_features if n_features is not None else _PADIM_DEFAULTS[backbone]
        self.epsilon = epsilon
        generator = torch.Generator().manual_seed(seed)
        self._idx_sel: torch.Tensor = torch.randperm(self._total_dim(), generator=generator)[
            : self.n_features
        ]
        self._mean: torch.Tensor | None = None
        self._inv_cov: torch.Tensor | None = None

    def _total_dim(self) -> int:
        layer_idx = {"layer1": 1, "layer2": 2, "layer3": 3}
        info = cast(list[dict[str, Any]], self.extractor.model.feature_info)
        return sum(info[layer_idx[layer]]["num_chs"] for layer in self.layers)

    def _embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self.extractor.embeddings(images, interp="nearest")

    def _select(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding.index_select(1, self._idx_sel.to(embedding.device))

    def fit(self, images: torch.Tensor) -> None:
        embedding = self._embedding(images)
        selected = self._select(embedding)  # (B, F, H, W)
        batch, features, height, width = selected.shape
        x = selected.permute(0, 2, 3, 1).reshape(batch, height * width, features)
        mean = x.mean(dim=0)  # (H*W, F)
        centered = x - mean.unsqueeze(0)
        cov = torch.einsum("bpf,bpe->pfe", centered, centered) / max(batch - 1, 1)
        cov = cov + self.epsilon * torch.eye(features, device=cov.device, dtype=cov.dtype)
        self._mean = mean  # (H*W, F)
        self._inv_cov = torch.linalg.inv(cov)  # (H*W, F, F)

    def score(self, image_embedding: torch.Tensor) -> torch.Tensor:
        """One image, (1, D', H, W). Returns (1, 1, H, W) Mahalanobis distance map."""
        assert self._mean is not None and self._inv_cov is not None
        selected = self._select(image_embedding)
        _, features, height, width = selected.shape
        flat = selected.permute(2, 3, 0, 1).reshape(height * width, 1, features)
        q = flat.squeeze(1)  # (H*W, F)
        centered = q - self._mean.to(q.dtype)  # (H*W, F)
        mahal = torch.einsum("pf,pfe,pe->p", centered, self._inv_cov, centered)
        return mahal.clamp_min(0).sqrt().reshape(1, 1, height, width)

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._mean is not None and self._inv_cov is not None, "call fit() first"
        with torch.no_grad():
            embedding = self._embedding(images)
            maps = torch.cat([self.score(embedding[i : i + 1]) for i in range(embedding.shape[0])])
            scores = maps.amax(dim=(-2, -1)).squeeze(1)
            return scores, maps
