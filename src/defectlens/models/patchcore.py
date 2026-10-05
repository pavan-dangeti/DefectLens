"""PatchCore — primary method (ADR-0003). Paper: arXiv:2106.08265."""

from __future__ import annotations

import torch
from torch import nn

from defectlens.models.coreset import greedy_coreset_indices
from defectlens.models.features import FeatureExtractor

_LAYERS = ("layer2", "layer3")  # paper default, both backbones


class PatchCore(nn.Module):
    """Training-free memory-bank model.

    ``fit`` extracts patch embeddings for every nominal training image, stacks them into a
    (N*H*W, D) bank, and keeps a greedy k-centre coreset of ``coreset_ratio`` r. Scoring finds
    the nearest coreset neighbour of each patch, then reweights the map (paper Eq. 7) using the
    local density of the memory bank around the nearest neighbour — a patch whose neighbour has
    few supporting samples is suspicious.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        pretrained: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(backbone, _LAYERS, pretrained=pretrained)
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.seed = seed
        self._coreset: torch.Tensor | None = None

    def _bank_for(self, images: torch.Tensor) -> torch.Tensor:
        return self.extractor.flatten(self.extractor.embeddings(images))

    def fit(self, images: torch.Tensor) -> None:
        with torch.no_grad():
            bank_full = self._bank_for(images)
            idx = greedy_coreset_indices(bank_full, self.coreset_ratio, seed=self.seed)
            self._coreset = bank_full[idx]

    @staticmethod
    def _nearest(embedding: torch.Tensor, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = torch.cdist(embedding, bank)
        return distances.min(dim=1)

    def _score_image(
        self, embedding: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """embedding: (H*W, D) patch rows. Returns (scalar score, (1,1,H,W) map)."""
        assert self._coreset is not None
        return self._score_image_given_bank(embedding, height, width, self._coreset)

    def _score_image_given_bank(
        self, embedding: torch.Tensor, height: int, width: int, bank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_scores, locations = self._nearest(embedding, bank)
        max_idx = int(patch_scores.argmax())
        nearest_coreset = bank[locations[max_idx]]
        n_support = min(self.num_neighbors, bank.shape[0])
        _, support_idx = torch.cdist(nearest_coreset.unsqueeze(0), bank).topk(
            n_support, largest=False
        )
        support = bank[support_idx.squeeze(0)]
        query = embedding[max_idx].unsqueeze(0)
        distances = torch.cdist(query, support).squeeze(0)
        weights = 1 - torch.softmax(distances, dim=0)
        image_score = weights[0] * patch_scores[max_idx]
        anomaly_map = patch_scores.reshape(1, 1, height, width)
        return image_score, anomaly_map

    def score_with_bank(
        self, bank: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict using an externally supplied memory bank (parity check, ADR-0004)."""
        with torch.no_grad():
            embedding = self.extractor.embeddings(images)
            batch, _, height, width = embedding.shape
            flat = self.extractor.flatten(embedding)
            per_image = height * width
            scores = []
            maps = []
            for b in range(batch):
                score, amap = self._score_image_given_bank(
                    flat[b * per_image : (b + 1) * per_image], height, width, bank
                )
                scores.append(score)
                maps.append(amap)
            return torch.stack(scores), torch.cat(maps, dim=0)

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._coreset is not None, "call fit() before predict()"
        with torch.no_grad():
            embedding = self.extractor.embeddings(images)
            batch, _, height, width = embedding.shape
            flat = self.extractor.flatten(embedding)
            per_image = height * width
            scores: list[torch.Tensor] = []
            maps: list[torch.Tensor] = []
            for b in range(batch):
                score, amap = self._score_image(
                    flat[b * per_image : (b + 1) * per_image], height, width
                )
                scores.append(score)
                maps.append(amap)
            return torch.stack(scores), torch.cat(maps, dim=0)
