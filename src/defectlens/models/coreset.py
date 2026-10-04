"""Greedy k-centre coreset subsampling (paper section 3.2; anomalib's KCenterGreedy equivalent).

The full 1536-d feature rows are too large to run greedy selection across the whole bank on a
laptop, so selection runs in a sparse-random-projection space with the same Johnson-Lindenstrauss
dimension as anomalib (n_components ~ 4*ln(n) / (eps^2/2 - eps^3/3), eps=0.9). The chosen vectors
come from the *original* feature bank: the projection only guides which ones to keep, never the
distances used at scoring time.
"""

from __future__ import annotations

import numpy as np
import torch

DEFAULT_EPS = 0.9


class SparseRandomProjector:
    """Johnson-Lindenstrauss sparse random projection (Li et al., KDD 2006), seedable."""

    def __init__(self, eps: float = DEFAULT_EPS, seed: int | None = None) -> None:
        self.eps = eps
        self.seed = seed
        self.n_components: int = 0
        self.matrix: torch.Tensor = torch.empty(0)

    def _project_dim(self, n_samples: int) -> int:
        denominator = (self.eps**2 / 2) - (self.eps**3 / 3)
        return int(np.log(n_samples) * 4 / denominator)

    def _sparse_matrix(
        self, n_features: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        import sklearn.utils.random as rnd

        density = 1.0 / np.sqrt(n_features)
        rng = np.random.default_rng(self.seed)
        components = torch.zeros((self.n_components, n_features), dtype=torch.float32)
        for i in range(self.n_components):
            nnz = int(rng.binomial(n_features, density))
            cols = rnd.sample_without_replacement(
                n_population=n_features, n_samples=nnz, random_state=self.seed
            )
            data = torch.tensor(rng.binomial(1, 0.5, size=nnz) * 2 - 1, dtype=torch.float32)
            components[i, torch.as_tensor(cols, dtype=torch.int64)] = data
        components *= np.sqrt(1.0 / density) / np.sqrt(self.n_components)
        return components.to(device=device, dtype=dtype)

    def fit(self, embedding: torch.Tensor) -> SparseRandomProjector:
        self.n_components = self._project_dim(embedding.shape[0])
        self.matrix = self._sparse_matrix(embedding.shape[1], embedding.device, embedding.dtype)
        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding @ self.matrix.T


def greedy_coreset_indices(
    embedding: torch.Tensor, ratio: float, seed: int | None = None
) -> list[int]:
    """Indices of a k-centre-greedy coreset of ``embedding`` of size ``int(n*ratio)``."""
    n = embedding.shape[0]
    coreset_size = int(n * ratio)
    if coreset_size <= 0:
        raise ValueError(f"coreset of ratio {ratio} over {n} rows is empty")
    projector = SparseRandomProjector(seed=seed)
    projector.fit(embedding)
    projected = projector.transform(embedding)

    generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    start = int(torch.randint(n, (1,), generator=generator).item())
    selected = [start]

    # min-distance of every point to the *current* selection
    min_distances = torch.full((n,), float("inf"), dtype=embedding.dtype)
    for _ in range(coreset_size - 1):
        last = projected[selected[-1]]
        distances = torch.linalg.norm(projected - last, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        next_idx = int(min_distances.argmax().item())
        min_distances[next_idx] = 0.0
        selected.append(next_idx)
    return selected


def select_coreset(embedding: torch.Tensor, ratio: float, seed: int | None = None) -> torch.Tensor:
    idx = greedy_coreset_indices(embedding, ratio, seed=seed)
    return embedding[idx]
