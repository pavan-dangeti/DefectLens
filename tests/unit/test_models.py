"""My implementation's scoring must agree with anomalib's scoring math."""

from __future__ import annotations

import torch

from defectlens.models.coreset import greedy_coreset_indices
from defectlens.models.patchcore import PatchCore


def test_coreset_is_k_centre_greedy_shape() -> None:
    rng = torch.Generator().manual_seed(0)
    bank = torch.randn(200, 32, generator=rng)
    idx = greedy_coreset_indices(bank, 0.1, seed=1)
    assert len(idx) == 20
    assert len(set(idx)) == 20  # no duplicates


def test_coreset_deterministic_per_seed() -> None:
    rng = torch.Generator().manual_seed(1)
    bank = torch.randn(120, 8, generator=rng)
    assert greedy_coreset_indices(bank, 0.25, seed=7) == greedy_coreset_indices(bank, 0.25, seed=7)


def test_patchcore_scores_match_anomalib_math() -> None:
    torch.manual_seed(0)
    try:
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    except Exception:
        import pytest

        pytest.skip("anomalib not available")

    bank = torch.randn(60, 64)
    model = PatchcoreModel(
        layers=["layer2", "layer3"], backbone="wide_resnet50_2", pre_trained=False
    )
    model.memory_bank = bank
    model.eval()

    num_images, patches, dim = 2, 16, 64
    embedding = torch.randn(num_images * patches, dim)
    distances = torch.cdist(embedding, bank)
    patch_scores, locations = distances.min(dim=1)
    patch_scores_img = patch_scores.reshape(num_images, patches)
    locations_img = locations.reshape(num_images, patches)
    scores_anomalib = model.compute_anomaly_score(patch_scores_img, locations_img, embedding)

    mine = PatchCore.__new__(PatchCore)
    torch.nn.Module.__init__(mine)
    mine._coreset = bank
    mine.num_neighbors = 9
    scores_mine = []
    for b in range(num_images):
        embedding_b = embedding[b * patches : (b + 1) * patches]
        s, _ = mine._score_image(embedding_b, 4, 4)
        scores_mine.append(s)
    scores_mine_t = torch.stack(scores_mine)
    assert torch.allclose(scores_mine_t, scores_anomalib, atol=1e-5)


def test_patchcore_fit_and_predict_shapes() -> None:
    pc = PatchCore(backbone="resnet18", coreset_ratio=0.2, pretrained=False)
    images = torch.rand(2, 3, 224, 224)
    pc.fit(images)
    assert pc._coreset is not None
    assert pc._coreset.shape[1] == 128 + 256
    scores, maps = pc.predict(torch.rand(2, 3, 224, 224))
    assert scores.shape == (2,)
    assert maps.shape[0] == 2 and maps.shape[1] == 1
    # score_with_bank against the same coreset must reproduce predict()'s scores
    images = torch.rand(2, 3, 224, 224)
    scores2, maps2 = pc.score_with_bank(pc._coreset, images)
    scores3, maps3 = pc.predict(images)
    assert torch.allclose(scores2, scores3)
    # score_with_bank returns raw 28x28 maps; predict upsamples to input resolution
    assert maps2.shape[-2:] == (28, 28)
    assert maps3.shape[-2:] == (224, 224)


def test_padim_shapes() -> None:
    import defectlens.models.padim as padim_mod

    model = padim_mod.PaDiM(backbone="resnet18", pretrained=False)
    images = torch.rand(2, 3, 224, 224)
    model.fit(images)
    scores, maps = model.predict(torch.rand(3, 3, 224, 224))
    assert scores.shape == (3,)
    assert maps.shape[0] == 3
