"""Artefact persistence and registry loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from defectlens.models.patchcore import PatchCore
from defectlens.serving.checksums import write_sidecar
from defectlens.serving.registry import load_registry


def _fitted() -> PatchCore:
    model = PatchCore(backbone="resnet18", coreset_ratio=0.2, pretrained=False)
    model.fit(torch.rand(1, 3, 224, 224))
    return model


def test_save_load_roundtrip_preserves_scores(tmp_path: Path) -> None:
    # pretrained=False gives a random backbone per construction, so the same seed is what makes
    # the roundtrip comparable; production artefacts use the deterministic ImageNet weights.
    torch.manual_seed(0)
    model = _fitted()
    model.save(tmp_path / "bottle.pt", category="bottle", threshold=0.42)

    torch.manual_seed(0)
    loaded = PatchCore.load(tmp_path / "bottle.pt")
    assert loaded.category == "bottle"
    assert loaded.threshold == 0.42
    assert loaded.extractor.name == "resnet18"
    assert torch.equal(loaded._coreset, model._coreset)

    probe = torch.rand(1, 3, 224, 224)
    original, _ = model.predict(probe)
    restored, _ = loaded.predict(probe)
    assert torch.allclose(original, restored)


def test_save_before_fit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        PatchCore(backbone="resnet18", pretrained=False).save(
            tmp_path / "bottle.pt", category="bottle", threshold=0.1
        )


def test_load_registry_indexes_by_category(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "bottle.pt", category="bottle", threshold=0.4)
    _fitted().save(tmp_path / "cable.pt", category="cable", threshold=0.6)
    write_sidecar(tmp_path / "bottle.pt")
    write_sidecar(tmp_path / "cable.pt")

    registry = load_registry(tmp_path)
    assert sorted(registry) == ["bottle", "cable"]
    assert registry["bottle"]["threshold"] == 0.4
    assert registry["cable"]["version"] == "cable"


def test_load_registry_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "nope")


def test_load_registry_empty_directory_is_unready(tmp_path: Path) -> None:
    assert load_registry(tmp_path) == {}


def test_load_registry_rejects_non_artifact(tmp_path: Path) -> None:
    (tmp_path / "bottle.pt").write_bytes(b"definitely not a torch archive")
    write_sidecar(tmp_path / "bottle.pt")
    with pytest.raises(ValueError, match=r"bottle\.pt is not a loadable"):
        load_registry(tmp_path)


def test_load_registry_rejects_artifact_without_category(tmp_path: Path) -> None:
    torch.save({"coreset_ratio": 0.1}, tmp_path / "bottle.pt")
    write_sidecar(tmp_path / "bottle.pt")
    with pytest.raises(ValueError, match="not a loadable"):
        load_registry(tmp_path)


def test_load_registry_refuses_artifact_without_sidecar(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "bottle.pt", category="bottle", threshold=0.4)
    with pytest.raises(ValueError, match="no checksum sidecar"):
        load_registry(tmp_path)


def test_load_registry_refuses_tampered_artifact(tmp_path: Path) -> None:
    path = tmp_path / "bottle.pt"
    _fitted().save(path, category="bottle", threshold=0.4)
    write_sidecar(path)
    # Flip one byte in place: the file still parses as a torch archive, so only the checksum
    # stands between a swapped bank and a silently wrong score.
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_registry(tmp_path)


def test_reference_scores_survive_the_roundtrip_and_enable_drift(tmp_path: Path) -> None:
    reference = torch.linspace(0.0, 1.0, 50)
    _fitted().save(
        tmp_path / "bottle.pt", category="bottle", threshold=0.4, reference_scores=reference
    )
    registry = load_registry(tmp_path)
    loaded = registry["bottle"]["model"]
    assert torch.equal(loaded.reference_scores, reference)
    assert registry["bottle"]["drift"] is not None


def test_artefact_without_reference_scores_runs_unmonitored(tmp_path: Path) -> None:
    path = tmp_path / "bottle.pt"
    _fitted().save(path, category="bottle", threshold=0.4)
    write_sidecar(path)
    registry = load_registry(tmp_path)
    assert registry["bottle"]["model"].reference_scores is None
    assert registry["bottle"]["drift"] is None
