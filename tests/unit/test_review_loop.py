"""End-to-end review loop: fit → serve → review → refit on operator feedback.

Runs on a synthetic dataset with ``--no-pretrained`` so it never touches the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from defectlens import cli
from defectlens.serving.app import create_app
from defectlens.serving.registry import load_registry

runner = CliRunner()


def _dataset(root: Path, category: str = "bottle", n: int = 8) -> Path:
    train = root / category / "train" / "good"
    train.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        noise = rng.integers(0, 40, size=(256, 256, 3), dtype=np.uint8)
        Image.fromarray(noise).save(train / f"{i:03d}.png")
    return root


def _fit(root: Path, registry: Path, *extra: str):
    return runner.invoke(
        cli.app,
        [
            "fit",
            "--root",
            str(root),
            "--category",
            "bottle",
            "--out",
            str(registry),
            "--backbone",
            "resnet18",
            "--no-pretrained",
            "--coreset-ratio",
            "0.5",
            *extra,
        ],
    )


def test_fit_then_serve_is_ready(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    registry = tmp_path / "registry"

    result = _fit(root, registry)
    assert result.exit_code == 0, result.output
    assert (registry / "bottle.pt").exists()
    assert "threshold:" in result.output

    loaded = load_registry(registry)
    assert list(loaded) == ["bottle"]
    assert loaded["bottle"]["threshold"] > 0.0
    assert TestClient(create_app(models=loaded)).get("/ready").status_code == 200


def test_review_feedback_is_consumed_by_refit(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    registry = tmp_path / "registry"
    assert _fit(root, registry).exit_code == 0

    # Flag an image, label it confirmed-good, then refit and confirm the bank grew by one.
    client = TestClient(
        create_app(
            models=load_registry(registry),
            spool_dir=tmp_path / "queue",
            review_log=tmp_path / "reviews.jsonl",
        )
    )
    payload = Path(next((tmp_path / "data" / "bottle" / "train" / "good").iterdir())).read_bytes()
    case_id = client.post(
        "/inspect?category=bottle", files={"file": ("t.png", payload, "image/png")}
    ).json()["case_id"]
    client.post(f"/review/{case_id}?label=confirmed-good")

    result = _fit(root, registry, "--review-log", str(tmp_path / "reviews.jsonl"))
    assert result.exit_code == 0, result.output
    # 8 nominal images split 6 fit / 1 calibration / 1 holdout, plus the confirmed-good review image.
    assert "fit on:    7" in result.output, result.output


def test_refit_without_review_log_still_works(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "data")
    result = _fit(root, tmp_path / "registry", "--review-log", str(tmp_path / "missing.jsonl"))
    assert result.exit_code == 0, result.output
    assert "fit on:    6" in result.output
