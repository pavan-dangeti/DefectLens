from __future__ import annotations

from pathlib import Path

import pytest

from defectlens.config import (
    load_dataset_by_name,
    load_dataset_config,
    load_settings,
)


def test_load_mvtec_config() -> None:
    cfg = load_dataset_by_name("mvtec")
    assert cfg.name == "mvtec"
    assert cfg.non_commercial is True
    assert cfg.url is None
    assert len(cfg.categories) == 15
    assert cfg.licence == "CC BY-NC-SA 4.0"


def test_load_visa_config() -> None:
    cfg = load_dataset_by_name("visa")
    assert cfg.name == "visa"
    assert cfg.non_commercial is False
    assert cfg.url is not None and cfg.url.endswith(".tar")
    assert cfg.expected_size_bytes == 1929840640
    assert len(cfg.categories) == 12


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        load_dataset_by_name("pascal-voc")


def test_category_index() -> None:
    cfg = load_dataset_by_name("visa")
    assert cfg.category_index("pcb1") == 0
    with pytest.raises(KeyError):
        cfg.category_index("not-a-category")


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFECTLENS_DATA_DIR", "/tmp/dl-data")
    monkeypatch.setenv("DEFECTLENS_MAX_UPLOAD_MB", "5")
    settings = load_settings()
    assert settings.data_dir == Path("/tmp/dl-data")
    assert settings.max_upload_mb == 5


def test_settings_defaults() -> None:
    settings = load_settings()
    assert settings.port == 8000
    assert settings.default_target_frr == 0.01


def test_default_host_binds_to_localhost_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # A bare `defectlens serve` must not listen on the network; the container opts in explicitly.
    monkeypatch.delenv("DEFECTLENS_HOST", raising=False)
    assert load_settings().host == "127.0.0.1"


def test_load_dataset_config_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_dataset_config(bad)
