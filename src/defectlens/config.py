from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    name: str
    licence: str
    non_commercial: bool = False
    url: str | None = None
    expected_size_bytes: int | None = None
    categories: list[str] = Field(default_factory=list)
    notes: str = ""

    def category_index(self, category: str) -> int:
        try:
            return self.categories.index(category)
        except ValueError:
            raise KeyError(f"{category!r} is not a category of {self.name}") from None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEFECTLENS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    mlruns_dir: Path = Path("mlruns")
    # Loopback by default: a bare `defectlens serve` must not expose the service on the network.
    # The container binds 0.0.0.0 explicitly, in the Dockerfile CMD.
    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: int = 20
    default_target_frr: float = 0.01


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a dataset config mapping")
    return DatasetConfig.model_validate(raw)


def load_dataset_by_name(name: str, root: Path = Path("configs/datasets")) -> DatasetConfig:
    for path in sorted(root.glob("*.yaml")):
        cfg = load_dataset_config(path)
        if cfg.name == name:
            return cfg
    raise KeyError(f"no dataset config named {name!r} under {root}")


def load_settings() -> Settings:
    return Settings()


class CoresetConfig(BaseModel):
    """Coreset ratio per category: the default for every category not listed in ``categories``."""

    default: float = Field(0.01, gt=0.0, le=1.0)
    categories: dict[str, float] = Field(default_factory=dict)

    def ratio_for(self, category: str) -> float:
        return self.categories.get(category, self.default)


def load_coreset_config(path: Path = Path("configs/coreset.yaml")) -> CoresetConfig:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a mapping with 'default' and 'categories'")
    return CoresetConfig.model_validate(raw)
