from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from defectlens.data.download import compute_sha256


class DataManifest(BaseModel):
    dataset: str
    archive: str
    sha256: str | None = None
    size_bytes: int
    expected_size_bytes: int | None = None
    downloaded_at: str
    extracted: bool = False

    def verify(self, root: Path) -> list[str]:
        issues: list[str] = []
        archive = root / self.archive
        if not archive.exists():
            return [f"{self.archive} missing under {root}"]
        size = archive.stat().st_size
        if size != self.size_bytes:
            issues.append(f"{self.archive}: size {size} != manifest {self.size_bytes}")
        if self.expected_size_bytes is not None and size != self.expected_size_bytes:
            issues.append(
                f"{self.archive}: size {size} != dataset specification {self.expected_size_bytes}"
            )
        if self.sha256 is not None:
            actual = compute_sha256(archive)
            if actual != self.sha256:
                issues.append(f"{self.archive}: sha256 mismatch")
        return issues


def build_manifest(
    dataset: str,
    archive: Path,
    expected_size: int | None = None,
    sha256: str | None = None,
    extracted: bool = False,
) -> DataManifest:
    return DataManifest(
        dataset=dataset,
        archive=archive.name,
        sha256=sha256 if sha256 is not None else compute_sha256(archive),
        size_bytes=archive.stat().st_size,
        expected_size_bytes=expected_size,
        downloaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        extracted=extracted,
    )


def to_yaml(manifest: DataManifest) -> str:
    return yaml.safe_dump(manifest.model_dump(), sort_keys=False)


def write_manifest(manifest: DataManifest, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(to_yaml(manifest))
    return dest


def read_manifest(path: Path) -> DataManifest:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a manifest mapping")
    return DataManifest.model_validate(raw)
