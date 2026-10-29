"""SHA-256 sidecars for model artefacts.

``torch.load`` on an artefact is only as safe as the file it reads, so every artefact is written
with a ``<name>.sha256`` sidecar and refused at load time if the sidecar is missing or does not
match. A silently swapped or truncated bank would otherwise score with the wrong coreset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Read in chunks: artefacts are hundreds of MB and must not be held in memory whole.
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_sidecar(path: Path) -> str:
    digest = sha256_of(path)
    sidecar_path(path).write_text(f"{digest}  {path.name}\n")
    return digest


def verify_sidecar(path: Path) -> None:
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise ValueError(f"{path} has no checksum sidecar {sidecar.name}; refusing to load")
    expected = sidecar.read_text().split()[0]
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"{path} checksum mismatch: expected {expected}, got {actual}")
