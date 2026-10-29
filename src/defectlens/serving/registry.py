"""Model registry — loads fitted PatchCore artefacts for the serving surface.

An artefact is one ``<category>.pt`` written by ``PatchCore.save``. The directory is the
contract: ``/ready`` reports 200 only once at least one artefact loads, so a deployment with a
missing or corrupt file is reported unready instead of silently scoring with a default bank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from defectlens.models.patchcore import PatchCore
from defectlens.serving.checksums import verify_sidecar

ARTIFACT_GLOB = "*.pt"


def load_registry(root: Path) -> dict[str, dict[str, Any]]:
    """Load every ``<category>.pt`` under ``root``.

    Raises FileNotFoundError if ``root`` is missing and ValueError if an artefact is malformed,
    so a broken deploy fails at startup instead of at first request.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"registry directory {root} does not exist")

    registry: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob(ARTIFACT_GLOB)):
        verify_sidecar(path)
        try:
            model = PatchCore.load(path)
        except Exception as exc:  # any load failure must name the file, not leak a pickle error
            raise ValueError(f"{path} is not a loadable PatchCore artefact") from exc
        if model.category is None:
            raise ValueError(f"{path} has no category recorded")
        registry[model.category] = {
            "model": model,
            "threshold": model.threshold,
            "version": path.stem,
        }
    return registry
