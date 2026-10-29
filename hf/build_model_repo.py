"""Build the Hugging Face model repository folder from a staging registry.

Reads every <category>.pt and its .sha256 sidecar from the staging registry, writes manifest.json (threshold,
checksum and size per category), and copies the model card as the repository README. Nothing is uploaded here.

    uv run python hf/build_model_repo.py --registry <staging>/registry --out hf/upload/defectlens-visa-patchcore
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from defectlens.serving.checksums import sha256_of, sidecar_path

CARD = Path("docs/model-card.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out_registry = args.out / "registry"
    out_registry.mkdir(parents=True, exist_ok=True)
    categories: dict[str, dict[str, object]] = {}
    for artefact in sorted(args.registry.glob("*.pt")):
        blob = torch.load(artefact, map_location="cpu", weights_only=True)
        sidecar = sidecar_path(artefact)
        shutil.copy2(artefact, out_registry / artefact.name)
        shutil.copy2(sidecar, out_registry / sidecar.name)
        categories[blob["category"]] = {
            "threshold": blob["threshold"],
            "coreset_ratio": blob["coreset_ratio"],
            "sha256": sha256_of(artefact),
            "bytes": artefact.stat().st_size,
            "file": f"registry/{artefact.name}",
        }
    manifest = {"version": "1.0.0", "target_false_reject_rate": 0.01, "categories": categories}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(CARD, args.out / "README.md")
    print(f"{len(categories)} categories -> {args.out}")


if __name__ == "__main__":
    main()
