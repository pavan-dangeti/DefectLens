"""Reproducible parity run (ADR-0004): my PatchCore vs anomalib's PatchCore.

Run: `uv run python results/parity/run.py`
Requires network on first run to fetch pretrained WRN50 weights via timm.
Writes results/parity/summary.json alongside this script.
"""

import json
import sys
from pathlib import Path

from defectlens.eval.parity import run_parity

if __name__ == "__main__":
    summary = run_parity()
    out = Path(__file__).parent / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(0)
