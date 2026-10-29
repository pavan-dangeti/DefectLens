"""DefectLens demo: pick a VisA product category, upload a photo of a part, see the score and where it differs.

Models come from the Hugging Face model repo named below. Each category's artefact is downloaded on first
use, its SHA-256 sidecar is checked, and the model is cached for the life of the Space process.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from defectlens.models.patchcore import PatchCore
from defectlens.serving.checksums import verify_sidecar
from defectlens.serving.imaging import decode_image

MODEL_REPO = "pavan-dangeti/defectlens-visa-patchcore"
MANIFEST = json.loads(Path(hf_hub_download(MODEL_REPO, "manifest.json")).read_text())
CATEGORIES = sorted(MANIFEST["categories"])


def _artefact(category: str) -> Path:
    name = f"registry/{category}.pt"
    path = Path(hf_hub_download(MODEL_REPO, name))
    hf_hub_download(MODEL_REPO, name + ".sha256")  # lands beside the artefact in the same snapshot
    verify_sidecar(path)
    return path


@lru_cache(maxsize=len(CATEGORIES))
def _model(category: str) -> PatchCore:
    return PatchCore.load(_artefact(category))


def _overlay(image: Image.Image, score_map: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    span = float(score_map.max() - score_map.min()) or 1.0
    heat = (score_map - score_map.min()) / span
    tinted = base.copy()
    tinted[..., 0] = np.clip(base[..., 0] * (1 - heat) + 255 * heat, 0, 255)
    return Image.fromarray(tinted.astype(np.uint8))


def inspect(image: Image.Image | None, category: str) -> tuple[str, Image.Image | None]:
    if image is None:
        return "Upload a photo of a part.", None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        decoded = decode_image(buffer.getvalue())
    except Exception as exc:  # the serving decoder raises HTTPException with a reason
        return f"Image rejected: {getattr(exc, 'detail', exc)}", None
    model = _model(category)
    with torch.no_grad():
        scores, maps = model.predict(decoded.tensor)
    score = float(scores[0])
    threshold = float(MANIFEST["categories"][category]["threshold"])
    decision = "anomaly" if score > threshold else "good"
    notes = ", ".join(decoded.reasons) or "none"
    summary = (
        f"Decision: {decision}\n"
        f"Anomaly score: {score:.2f} (threshold {threshold:.2f}, set to reject 1% of good parts)\n"
        f"Image flags: {notes}"
    )
    return summary, _overlay(image, maps[0, 0].numpy())


DESCRIPTION = """
# DefectLens demo

Flags defects on manufactured parts from a single photo, using a model trained only on **good** parts.

Choose the product category, upload a photo, and press **Inspect**. The heatmap shows where the part
differs from good parts in the category's training set.

Trained and calibrated on the VisA dataset (CC BY 4.0). Research demo: not for production decisions.
Model card and limitations: see the model repository.
"""

with gr.Blocks(title="DefectLens demo") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        category = gr.Dropdown(CATEGORIES, value=CATEGORIES[0], label="Product category")
        photo = gr.Image(type="pil", label="Photo of a part")
    button = gr.Button("Inspect", variant="primary")
    result = gr.Textbox(label="Result", lines=3)
    heatmap = gr.Image(label="Where it differs from good parts (red = more different)")
    button.click(inspect, inputs=[photo, category], outputs=[result, heatmap])

if __name__ == "__main__":
    demo.launch()
