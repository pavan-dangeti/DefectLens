"""Record the DefectLens demo: inspect VisA images through a running local service and render the result.

Start the service first, bound to loopback, with the released models:

    uv run defectlens serve --registry hf/upload/defectlens-visa-patchcore/registry --host 127.0.0.1 --port 8001

Then run this script. It posts each sample to /inspect, takes the decision, score and heatmap the service
returns, and renders them into frames. The frames are written as docs/media/demo.gif and, through ffmpeg,
docs/media/demo.mp4. The images come from the local VisA dataset and are credited on screen; they are not
committed as separate files.

    uv run python hf/demo/make_demo.py
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = "http://127.0.0.1:8001"
DATA = Path("data/visa")
OUT = Path("docs/media")
# (category, folder, description). The defect images are the first file in each folder, not chosen by hand.
# The set shows a normal part, two defects the model catches and two it misses: escape rates at the 1% target
# are high for pcb1 and macaroni1, and a demo that showed only catches would overstate the model.
SAMPLES = [
    ("pcb1", "good", "Normal part"),
    ("chewinggum", "anomaly", "Defect caught"),
    ("candle", "anomaly", "Defect caught"),
    ("pcb1", "anomaly", "Defect missed"),
    ("macaroni1", "anomaly", "Defect missed"),
]
FRAME_W, FRAME_H = 960, 540
SECONDS_PER_FRAME = 3


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _first_image(category: str, folder: str) -> Path:
    files = sorted(p for p in (DATA / category / "test" / folder).iterdir() if p.is_file())
    return files[0]


def _inspect(client: httpx.Client, path: Path, category: str) -> dict[str, object]:
    response = client.post(
        "/inspect",
        params={"category": category},
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
    )
    response.raise_for_status()
    return response.json()


def _colour_heatmap(png_b64: str, size: int) -> Image.Image:
    grey = np.asarray(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("L"), dtype=np.float32)
    level = grey / 255.0
    # A heat scale from dark through red and yellow to white, so low scores stay dark.
    red = np.clip(level * 2.2, 0, 1)
    green = np.clip(level * 2.2 - 1.0, 0, 1)
    blue = np.clip(level * 2.2 - 1.8, 0, 1)
    rgb = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


def _frame(source: Path, result: dict[str, object], category: str, description: str) -> Image.Image:
    frame = Image.new("RGB", (FRAME_W, FRAME_H), (18, 18, 22))
    draw = ImageDraw.Draw(frame)
    title, body, small = _font(30), _font(24), _font(16)
    draw.text((30, 20), f"{category}: {description}", font=title, fill=(235, 235, 235))

    panel = 300
    original = Image.open(source).convert("RGB").resize((panel, panel), Image.Resampling.BILINEAR)
    heat = _colour_heatmap(str(result["heatmap_png"]), panel)
    frame.paste(original, (30, 90))
    frame.paste(heat, (30 + panel + 30, 90))
    draw.text((30, 90 + panel + 8), "Photo", font=small, fill=(170, 170, 170))
    draw.text((30 + panel + 30, 90 + panel + 8), "Where it differs (red = more)", font=small,
              fill=(170, 170, 170))

    decision = str(result["decision"])
    colour = (230, 90, 90) if decision == "anomaly" else (110, 200, 130)
    x = 30 + 2 * panel + 70
    draw.text((x, 100), f"Decision: {decision}", font=_font(22), fill=colour)
    draw.text((x, 150), f"Score: {float(result['score']):.2f}", font=body, fill=(235, 235, 235))
    draw.text((x, 185), f"Threshold: {float(result['threshold']):.2f}", font=body, fill=(235, 235, 235))
    draw.text((x, 220), "Threshold rejects 1% of good parts.", font=small, fill=(170, 170, 170))
    draw.text((30, FRAME_H - 40), "Served locally by DefectLens v1.0.0. Images: VisA, CC BY 4.0.",
              font=small, fill=(150, 150, 150))
    return frame


def _title() -> Image.Image:
    frame = Image.new("RGB", (FRAME_W, FRAME_H), (18, 18, 22))
    draw = ImageDraw.Draw(frame)
    draw.text((60, 180), "DefectLens", font=_font(64), fill=(235, 235, 235))
    draw.text((60, 270), "Flags defects from a single photo, trained on good parts only.",
              font=_font(26), fill=(200, 200, 200))
    draw.text((60, 320), "Running locally on 127.0.0.1 with the v1.0.0 VisA models.",
              font=_font(22), fill=(160, 160, 160))
    draw.text((60, FRAME_H - 40), "Images: VisA dataset, CC BY 4.0.", font=_font(16), fill=(150, 150, 150))
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = [_title()]
    with httpx.Client(base_url=BASE, timeout=60) as client:
        for category, folder, description in SAMPLES:
            source = _first_image(category, folder)
            result = _inspect(client, source, category)
            print(f"{category}/{folder}: {result['decision']} score={float(result['score']):.2f} "
                  f"threshold={float(result['threshold']):.2f}", flush=True)
            frames.append(_frame(source, result, category, description))

    gif_path = OUT / "demo.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=SECONDS_PER_FRAME * 1000, loop=0, optimize=True)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is needed for the MP4")
    with tempfile.TemporaryDirectory() as tmp:
        for index, frame in enumerate(frames):
            frame.save(Path(tmp) / f"frame_{index:02d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", f"1/{SECONDS_PER_FRAME}",
                "-i", str(Path(tmp) / "frame_%02d.png"),
                "-vf", "fps=25", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(OUT / "demo.mp4"),
            ],
            check=True,
        )
    print(f"wrote {gif_path} and {OUT / 'demo.mp4'}")


if __name__ == "__main__":
    main()
