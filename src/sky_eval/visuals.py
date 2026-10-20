"""Diagnostic-only Pillow panels. Quantitative scores never read these images."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .masks import Regions


def _rgb_copy(source: Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def mask_diagnostic(rgb_path: Path | Image.Image, regions: Regions, output: Path) -> None:
    rgb = _rgb_copy(rgb_path)
    base = np.asarray(rgb)
    panels = [rgb.copy()]
    for mask, color in ((regions.sky, (70, 180, 255)), (regions.interior, (50, 230, 80))):
        overlay = base.copy()
        overlay[mask] = (0.5 * base[mask] + 0.5 * np.array(color)).astype(np.uint8)
        panels.append(Image.fromarray(overlay))
    bands = base.copy()
    bands[regions.sky_band] = (40, 180, 255)
    bands[regions.foreground_band] = (255, 160, 40)
    panels.append(Image.fromarray(bands))
    paired = rgb.copy()
    draw = ImageDraw.Draw(paired)
    for i in np.linspace(0, max(regions.n_pairs - 1, 0), min(24, regions.n_pairs), dtype=int):
        sy, sx = np.unravel_index(regions.sky_indices[i, -1], regions.sky.shape)
        fy, fx = np.unravel_index(regions.foreground_indices[i, -1], regions.sky.shape)
        draw.line((int(sx), int(sy), int(fx), int(fy)), fill=(255, 50, 50), width=2)
        draw.ellipse((int(fx)-2, int(fy)-2, int(fx)+2, int(fy)+2), fill=(255, 255, 0))
    panels.append(paired)
    _save_panels(panels, ["RGB", "Semantic sky", "Sky interior", "2% bands: sky / foreground",
                          f"Normals: yellow = foreground; pairs={regions.n_pairs}"], output)


def prediction_diagnostic(rgb_path: Path | Image.Image, predictions: dict[str, NDArray[np.float64]], output: Path) -> None:
    """Shared whole-image display range, after documented direction conversion."""
    panels = [_rgb_copy(rgb_path)]
    low = min(float(d.min()) for d in predictions.values())
    high = max(float(d.max()) for d in predictions.values())
    labels = ["RGB"]
    for name, depth in predictions.items():
        # Scaling is display-only; it is never fed back into evaluation.
        scaled = (depth - low) / (high - low) if high > low else np.zeros_like(depth)
        panels.append(Image.fromarray(np.round(scaled * 255).astype(np.uint8)).convert("RGB"))
        labels.append(f"{name}: white=far; shared [{low:.5g}, {high:.5g}]")
    _save_panels(panels, labels, output)


def _save_panels(panels: list[Image.Image], labels: list[str], output: Path) -> None:
    width = min(640, panels[0].width)
    height = round(panels[0].height * width / panels[0].width)
    canvas = Image.new("RGB", (width, (height + 28) * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        y = index * (height + 28)
        draw.text((6, y + 6), label, fill="black")
        canvas.paste(panel.resize((width, height), Image.Resampling.NEAREST), (0, y + 28))
    with output.open("xb") as stream:
        canvas.save(stream, format="PNG")
