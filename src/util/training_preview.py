"""Periodic folder inference using the current in-memory training weights."""

import copy
import logging
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch


class TrainingPreview:
    def __init__(self, config, output_dir):
        self.config = config
        self.period = int(config.get("period", 2000))
        if self.period < 0:
            raise ValueError("test_inference.period must be non-negative")
        self.output_dir = Path(output_dir) / "test_files"
        self.files = []
        folder = config.get("input_dir")
        if not folder or self.period == 0:
            return
        self.input_dir = Path(folder).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise ValueError(f"Test image folder does not exist: {self.input_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.files = sorted(p for p in self.input_dir.rglob("*")
                            if p.is_file() and p.suffix.lower() in extensions)
        if not self.files:
            raise ValueError(f"No supported test images in {self.input_dir}")
        if int(config.get("denoising_steps", 50)) < 1:
            raise ValueError("test_inference.denoising_steps must be positive")
        if int(config.get("processing_res", 768)) < 0:
            raise ValueError("test_inference.processing_res must be non-negative")
        logging.info("Training preview: %s images every %s optimizer updates",
                     len(self.files), self.period)

    @torch.no_grad()
    def maybe_run(self, model, step, device):
        if not self.files or step <= 0 or step % self.period:
            return
        destination = self.output_dir / f"iter_{step:06d}"
        device = torch.device(device)
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        # Inference must not change the training RNG, module modes or scheduler.
        modules = {}
        for name in ("unet", "vae", "da2"):
            component = getattr(model, name, None)
            if isinstance(component, torch.nn.Module):
                modules.update({module: module.training for module in component.modules()})
        scheduler = model.scheduler
        random_state, numpy_state = random.getstate(), np.random.get_state()
        logging.info("Running training preview at update %s -> %s", step, destination)
        try:
            model.scheduler = copy.deepcopy(scheduler)
            for module in modules:
                module.training = False
            with torch.random.fork_rng(devices=cuda_devices):
                for index, path in enumerate(self.files):
                    try:
                        with Image.open(path) as source:
                            rgb = ImageOps.exif_transpose(source).convert("RGB")
                    except OSError:
                        logging.warning("Skipping unreadable test image: %s", path, exc_info=True)
                        continue
                    generator = torch.Generator(device=device).manual_seed(
                        int(self.config.get("seed", 2024)) + index
                    )
                    result = model(
                        rgb,
                        denoising_steps=int(self.config.get("denoising_steps", 50)),
                        ensemble_size=1,
                        processing_res=int(self.config.get("processing_res", 768)),
                        match_input_res=True,
                        batch_size=1,
                        generator=generator,
                        color_map=self.config.get("color_map", "Spectral"),
                        show_progress_bar=False,
                    )
                    # Keep the extension to distinguish e.g. scene.jpg/scene.png.
                    relative = path.relative_to(self.input_dir)
                    output = destination / relative.parent / (relative.name + ".depth.png")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    result.depth_colored.save(output)
                    del result
        finally:
            model.scheduler = scheduler
            for module, training in modules.items():
                module.training = training
            random.setstate(random_state)
            np.random.set_state(numpy_state)
        logging.info("Training preview finished at update %s", step)
