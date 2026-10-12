"""Record clean-depth estimates along an unchanged multi-step DDIM trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True, help="SD2/pipeline directory used in training")
    parser.add_argument("--unet_checkpoint", required=True, help="Training checkpoint containing unet/")
    parser.add_argument("--decoder_checkpoint", help="Optional calibrated decoder directory containing vae/")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--processing_res", type=int, default=768)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--sky_mask", help="Optional hand-checked PNG; nonzero pixels mark sky interior")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 2 or args.every < 1 or args.processing_res < 0:
        parser.error("Use --steps >= 2, --every >= 1, --processing_res >= 0")

    # Lazy import keeps --help usable without the external DA2 package.
    from diffusers import DDIMScheduler
    from marigold import MarigoldPipeline
    from src.util.model_overrides import apply_model_overrides
    from marigold.util.image_util import colorize_depth_maps, chw2hwc

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    rgb = Image.open(args.image).convert("RGB")
    sky_mask = None
    if args.sky_mask:
        sky_mask = Image.open(args.sky_mask).convert("L")
        if sky_mask.size != rgb.size:
            raise ValueError("Sky mask must be registered to the original RGB image")
    pipe = MarigoldPipeline.from_pretrained(args.base_checkpoint, torch_dtype=torch.float32)
    apply_model_overrides(pipe, args.unet_checkpoint, args.decoder_checkpoint)
    if not isinstance(pipe.scheduler, DDIMScheduler):
        raise ValueError("This diagnostic requires the multi-step DDIM model")
    pipe.to(args.device)
    pipe.unet.eval()
    pipe.da2.eval()
    pipe.vae.eval()

    records = []
    step_count = 0
    original_step = pipe.scheduler.step

    @torch.no_grad()
    def record_step(*step_args, **step_kwargs):
        nonlocal step_count
        result = original_step(*step_args, **step_kwargs)
        step_count += 1
        if step_count == 1 or step_count % args.every == 0 or step_count == args.steps:
            timestep = int(step_args[1])
            # Decode x0, not the noisy intermediate latent. Do not min-max
            # normalize each frame: doing so hides temporal depth-scale drift.
            depth_raw = pipe.decode_depth(result.pred_original_sample).float()[0, 0].cpu().numpy()
            scaled = (depth_raw + 1) / 2
            prefix = destination / f"step_{step_count:03d}_t{timestep:04d}"
            np.save(f"{prefix}.npy", scaled)
            colored = colorize_depth_maps(np.clip(scaled, 0, 1), 0, 1, cmap="Spectral").squeeze()
            Image.fromarray((chw2hwc(colored) * 255).astype(np.uint8)).save(f"{prefix}.png")
            record = {"step": step_count, "timestep": timestep,
                      "raw_min": float(scaled.min()), "raw_max": float(scaled.max())}
            if sky_mask is not None:
                mask = np.asarray(sky_mask.resize((scaled.shape[1], scaled.shape[0]), Image.Resampling.NEAREST)) > 0
                values = scaled[mask]
                if values.size:
                    record.update(sky_mean=float(values.mean()), sky_std=float(values.std()),
                                  sky_p05=float(np.quantile(values, .05)),
                                  sky_below_09=float((values < .9).mean()))
            records.append(record)
        return result

    pipe.scheduler.step = record_step
    try:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        result = pipe(rgb, denoising_steps=args.steps, ensemble_size=1, batch_size=1,
                      processing_res=args.processing_res, generator=generator)
    finally:
        pipe.scheduler.step = original_step
    np.save(destination / "final_depth.npy", result.depth_np)
    result.depth_colored.save(destination / "final_depth.png")
    (destination / "trajectory.json").write_text(
        json.dumps({"settings": vars(args), "trajectory": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
