"""Cache final latents from the original VGC model's complete multi-step DDIM."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True, help="Full SD2 pipeline used by original VGC")
    parser.add_argument("--unet_checkpoint", required=True, help="Original VGC checkpoint containing unet/")
    parser.add_argument("--manifest", required=True, help="RGB/semantic mask JSONL, with train/val splits")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--processing_res", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 2 or args.processing_res < 64 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use steps >= 2, processing_res >= 64, distinct seeds")

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffusers import DDIMScheduler, UNet2DConditionModel
    from marigold import MarigoldPipeline
    from src.util.decoder_calibration import read_manifest, read_labels, capture_terminal_prediction

    rows = read_manifest(args.manifest)
    # Validate labels before loading expensive models or generating the cache.
    for row in rows:
        with Image.open(row["image"]) as rgb:
            read_labels(row["mask"], rgb.size)
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    pipe = MarigoldPipeline.from_pretrained(args.base_checkpoint, torch_dtype=torch.float32)
    pipe.unet = UNet2DConditionModel.from_pretrained(Path(args.unet_checkpoint) / "unet")
    if pipe.unet.config.in_channels != 12 or not isinstance(pipe.scheduler, DDIMScheduler):
        raise ValueError("Use original 12-channel VGC with a DDIM scheduler")
    pipe.to(args.device)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.da2):
        module.requires_grad_(False).eval()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except (ImportError, ModuleNotFoundError):
        pass
    # Snapshot the exact VAE: decoder fitting needs neither SD2 U-Net nor DA2.
    pipe.vae.save_pretrained(destination / "source_vae")
    metadata = {"version": 1, "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                "unet_checkpoint": str(Path(args.unet_checkpoint).resolve()),
                "steps": args.steps, "processing_res": args.processing_res,
                "seeds": args.seeds, "depth_latent_scale_factor": pipe.depth_latent_scale_factor,
                "scheduler": dict(pipe.scheduler.config), "records": []}
    for row in rows:
        with Image.open(row["image"]) as source:
            rgb = source.convert("RGB")
        labels = read_labels(row["mask"], rgb.size)
        for seed in args.seeds:
            latent, reference = capture_terminal_prediction(
                pipe, rgb, args.steps, args.processing_res, seed, args.device)
            resized_labels = F.interpolate(labels.float(), reference.shape[-2:], mode="nearest").to(torch.uint8)
            name = f"{row['identity']}_{seed}.pt"
            torch.save({"latent": latent, "reference": reference, "labels": resized_labels}, destination / name)
            metadata["records"].append({"file": name, "split": row["split"], "image": row["image"],
                                        "identity": row["identity"], "seed": seed})
            print(f"Cached {row['split']}: {Path(row['image']).name}, seed={seed}", flush=True)
    # index.json only appears for a complete cache, so partial runs cannot train.
    (destination / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
