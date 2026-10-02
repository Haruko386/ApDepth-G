"""Fit only the VAE decoder on cached complete VGC inference results."""
import argparse
import json
import math
from pathlib import Path
import random

import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf

from src.util.decoder_calibration import (
    calibration_loss, decode_raw, eligible_checkpoint, region_metrics,
    supervision_masks, train_decoder_only,
)


def load_record(root, row, device="cpu"):
    item = torch.load(root / row["file"], map_location=device, weights_only=True)
    latent, reference, labels = (item[k] for k in ("latent", "reference", "labels"))
    if (latent.ndim != 4 or latent.shape[:2] != (1, 4)
            or reference.ndim != 4 or reference.shape[:2] != (1, 1)
            or labels.shape != reference.shape):
        raise ValueError("Invalid cached tensor shapes")
    if not torch.isfinite(latent).all() or not torch.isfinite(reference).all():
        raise ValueError("Non-finite cache")
    if not ((labels == 0) | (labels == 128) | (labels == 255)).all():
        raise ValueError("Invalid cache labels")
    return item


@torch.no_grad()
def evaluate(vae, root, records, scale, erosion):
    vae.eval()
    device = next(vae.parameters()).device
    values = {}
    for row in records:
        item = load_record(root, row, device)
        prediction = decode_raw(vae, item["latent"], scale)
        for key, value in region_metrics(prediction, item["reference"], item["labels"], erosion).items():
            values.setdefault(key, []).append(value)
    required = {"sky_mae", "sky_below_09", "non_sky_mae", "non_sky_p95"}
    if not required.issubset(values):
        raise ValueError("Validation needs both sky interiors and non-sky supervision")
    # Worst-image p95 prevents good replay images hiding damage to one scene.
    return {key: max(v) if key == "non_sky_p95" else sum(v) / len(v) for key, v in values.items()}


def validate_config(cfg):
    for key in ("max_steps", "accumulation_steps", "evaluate_every", "save_every"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm", "sky_weight", "preserve_weight", "gradient_weight",
                "max_non_sky_mae", "max_non_sky_p95"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["erosion"] < 0 or cfg["warmup_steps"] < 0:
        raise ValueError("erosion and warmup_steps must be nonnegative")


def fit(cache_dir, output_dir, cfg, device="cuda", resume=None):
    """No DA2/pipeline import, no denoiser calls, and no RGB/GT re-encoding."""
    validate_config(cfg)
    root, destination = Path(cache_dir).resolve(), Path(output_dir).resolve()
    index_text = (root / "index.json").read_text(encoding="utf-8")
    metadata = json.loads(index_text)
    if metadata["version"] != 1 or metadata["steps"] < 2:
        raise ValueError("Expected a complete multi-step cache")
    records = metadata["records"]
    train_rows = [row for row in records if row["split"] == "train"]
    val_rows = [row for row in records if row["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("Need train and val cache records")
    if {r["identity"] for r in train_rows} & {r["identity"] for r in val_rows}:
        raise ValueError("Same image cannot be both training and validation")
    sky_rows, replay_rows = [], []
    for row in train_rows:
        item = load_record(root, row)
        sky, keep = supervision_masks(item["labels"], cfg["erosion"])
        if not (sky.any() or keep.any()):
            raise ValueError(f"No supervised pixels: {row['file']}")
        (sky_rows if sky.any() else replay_rows).append(row)
    if not sky_rows:
        raise ValueError("No training sky survives mask erosion")
    if not replay_rows:
        print("No no-sky replay images: non-sky regions within sky images are retained, but indoor preservation is untested.")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (destination / "cache_index.json").write_text(index_text, encoding="utf-8")
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    vae = AutoencoderKL.from_pretrained(root / "source_vae").to(device=device, dtype=torch.float32)
    scale = metadata["depth_latent_scale_factor"]
    parameters = train_decoder_only(vae)
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0)
    baseline = evaluate(vae, root, val_rows, scale, cfg["erosion"])
    if baseline["non_sky_p95"] > 1e-4:
        raise ValueError("Source VAE does not reproduce cached baseline; check dtype, scale and cache provenance")
    best_sky, start, best_step = baseline["sky_mae"], 0, 0
    if resume:
        previous = Path(resume).resolve()
        state = torch.load(previous / "train_state.pt", map_location="cpu", weights_only=True)
        if state["cache_index"] != index_text or state["config"] != cfg:
            raise ValueError("Resume must use exactly the same cache and training configuration")
        trained_vae = AutoencoderKL.from_pretrained(previous / "vae")
        vae.load_state_dict(trained_vae.state_dict(), strict=True)
        del trained_vae
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.startswith("cuda") and state["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start = state["step"]
        # Select best among candidates in this new continuation directory.
        # Previous run's best remains untouched; baseline comparison persists.

    def save(folder, step, metrics, with_state=False):
        folder.mkdir(parents=True, exist_ok=True)
        vae.save_pretrained(folder / "vae")
        report = {"step": step, "metrics": metrics, "baseline": baseline,
                  "source": {k: v for k, v in metadata.items() if k != "records"}}
        (folder / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if with_state:
            torch.save({"step": step, "optimizer": optimizer.state_dict(), "config": cfg,
                        "cache_index": index_text, "rng_state": rng.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.startswith("cuda") else []},
                       folder / "train_state.pt")

    def report(step):
        nonlocal best_sky, best_step
        metrics = evaluate(vae, root, val_rows, scale, cfg["erosion"])
        selected = eligible_checkpoint(metrics, baseline, best_sky,
                                       cfg["max_non_sky_mae"], cfg["max_non_sky_p95"])
        if selected:
            best_sky, best_step = metrics["sky_mae"], step
            save(destination / "best", step, metrics)
        record = {"step": step, "validation": metrics, "selected": selected}
        with (destination / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        return metrics

    (destination / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    if start:
        report(start)
    print(f"Training decoder only: {sum(p.numel() for p in parameters):,} parameters; baseline={baseline}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start + 1, cfg["max_steps"] + 1):
        vae.decoder.train()
        running = 0.0
        for _ in range(cfg["accumulation_steps"]):
            # Preserve indoor/no-sky behavior with replay when available.
            pool = replay_rows if replay_rows and rng.random() < .25 else sky_rows
            item = load_record(root, rng.choice(pool), device)
            prediction = decode_raw(vae, item["latent"], scale)
            loss, _ = calibration_loss(prediction, item["reference"], item["labels"],
                                       cfg["erosion"], cfg["sky_weight"], cfg["preserve_weight"], cfg["gradient_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Decoder loss became non-finite")
            (loss / cfg["accumulation_steps"]).backward()
            running += loss.item() / cfg["accumulation_steps"]
        torch.nn.utils.clip_grad_norm_(parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
        warmup = min(1.0, step / max(1, cfg["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"] * warmup
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            print(f"step={step} loss={running:.6f}", flush=True)
        metrics = None
        if step % cfg["evaluate_every"] == 0 or step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
            metrics = report(step)
        if step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
            save(destination / f"step_{step:06d}", step, metrics, with_state=True)
    print(f"Best accepted step: {best_step}. " + ("Use best/vae with original VGC U-Net." if best_step
          else "No checkpoint passed held-out improvement/preservation checks; keep the original VGC model."), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train_decoder_calibration.yaml")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True, help="New run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", help="step_XXXXXX directory; continue into a new output directory")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    fit(args.cache_dir, args.output_dir, cfg, args.device, args.resume)


if __name__ == "__main__":
    main()
