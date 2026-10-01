"""Pixel-space calibration on frozen, fully denoised VGC latents."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def read_manifest(path):
    """JSONL: image, mask, split (train/val), optional group (scene identity).

    Masks are single-channel PNG: 255 sky, 0 non-sky, 128 unknown.
    Paths resolve relative to the manifest. Split by scene before caching seeds.
    """
    path = Path(path).resolve()
    rows, seen, groups = [], set(), {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") not in {"train", "val"}:
            raise ValueError("Each manifest row needs split=train or val")
        for key in ("image", "mask"):
            row[key] = str((path.parent / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        # Duplicate image bytes must not occur in both sets, even under aliases.
        identity = hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest()
        if identity in seen:
            raise ValueError("Duplicate RGB content in manifest; list each image once")
        seen.add(identity)
        group = str(row.get("group", identity))
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Scene group occurs in both train and val")
        groups[group] = row["split"]
        row["identity"] = identity
        rows.append(row)
    if {row["split"] for row in rows} != {"train", "val"}:
        raise ValueError("Provide separate train and val images")
    return rows


def read_labels(path, image_size):
    with Image.open(path) as mask:
        if mask.size != image_size:
            raise ValueError(f"Mask must align with original RGB: {path}")
        if mask.mode not in {"L", "P"}:
            raise ValueError("Use an 8-bit single-channel 0/128/255 mask")
        labels = np.array(mask, dtype=np.uint8)
    if not np.isin(labels, [0, 128, 255]).all():
        raise ValueError("Labels must be 0=non-sky, 128=unknown, 255=sky")
    return torch.from_numpy(labels.copy())[None, None]


def supervision_masks(labels, erosion=2):
    if labels.ndim != 4 or labels.shape[1] != 1 or erosion < 0:
        raise ValueError("Expected [B,1,H,W] labels and nonnegative erosion")
    sky = labels == 255
    if erosion:
        # Replicate padding: an image border is not a semantic boundary.
        invalid = F.pad((~sky).float(), (erosion,) * 4, mode="replicate")
        sky = F.max_pool2d(invalid, 2 * erosion + 1, stride=1) == 0
    return sky, labels == 0


def decode_raw(vae, latent, scale):
    """Exactly the pipeline's unclipped depth decode; keep gradients."""
    if scale <= 0:
        raise ValueError("Latent scale must be positive")
    return vae.decoder(vae.post_quant_conv(latent / scale)).mean(1, keepdim=True)


def train_decoder_only(vae):
    vae.requires_grad_(False).eval()
    vae.decoder.requires_grad_(True).train()
    return list(vae.decoder.parameters())


@torch.no_grad()
def capture_terminal_prediction(pipe, image, steps, processing_res, seed, device):
    """Observe the existing pipeline, without reimplementing or truncating DDIM."""
    if steps < 2:
        raise ValueError("Use multi-step inference (steps >= 2)")
    original_decode = pipe.decode_depth
    captures, calls = [], []

    def capture(latent):
        raw = original_decode(latent)
        captures.append((latent.detach().cpu().float(), raw.detach().cpu().float()))
        return raw

    handle = pipe.unet.register_forward_hook(lambda *args: calls.append(1))
    pipe.decode_depth = capture
    try:
        pipe(image, denoising_steps=steps, ensemble_size=1, batch_size=1,
             processing_res=processing_res, match_input_res=False,
             color_map=None, show_progress_bar=False,
             generator=torch.Generator(device=device).manual_seed(seed))
    finally:
        pipe.decode_depth = original_decode
        handle.remove()
    if len(captures) != 1 or len(calls) != steps:
        raise RuntimeError("Expected one final decode after exactly the configured number of U-Net calls")
    latent, reference = captures[0]
    if not torch.isfinite(latent).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Non-finite frozen VGC prediction")
    return latent, reference


def masked_mean(value, mask):
    """Equal image weighting; absent regions neither divide by zero nor dilute."""
    count = mask.sum((1, 2, 3))
    active = count > 0
    per_image = (value * mask).sum((1, 2, 3)) / count.clamp_min(1)
    return (per_image * active).sum() / active.sum().clamp_min(1)


def calibration_loss(prediction, reference, labels, erosion=2,
                     sky_weight=1.0, preserve_weight=5.0, gradient_weight=0.5):
    """Signed output-space targets: raw +1 is far, raw -1 is near.

    No affine alignment, per-image min/max normalization or clamping in loss.
    Clamping here would kill corrective gradients for saturated predictions.
    """
    if prediction.shape != reference.shape or labels.shape != prediction.shape:
        raise ValueError("Prediction, reference and labels must have matching [B,1,H,W]")
    sky, keep = supervision_masks(labels, erosion)
    reference = reference.detach()
    sky_loss = masked_mean((prediction - 1).square(), sky)
    preserve = masked_mean(F.smooth_l1_loss(prediction, reference, reduction="none", beta=0.05), keep)
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (reference[..., 1:] - reference[..., :-1])
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (reference[..., 1:, :] - reference[..., :-1, :])
    edge = 0.5 * (masked_mean(dx.abs(), keep[..., 1:] & keep[..., :-1])
                  + masked_mean(dy.abs(), keep[..., 1:, :] & keep[..., :-1, :]))
    loss = sky_weight * sky_loss + preserve_weight * preserve + gradient_weight * edge
    return loss, {"sky_mse": sky_loss.detach(), "preserve": preserve.detach(), "edge": edge.detach()}


@torch.no_grad()
def region_metrics(prediction, reference, labels, erosion=2):
    sky, keep = supervision_masks(labels, erosion)
    # Report in [0,1] depth units, without clipping away overshoot.
    pred, ref = (prediction + 1) / 2, (reference + 1) / 2
    results = {}
    if sky.any():
        values = pred[sky]
        results.update(sky_mae=(values - 1).abs().mean().item(),
                       sky_std=values.std(unbiased=False).item(),
                       sky_below_09=(values < 0.9).float().mean().item())
    if keep.any():
        error = (pred - ref)[keep].abs()
        results.update(non_sky_mae=error.mean().item(), non_sky_p95=torch.quantile(error, .95).item())
    return results


def eligible_checkpoint(metrics, baseline, best_sky, max_drift=0.02, max_p95=0.05):
    """Select on held-out sky improvement subject to non-sky preservation."""
    return (metrics["sky_mae"] < best_sky
            and metrics["sky_below_09"] <= baseline["sky_below_09"]
            and metrics["non_sky_mae"] <= max_drift
            and metrics["non_sky_p95"] <= max_p95)
