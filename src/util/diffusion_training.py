"""Utilities for the v-prediction training schedule and masked latent targets."""

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler


V_PREDICTION = "v_prediction"


def min_snr_v_weight(alpha_bar, gamma=5.0, endpoint_floor=0.05):
    """Min-SNR weight expressed in v-prediction space.

    The small endpoint floor keeps the pure-noise endpoint trainable after the
    schedule is rescaled to zero terminal SNR.
    """
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be positive")
    if not 0 <= endpoint_floor <= gamma:
        raise ValueError("endpoint_floor must be in [0, gamma]")
    alpha = alpha_bar.float()
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    bounded_snr = snr.clamp(min=endpoint_floor, max=gamma)
    return bounded_snr / (snr + 1.0)


def terminal_noise_fade(timesteps, num_train_timesteps, fade_fraction=0.1):
    """Fade structured noise augmentation near the pure-noise endpoint."""
    if num_train_timesteps < 2:
        raise ValueError("At least two training timesteps are required")
    if not 0 < fade_fraction <= 1:
        raise ValueError("fade_fraction must be in (0, 1]")
    progress = timesteps.float() / float(num_train_timesteps - 1)
    return ((1.0 - progress) / fade_fraction).clamp(0.0, 1.0)


def make_v_prediction_schedulers(
    base_training_scheduler, base_inference_scheduler, config=None
):
    """Create matching DDPM/DDIM schedulers for training and multi-step inference."""
    config = config or {}
    common = {
        "prediction_type": str(config.get("prediction_type", V_PREDICTION)),
        "rescale_betas_zero_snr": bool(
            config.get("rescale_betas_zero_snr", True)
        ),
        "timestep_spacing": str(config.get("timestep_spacing", "trailing")),
    }
    if common != {
        "prediction_type": V_PREDICTION,
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
    }:
        raise ValueError(
            "This experiment requires v_prediction, zero terminal SNR, and "
            "trailing timestep spacing"
        )
    training = DDPMScheduler.from_config(base_training_scheduler.config, **common)
    inference = DDIMScheduler.from_config(
        base_inference_scheduler.config,
        clip_sample=False,
        **common,
    )
    return training, inference


def fill_invalid_depth_with_prior(depth, valid_mask, prior):
    """Remove invalid-value leakage before the frozen image VAE sees the target.

    Invalid raw depth is normalized to the near plane in the existing datasets.
    A VAE latent covers a wider area than a single 8x8 downsampling block, so
    masking the latent loss afterwards cannot undo that contamination.  The
    frozen prior is used only as an encoder-safe fill; invalid pixels remain
    excluded from every supervised loss.
    """
    valid = valid_mask.bool()
    if depth.ndim != 4 or valid.shape != depth.shape:
        raise ValueError("depth and valid_mask must have matching [B, 1, H, W] shapes")
    if prior.ndim != 4 or prior.shape[0] != depth.shape[0] or prior.shape[-2:] != depth.shape[-2:]:
        raise ValueError("prior must have the same batch and spatial dimensions as depth")
    prior_depth = prior[:, :1].to(device=depth.device, dtype=depth.dtype)
    return torch.where(valid, depth, prior_depth)


def conservative_latent_valid_mask(valid_pixels, latent_hw, boundary_margin=2):
    """Keep only fully valid latent cells away from invalid-depth boundaries.

    The first pooling step maps the pixel mask to the actual latent resolution.
    The second step erodes the result in latent space to cover the VAE receptive
    field beyond its nominal 8x spatial reduction.
    """
    if boundary_margin < 0:
        raise ValueError("boundary_margin must be non-negative")
    valid = valid_pixels.bool()
    if valid.ndim != 4 or valid.shape[1] != 1:
        raise ValueError("valid_pixels must have shape [B, 1, H, W]")
    invalid_fraction = F.adaptive_max_pool2d((~valid).float(), latent_hw)
    latent_valid = invalid_fraction == 0
    if boundary_margin:
        kernel = 2 * boundary_margin + 1
        near_invalid = F.max_pool2d(
            (~latent_valid).float(), kernel_size=kernel, stride=1,
            padding=boundary_margin,
        ).bool()
        latent_valid = ~near_invalid
    return latent_valid


def validate_v_scheduler(scheduler):
    """Reject checkpoints that could silently fall back to epsilon prediction."""
    config = scheduler.config
    if config.prediction_type != V_PREDICTION:
        raise ValueError("Checkpoint scheduler must use v_prediction")
    if not config.rescale_betas_zero_snr:
        raise ValueError("Checkpoint scheduler must use zero terminal SNR")
    if config.timestep_spacing != "trailing":
        raise ValueError("Checkpoint scheduler must use trailing timesteps")
