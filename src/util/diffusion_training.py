"""Diffusion schedule and masked-training utilities."""

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler


def min_snr_weight(alpha_bar, prediction_type, gamma=5.0, snr_floor=0.0):
    """Return the prediction-space weight for a bounded x0-space objective.

    ``snr_floor`` is only supported for v-prediction. With a zero-terminal-SNR
    schedule, the usual Min-SNR formula assigns exactly zero weight to the pure
    noise endpoint. A small floor keeps that endpoint trainable while retaining
    the upper Min-SNR cap.
    """
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be positive")
    if not 0 <= snr_floor <= gamma:
        raise ValueError("SNR floor must be in [0, gamma]")
    if snr_floor and prediction_type != "v_prediction":
        raise ValueError("A positive SNR floor requires v_prediction")
    alpha = alpha_bar.float()
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    bounded = snr.clamp(min=snr_floor, max=gamma)
    if prediction_type == "epsilon":
        weight = torch.where(snr <= gamma, torch.ones_like(snr), bounded / snr.clamp_min(1e-8))
    elif prediction_type == "v_prediction":
        weight = bounded / (snr + 1.0)
    elif prediction_type == "sample":
        weight = bounded
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    return weight


def _schedule_kwargs(config):
    prediction_type = str(config.get("prediction_type", "epsilon"))
    zero_terminal = bool(config.get("rescale_betas_zero_snr", False))
    timestep_spacing = str(config.get("timestep_spacing", "leading"))
    if zero_terminal and prediction_type != "v_prediction":
        raise ValueError("Zero-terminal-SNR training requires v_prediction")
    if timestep_spacing not in ("leading", "trailing", "linspace"):
        raise ValueError(f"Unsupported timestep spacing: {timestep_spacing}")
    return {
        "prediction_type": prediction_type,
        "rescale_betas_zero_snr": zero_terminal,
        "timestep_spacing": timestep_spacing,
    }


def align_training_scheduler(base_scheduler, config):
    """Build the training scheduler from the base schedule plus explicit overrides."""
    return DDPMScheduler.from_config(base_scheduler.config, **_schedule_kwargs(config))


def align_inference_scheduler(base_scheduler, config):
    """Build a matching multi-step DDIM scheduler for inference."""
    return DDIMScheduler.from_config(
        base_scheduler.config,
        clip_sample=False,
        **_schedule_kwargs(config),
    )


def terminal_noise_fade(timesteps, num_train_timesteps, fade_fraction=0.1):
    """Fade structured noise augmentation to zero near the pure-noise endpoint."""
    if num_train_timesteps < 2:
        raise ValueError("At least two training timesteps are required")
    if not 0 < fade_fraction <= 1:
        raise ValueError("Terminal noise fade fraction must be in (0, 1]")
    progress = timesteps.float() / float(num_train_timesteps - 1)
    return ((1.0 - progress) / fade_fraction).clamp(0.0, 1.0)


def latent_validity_masks(valid_pixels, scale=8):
    """Mixed cells belong to neither set; invalid does NOT mean semantic sky.

    These masks cover each downsampling block, not the full VAE receptive field.
    """
    valid = valid_pixels.bool()
    fully_valid = ~F.max_pool2d((~valid).float(), scale, scale).bool()
    fully_invalid = ~F.max_pool2d(valid.float(), scale, scale).bool()
    return fully_valid, fully_invalid
