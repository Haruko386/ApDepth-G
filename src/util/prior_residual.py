"""Prior-anchored parameterization for conditional depth diffusion."""

import json
from pathlib import Path


ABSOLUTE = "absolute"
PRIOR_RESIDUAL = "prior_residual"


def validate_parameterization(mode, residual_scale=1.0):
    mode = str(mode)
    scale = float(residual_scale)
    if mode not in (ABSOLUTE, PRIOR_RESIDUAL):
        raise ValueError(f"Unknown depth parameterization: {mode}")
    if scale <= 0:
        raise ValueError("depth residual scale must be positive")
    return mode, scale


def diffusion_target(depth_latent, prior_latent, mode, residual_scale=1.0):
    """Map an absolute GT depth latent into the chosen diffusion state."""
    mode, scale = validate_parameterization(mode, residual_scale)
    if mode == PRIOR_RESIDUAL:
        return (depth_latent - prior_latent.detach()) / scale
    return depth_latent


def compose_depth_latent(state_latent, prior_latent, mode, residual_scale=1.0):
    """Map a denoised diffusion state back to an absolute depth latent."""
    mode, scale = validate_parameterization(mode, residual_scale)
    if mode == PRIOR_RESIDUAL:
        return prior_latent + scale * state_latent
    return state_latent


def save_parameterization(path, mode, residual_scale=1.0):
    mode, scale = validate_parameterization(mode, residual_scale)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"mode": mode, "residual_scale": scale}, indent=2),
        encoding="utf-8",
    )


def load_parameterization(path, default_mode=ABSOLUTE, default_scale=1.0):
    source = Path(path)
    if not source.is_file():
        return validate_parameterization(default_mode, default_scale)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_parameterization(
        payload.get("mode", default_mode),
        payload.get("residual_scale", default_scale),
    )
