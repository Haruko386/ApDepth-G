"""Training-only, SNR-weighted completion in the frozen VAE's latent space."""

import torch
import torch.nn.functional as F


def min_snr_weights(alpha_bar, prediction_type, gamma=5.0):
    """Return prediction-space MSE and bounded x0-space auxiliary weights.

    For each parameterization, weighted prediction MSE equals
    min(SNR, gamma) * MSE(x0_pred, x0). The auxiliary weight is that
    x0 weight divided by gamma, without batch renormalization.
    """
    if gamma <= 0:
        raise ValueError("Min-SNR gamma must be positive")
    alpha = alpha_bar.float()
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    capped = snr.clamp(max=gamma)
    if prediction_type == "epsilon":
        weight = torch.where(snr <= gamma, torch.ones_like(snr), capped / snr.clamp_min(1e-8))
    elif prediction_type == "v_prediction":
        weight = capped / (snr + 1.0)
    elif prediction_type == "sample":
        weight = capped
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")
    return weight, capped / gamma


def latent_validity_masks(valid_pixels, scale=8):
    """Mixed cells belong to neither set; invalid does NOT mean semantic sky.

    These masks cover each downsampling block, not the full VAE receptive field.
    """
    valid = valid_pixels.bool()
    fully_valid = ~F.max_pool2d((~valid).float(), scale, scale).bool()
    fully_invalid = ~F.max_pool2d(valid.float(), scale, scale).bool()
    return fully_valid, fully_invalid


def residual_completion_loss(pred_x0, prior_latent, invalid_mask, x0_weight,
                             anchor_weight=0.02, gradient_weight=0.005,
                             min_invalid_ratio=0.08):
    """Match residual DC and signed spatial derivatives on invalid interiors.

    Unlike TV(pred_x0), this has zero loss at the prior even when the prior's
    latent encoding is spatially nonconstant. Both endpoints of an edge must
    be invalid. Each sample is normalized independently before SNR weighting.
    DA2 is detached: only the U-Net is optimized.
    """
    pred = pred_x0.float()
    mask = invalid_mask.bool().expand_as(pred)
    residual = pred - prior_latent.detach().float()
    # Exclude unused values before arithmetic reductions, including NaN targets.
    residual = torch.where(mask, residual, torch.zeros_like(residual))
    count = mask.sum(dim=(2, 3)).clamp_min(1)
    anchor = (residual.sum(dim=(2, 3)) / count).square().mean(dim=1)

    mx = mask[..., 1:] & mask[..., :-1]
    my = mask[..., 1:, :] & mask[..., :-1, :]
    dx = residual[..., 1:] - residual[..., :-1]
    dy = residual[..., 1:, :] - residual[..., :-1, :]
    gx = torch.where(mx, dx.square(), 0.0).flatten(1).sum(1) / mx.flatten(1).sum(1).clamp_min(1)
    gy = torch.where(my, dy.square(), 0.0).flatten(1).sum(1) / my.flatten(1).sum(1).clamp_min(1)
    gradient = 0.5 * (gx + gy)

    ratio = mask.float().flatten(1).mean(1)
    eligible = (ratio >= min_invalid_ratio) & mask.flatten(1).any(1)
    weight = x0_weight.detach().reshape(pred.shape[0]) * eligible
    # Do not divide by sum(weight): that would cancel noise attenuation.
    denominator = eligible.sum().clamp_min(1)
    anchor_loss = (weight * anchor).sum() / denominator
    gradient_loss = (weight * gradient).sum() / denominator
    loss = anchor_weight * anchor_loss + gradient_weight * gradient_loss
    return loss, {
        "completion_anchor": anchor_loss.detach(),
        "completion_gradient": gradient_loss.detach(),
        "completion_coverage": ratio.mean().detach(),
        "completion_snr_weight": (weight.sum() / denominator).detach(),
    }
