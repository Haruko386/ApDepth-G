"""Load a trained depth U-Net together with its prediction parameterization."""

from pathlib import Path

from src.util.prior_residual import load_parameterization


def load_unet_checkpoint(pipe, checkpoint, torch_dtype=None):
    from marigold.modules.unet_2d_condition import UNet2DConditionModel

    root = Path(checkpoint)
    unet_dir = root / "unet"
    pipe.unet = UNet2DConditionModel.from_pretrained(
        unet_dir,
        torch_dtype=torch_dtype,
    )
    if pipe.unet.config.in_channels != 12:
        raise ValueError("Expected a trained 12-channel depth U-Net")
    mode, scale = load_parameterization(root / "depth_parameterization.json")
    pipe.set_depth_parameterization(mode, scale)
    pipe.unet.eval()
    return mode, scale
