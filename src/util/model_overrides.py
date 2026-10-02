"""Load original VGC U-Net and an optional calibrated decoder for inference."""
from pathlib import Path

import torch


def apply_model_overrides(pipe, unet_checkpoint=None, decoder_checkpoint=None):
    from diffusers import AutoencoderKL, UNet2DConditionModel

    if unet_checkpoint:
        path = Path(unet_checkpoint)
        unet = UNet2DConditionModel.from_pretrained(path / "unet", torch_dtype=pipe.unet.dtype)
        if unet.config.in_channels != 12:
            raise ValueError("Expected a trained 12-channel VGC U-Net")
        if unet.config.get("addition_embed_type") != pipe.unet.config.get("addition_embed_type"):
            raise ValueError("U-Net backbone mismatch: select --backbone matching the checkpoint")
        pipe.unet = unet
    if decoder_checkpoint:
        if pipe.unet.config.in_channels != 12:
            raise ValueError("Load original VGC U-Net before its calibrated decoder")
        path = Path(decoder_checkpoint)
        vae_path = path / "vae" if (path / "vae").is_dir() else path
        calibrated = AutoencoderKL.from_pretrained(vae_path, torch_dtype=pipe.vae.dtype)
        if calibrated.config.scaling_factor != pipe.vae.config.scaling_factor:
            raise ValueError("Calibration source VAE mismatch: scaling_factor")
        # Conditioning must use the same frozen encoder/post-quant transform
        # that generated the cache. Fail instead of silently changing latents.
        for name in ("encoder", "quant_conv", "post_quant_conv"):
            current = getattr(pipe.vae, name).state_dict()
            saved = getattr(calibrated, name).state_dict()
            if current.keys() != saved.keys() or any(
                    not torch.equal(current[k].cpu(), saved[k].cpu()) for k in current):
                raise ValueError(f"Calibration source VAE mismatch: {name}")
        pipe.vae.decoder.load_state_dict(calibrated.decoder.state_dict(), strict=True)
        del calibrated
    pipe.unet.eval()
    pipe.vae.eval()
    if pipe.da2 is not None:
        pipe.da2.eval()
