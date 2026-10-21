"""Load a main-training U-Net and its matching scheduler for inference."""
from pathlib import Path
import json

from omegaconf import OmegaConf

from src.util.da2_config import load_da2_config


def checkpoint_inference_settings(unet_checkpoint):
    """Read checkpoint metadata, with a fallback for existing training runs."""
    if not unet_checkpoint:
        return {}
    path = Path(unet_checkpoint)
    metadata_path = path / "inference_settings.json"
    if metadata_path.is_file():
        settings = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        # Training layout: <run>/checkpoint/<iteration> and <run>/config.yaml.
        candidates = (path / "config.yaml", path.parent.parent / "config.yaml")
        config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if config_path is None:
            raise FileNotFoundError(
                f"No inference_settings.json or run config.yaml found for {path}. "
                "Keep the checkpoint in its training run or copy its saved config.yaml into it."
            )
        cfg = OmegaConf.load(config_path)
        settings = {
            "backbone": cfg.get("model", {}).get("backbone", "sd2"),
            "da2_conditioning_enabled": cfg.get("da2_conditioning_enabled", True),
            "da2_config": OmegaConf.to_container(cfg.da2, resolve=True) if cfg.get("da2") is not None else None,
            "diffusion_schedule": OmegaConf.to_container(cfg.diffusion_schedule, resolve=True)
            if cfg.get("diffusion_schedule") is not None else None,
        }
    if not isinstance(settings.get("da2_conditioning_enabled"), bool):
        raise ValueError("Checkpoint da2_conditioning_enabled must be a boolean")
    return settings


def checkpoint_pipeline_kwargs(unet_checkpoint=None, da2_config_path=None):
    settings = checkpoint_inference_settings(unet_checkpoint)
    kwargs = {}
    if settings:
        kwargs["da2_conditioning_enabled"] = settings["da2_conditioning_enabled"]
    if da2_config_path:
        kwargs["da2_config"] = load_da2_config(da2_config_path)
    elif settings.get("da2_config") is not None:
        kwargs["da2_config"] = settings["da2_config"]
    return kwargs


def apply_model_overrides(pipe, unet_checkpoint=None):
    from diffusers import DDIMScheduler, UNet2DConditionModel

    if unet_checkpoint:
        path = Path(unet_checkpoint)
        settings = checkpoint_inference_settings(path)
        expected_backbone = settings.get("backbone")
        actual_backbone = getattr(pipe, "depth_backbone", None)
        if expected_backbone and actual_backbone and expected_backbone != actual_backbone:
            raise ValueError(f"U-Net backbone mismatch: checkpoint is {expected_backbone}, selected {actual_backbone}")
        unet = UNet2DConditionModel.from_pretrained(path / "unet", torch_dtype=pipe.unet.dtype)
        if unet.config.in_channels not in (8, 12):
            raise ValueError("Expected a trained 8- or 12-channel depth U-Net")
        if unet.config.in_channels == 8 and settings["da2_conditioning_enabled"]:
            raise ValueError("8-channel checkpoint cannot use DA2 input conditioning")
        if settings.get("unet_in_channels", unet.config.in_channels) != unet.config.in_channels:
            raise ValueError("Checkpoint input channels disagree with inference settings")
        if unet.config.get("addition_embed_type") != pipe.unet.config.get("addition_embed_type"):
            raise ValueError("U-Net backbone mismatch: select --backbone matching the checkpoint")
        # SSD and full SDXL share text_time, but their U-Net layouts differ.
        for key in ("block_out_channels", "down_block_types", "up_block_types",
                    "mid_block_type", "transformer_layers_per_block",
                    "reverse_transformer_layers_per_block", "cross_attention_dim"):
            actual = json.loads(json.dumps(unet.config.get(key)))
            expected = json.loads(json.dumps(pipe.unet.config.get(key)))
            if actual != expected:
                raise ValueError(f"U-Net backbone mismatch: {key} differs from the base checkpoint")
        pipe.unet = unet
        scheduler_path = path / "scheduler"
        if not scheduler_path.is_dir():
            raise FileNotFoundError(f"Checkpoint scheduler is missing: {scheduler_path}")
        pipe.scheduler = DDIMScheduler.from_pretrained(scheduler_path)
        # The endpoint ablation deliberately uses nonzero terminal SNR and
        # leading spacing. Validate against its own training settings.
        expected = settings.get("diffusion_schedule")
        if expected:
            for key in ("prediction_type", "rescale_betas_zero_snr", "timestep_spacing"):
                if key in expected and getattr(pipe.scheduler.config, key, None) != expected[key]:
                    raise ValueError(f"Checkpoint scheduler disagrees with training config: {key}")
        pipe.da2_conditioning_enabled = settings["da2_conditioning_enabled"]
        pipe.register_to_config(da2_conditioning_enabled=pipe.da2_conditioning_enabled)
    pipe.unet.eval()
    pipe.vae.eval()
    if pipe.da2 is not None:
        pipe.da2.eval()
