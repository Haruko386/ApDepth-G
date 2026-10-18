"""Resolve the shared DA2 configuration without loading model weights."""

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DA2_CONFIG = REPO_ROOT / "config" / "da2.yaml"


def load_da2_config(path=DEFAULT_DA2_CONFIG):
    config = OmegaConf.load(path)
    if "da2" not in config:
        raise ValueError(f"DA2 config must contain a 'da2' section: {path}")
    return OmegaConf.to_container(config.da2, resolve=True)


def resolve_da2_config(config=None):
    config = load_da2_config() if config is None else dict(config)
    encoder = config["encoder"]
    if encoder not in ("vits", "vitb", "vitl", "vitg"):
        raise ValueError(f"Unsupported DA2 encoder: {encoder}")
    # Accept either a YAML profile collection or an already resolved pipeline config.
    architecture = config["model_configs"][encoder] if "model_configs" in config else config
    features = architecture["features"]
    channels = list(architecture["out_channels"])
    if (not isinstance(features, int) or features <= 0 or len(channels) != 4
            or any(not isinstance(c, int) or c <= 0 for c in channels)):
        raise ValueError("DA2 needs positive features and four positive out_channels")
    checkpoint = config.get("checkpoint") or f"DA2/checkpoints/depth_anything_v2_{encoder}.pth"
    return {"encoder": encoder, "features": features, "out_channels": channels,
            "checkpoint": str(checkpoint)}


def da2_checkpoint_path(config):
    path = Path(config["checkpoint"]).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path
