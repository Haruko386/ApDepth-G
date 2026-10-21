"""Download the Diffusers SSD-1B base and optional external model weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SSD_REPO = "segmind/SSD-1B"
SSD_REVISION = "60987f37e94cd59c36b1cba832b9f97b57395a10"
# Ordinary FP32 safetensors support both FP32 training and explicit AMP. Avoid
# duplicate fp16 variants and the root-level A1111 monolithic checkpoints.
SSD_FILES = [
    "model_index.json", "scheduler/scheduler_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
    *[f"{tokenizer}/{name}" for tokenizer in ("tokenizer", "tokenizer_2")
      for name in ("merges.txt", "vocab.json", "tokenizer_config.json", "special_tokens_map.json")],
]
DA2_REPOS = {
    "vits": "depth-anything/Depth-Anything-V2-Small",
    "vitb": "depth-anything/Depth-Anything-V2-Base",
    "vitl": "depth-anything/Depth-Anything-V2-Large",
}


def download_ssd(destination, revision=SSD_REVISION):
    from huggingface_hub import snapshot_download
    snapshot_download(SSD_REPO, revision=revision, local_dir=str(destination),
                      allow_patterns=SSD_FILES)
    missing = [name for name in SSD_FILES if not (destination / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete SSD Diffusers checkpoint: {missing}")
    config = json.loads((destination / "unet/config.json").read_text())
    if config.get("addition_embed_type") != "text_time" or config.get("in_channels") != 4:
        raise ValueError("Expected the original SSD text_time, 4-channel base U-Net")


def download_da2(config_path, repo_id=None):
    from huggingface_hub import hf_hub_download
    from src.util.da2_config import load_da2_config, resolve_da2_config, da2_checkpoint_path
    config = resolve_da2_config(load_da2_config(config_path))
    target = da2_checkpoint_path(config)
    if target.is_file():
        print(f"Using existing DA2 weights: {target}")
        return
    repo_id = repo_id or DA2_REPOS.get(config["encoder"])
    if not repo_id:
        raise ValueError(
            "Main uses DA2 vitg. Supply your existing matching Giant checkpoint at "
            f"{target}, set da2.checkpoint in {config_path}, or pass --da2-repo for a "
            "repository you can access. The downloader does not replace vitg with a smaller encoder."
        )
    source = hf_hub_download(repo_id, f"depth_anything_v2_{config['encoder']}.pth")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(target)
    print(f"DA2 weights: {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_ckpt_dir", default=os.environ.get("BASE_CKPT_DIR", "checkpoint"))
    parser.add_argument("--revision", default=SSD_REVISION)
    parser.add_argument("--with-da2", action="store_true")
    parser.add_argument("--da2-config", default=str(ROOT / "config/da2.yaml"))
    parser.add_argument("--da2-repo", help="Optional repository containing the matching DA2 .pth")
    parser.add_argument("--with-sky-model", action="store_true")
    parser.add_argument("--sky-repo", default="facebook/mask2former-swin-large-cityscapes-semantic")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without network or file writes")
    args = parser.parse_args(argv)
    base = Path(args.base_ckpt_dir).expanduser().resolve()
    print(json.dumps({"repo": SSD_REPO, "revision": args.revision,
                      "destination": str(base / "SSD-1B"), "files": SSD_FILES,
                      "da2_config": args.da2_config if args.with_da2 else None,
                      "sky_repo": args.sky_repo if args.with_sky_model else None}, indent=2))
    if args.dry_run:
        return 0
    # Validate the optional prior first, before spending time downloading SSD.
    if args.with_da2:
        download_da2(args.da2_config, args.da2_repo)
    download_ssd(base / "SSD-1B", args.revision)
    if args.with_sky_model:
        from huggingface_hub import snapshot_download
        target = base / "mask2former"
        snapshot_download(args.sky_repo, local_dir=str(target),
                          allow_patterns=["*.json", "*.safetensors"])
        if not list(target.glob("*.safetensors")):
            raise FileNotFoundError(f"Sky segmentation requires safetensors weights: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
