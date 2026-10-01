"""Generate candidate semantic masks for decoder calibration, offline only."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_output_directory(destination):
    """Allow retry of an empty directory left by an earlier loading failure."""
    if not destination.exists():
        return
    if not destination.is_dir():
        raise FileExistsError(f"Output is not a directory: {destination}")
    entries = list(destination.iterdir())
    if not entries:
        return
    if (len(entries) == 1 and entries[0].name == "masks"
            and entries[0].is_dir() and not entries[0].is_symlink()
            and not any(entries[0].iterdir())):
        return
    raise FileExistsError(f"Output contains existing results: {destination}. Use a new output directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_rgb_dir", required=True)
    parser.add_argument("--val_rgb_dir", required=True, help="Separate held-out scenes; not different seeds of training images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="nvidia/segformer-b5-finetuned-ade-640-640",
                        help="Local SegFormer directory or Hugging Face model ID")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    files = {}
    for split, folder in [("train", args.train_rgb_dir), ("val", args.val_rgb_dir)]:
        files[split] = sorted(p for p in Path(folder).rglob("*")
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file())
        if not files[split]:
            raise ValueError(f"No RGB images for {split}")
    destination = Path(args.output_dir).resolve()
    validate_output_directory(destination)
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=False)
    # Never fall back to pickle .bin weights. Recent Transformers correctly
    # refuse those on torch < 2.6; safetensors works with the project's torch 2.4.
    # For Hub repos with only .bin on main, Transformers can resolve the
    # safetensors conversion revision (the user's log already downloaded it).
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model, use_safetensors=True
    ).to(args.device).eval()
    sky_ids = [int(i) for i, name in model.config.id2label.items() if name.strip().lower() == "sky"]
    if len(sky_ids) != 1:
        raise ValueError("Model must have one class explicitly named 'sky'")
    # Do not leave new output directories behind if model loading fails.
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "masks").mkdir(exist_ok=True)
    seen, rows = set(), []
    with torch.no_grad():
        for split, images in files.items():
            for path in images:
                identity = hashlib.sha256(path.read_bytes()).hexdigest()
                if identity in seen:
                    raise ValueError(f"Duplicate RGB across input folders: {path}")
                seen.add(identity)
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                inputs = processor(images=rgb, return_tensors="pt").to(args.device)
                probs = model(**inputs).logits.float().softmax(1)
                # Resize two probability maps, not the entire 150-class volume.
                sky = probs[:, sky_ids[0]:sky_ids[0] + 1]
                confidence = probs.amax(1, keepdim=True)
                maps = F.interpolate(torch.cat([sky, confidence], 1), (rgb.height, rgb.width),
                                     mode="bilinear", align_corners=False)[0].cpu().numpy()
                mask = np.full((rgb.height, rgb.width), 128, dtype=np.uint8)
                mask[(maps[0] < .05) & (maps[1] > .5)] = 0
                mask[maps[0] > .9] = 255
                relative = f"masks/{identity}.png"
                Image.fromarray(mask).save(destination / relative)
                rows.append({"image": str(path.resolve()), "mask": relative, "split": split})
                print(f"{split}: {path.name}, sky={(mask == 255).mean():.3f}, unknown={(mask == 128).mean():.3f}", flush=True)
    (destination / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Candidate masks: {destination}. Inspect/correct cloud, fog, tree, wire and indoor boundaries before caching.")


if __name__ == "__main__":
    main()
