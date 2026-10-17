import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

from script.prepare_sky_masks import main, validate_output_directory


class PrepareSkyMasksTest(unittest.TestCase):
    def test_empty_retry_allowed_existing_results_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "labels"
            validate_output_directory(root)
            (root / "masks").mkdir(parents=True)
            validate_output_directory(root)
            saved = root / "masks/existing.png"
            saved.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                validate_output_directory(root)
            self.assertEqual(saved.read_bytes(), b"keep")
            saved.unlink()
            (root / "manifest.jsonl").write_text("[]", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                validate_output_directory(root)

    def test_safe_weights_generate_masks_without_torch_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 32, 64],
                                     num_attention_heads=[1, 2, 4, 8], decoder_hidden_size=16,
                                     num_labels=2, id2label={0: "sky", 1: "road"})
            SegformerForSemanticSegmentation(config).save_pretrained(root / "model", safe_serialization=True)
            SegformerImageProcessor(size={"height": 32, "width": 32}).save_pretrained(root / "model")
            for split, color in [("train", "red"), ("val", "blue")]:
                (root / split).mkdir()
                Image.new("RGB", (16, 24), color).save(root / split / "image.png")
            # Reproduce the empty masks directory shown in the user's screenshot.
            (root / "labels/masks").mkdir(parents=True)
            argv = ["prepare_sky_masks", "--train_rgb_dir", str(root / "train"),
                    "--val_rgb_dir", str(root / "val"), "--output_dir", str(root / "labels"),
                    "--model", str(root / "model"), "--device", "cpu"]
            with patch.object(sys, "argv", argv), patch.object(torch, "load", side_effect=AssertionError("pickle load forbidden")):
                main()
            rows = [json.loads(line) for line in (root / "labels/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["split"] for row in rows}, {"train", "val"})
            for row in rows:
                with Image.open(root / "labels" / row["mask"]) as mask:
                    self.assertEqual(mask.size, (16, 24))
                    self.assertTrue(set(mask.getdata()) <= {0, 128, 255})


if __name__ == "__main__":
    unittest.main()
