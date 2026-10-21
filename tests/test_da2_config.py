import tempfile
import unittest
import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import torch

from src.util.config_util import recursive_load_config
from src.util.da2_config import load_da2_config, resolve_da2_config, da2_checkpoint_path, REPO_ROOT


class DA2ConfigTest(unittest.TestCase):
    def test_training_and_inference_share_default(self):
        cfg = recursive_load_config("config/train_marigold.yaml")
        resolved = resolve_da2_config(cfg.da2)
        self.assertEqual(resolved, resolve_da2_config())
        self.assertEqual(resolved["encoder"], "vitg")
        self.assertEqual(resolved["out_channels"], [1536] * 4)
        self.assertEqual(resolve_da2_config(resolved), resolved)

    def test_encoder_switch_selects_matching_architecture_and_weights(self):
        for encoder, features in [("vits", 64), ("vitb", 128), ("vitl", 256), ("vitg", 384)]:
            cfg = load_da2_config()
            cfg["encoder"] = encoder
            resolved = resolve_da2_config(cfg)
            self.assertEqual(resolved["features"], features)
            self.assertEqual(da2_checkpoint_path(resolved),
                             REPO_ROOT / f"DA2/checkpoints/depth_anything_v2_{encoder}.pth")

    def test_custom_config_and_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.yaml"
            path.write_text("da2:\n  encoder: vits\n  features: 64\n"
                            "  out_channels: [48, 96, 192, 384]\n"
                            "  checkpoint: custom/weights.pth\n", encoding="utf-8")
            cfg = resolve_da2_config(load_da2_config(path))
            self.assertEqual(da2_checkpoint_path(cfg), REPO_ROOT / "custom/weights.pth")
            cfg["checkpoint"] = str(Path(directory) / "weights.pth")
            self.assertEqual(da2_checkpoint_path(cfg), Path(directory) / "weights.pth")
            path.write_text("wrong: value\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_da2_config(path)

    def test_unknown_encoder_rejected(self):
        with self.assertRaises(ValueError):
            resolve_da2_config({"encoder": "unknown"})

    def test_pipeline_loads_custom_prior_and_keeps_it_frozen_fp32(self):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextConfig, CLIPTextModel
        # The repository's tests do not require the external DA2 source/weights.
        stub = ModuleType("DA2.depth_anything_v2.dpt")
        stub.DepthAnythingV2 = torch.nn.Module
        with patch.dict(sys.modules, {"DA2.depth_anything_v2.dpt": stub}):
            module = importlib.import_module("marigold.marigold_pipeline")

        class TinyPrior(torch.nn.Linear):
            def __init__(self, **architecture):
                super().__init__(1, 1)
                self.architecture = architecture

        unet = UNet2DConditionModel(
            sample_size=8, in_channels=12, out_channels=4, layers_per_block=1,
            down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
            block_out_channels=(8,), norm_num_groups=4, cross_attention_dim=8,
        )
        vae = AutoencoderKL(block_out_channels=(8,), norm_num_groups=4)
        text_encoder = CLIPTextModel(CLIPTextConfig(
            vocab_size=4, hidden_size=8, intermediate_size=8,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8,
        ))
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "DepthAnythingV2", TinyPrior):
            checkpoint = Path(directory) / "prior.pth"
            torch.save(TinyPrior().state_dict(), checkpoint)
            config = load_da2_config()
            config.update(encoder="vits", checkpoint=str(checkpoint))
            pipe = module.MarigoldPipeline(
                unet, vae, DDIMScheduler(), text_encoder, None, da2_config=config,
            )
            self.assertEqual(pipe.da2.architecture["features"], 64)
            self.assertFalse(pipe.da2.training)
            self.assertFalse(pipe.da2.weight.requires_grad)
            self.assertIs(pipe.to("cpu", torch.float16, silence_dtype_warnings=True), pipe)
            self.assertEqual(pipe.unet.dtype, torch.float16)
            self.assertEqual(pipe.da2.weight.dtype, torch.float32)
            self.assertEqual(pipe.da2.weight.device, pipe.device)
            # Exercise Diffusers' actual optional constructor argument handling.
            pipe.save_config(directory)
            loaded = module.MarigoldPipeline.from_pretrained(directory, **pipe.components)
            self.assertEqual(dict(loaded.config.da2_config), dict(pipe.config.da2_config))
            self.assertEqual(loaded.da2.architecture, pipe.da2.architecture)
            # Exercise actual denoising input construction, not just config flags.
            pipe.da2.infer_batch = unittest.mock.MagicMock(return_value=torch.ones(1, 3, 8, 8))
            pipe.empty_text_embed = torch.zeros(1, 1, 8)
            captured = []

            def predict(sample, *args, **kwargs):
                captured.append(sample[:, 4:8].detach().clone())
                return type("Prediction", (), {"sample": torch.zeros_like(sample[:, :4])})()

            with patch.object(pipe, "encode_rgb", side_effect=lambda x: x.mean(1, keepdim=True).repeat(1, 4, 1, 1)), \
                    patch.object(pipe, "decode_depth", return_value=torch.zeros(1, 1, 8, 8)), \
                    patch.object(pipe.unet, "forward", side_effect=predict):
                for enabled in (False, True):
                    pipe.da2_conditioning_enabled = enabled
                    captured.clear()
                    pipe.da2.infer_batch.reset_mock()
                    pipe.single_infer(torch.zeros(1, 3, 8, 8), num_inference_steps=2,
                                      show_pbar=False, generator=None)
                    for prior_channels in captured:
                        torch.testing.assert_close(prior_channels, torch.full_like(prior_channels, float(enabled)))
                    self.assertEqual(pipe.da2.infer_batch.call_count, int(enabled))


if __name__ == "__main__":
    unittest.main()
