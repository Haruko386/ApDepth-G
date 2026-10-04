"""Offline CPU integration tests with real tiny Diffusers/Transformers modules."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from marigold.marigold_pipeline import MarigoldPipeline
from marigold.sdxl_depth_pipeline import SDXLDepthPipeline
from src.util.config_util import recursive_load_config
from src.util.model_overrides import apply_model_overrides
from src.util.pipeline_loader import load_depth_pipeline


class DummyPrior(nn.Module):
    def infer_batch(self, rgb):
        return rgb.mean(1, keepdim=True).expand_as(rgb)


def tiny_components(directory):
    root = Path(directory)
    (root / "vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1}))
    (root / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(root / "vocab.json"), str(root / "merges.txt"), model_max_length=8)
    text_cfg = dict(vocab_size=2, intermediate_size=24, num_hidden_layers=2,
                    max_position_embeddings=8, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    encoder = CLIPTextModel(CLIPTextConfig(hidden_size=8, num_attention_heads=2, **text_cfg))
    encoder2 = CLIPTextModelWithProjection(CLIPTextConfig(
        hidden_size=12, projection_dim=8, num_attention_heads=3, **text_cfg))
    unet = UNet2DConditionModel(
        sample_size=4, in_channels=4, out_channels=4, block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), layers_per_block=1,
        cross_attention_dim=20, attention_head_dim=2, norm_num_groups=4,
        addition_embed_type="text_time", addition_time_embed_dim=2,
        projection_class_embeddings_input_dim=20,
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8, 8, 8),
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        norm_num_groups=4, scaling_factor=0.13025,
    )
    return dict(unet=unet, vae=vae, text_encoder=encoder, text_encoder_2=encoder2,
                tokenizer=tokenizer, tokenizer_2=tokenizer,
                scheduler=EulerDiscreteScheduler(num_train_timesteps=100, prediction_type="epsilon"))


def tiny_pipeline(directory):
    with patch.object(MarigoldPipeline, "_load_depth_prior", return_value=DummyPrior()):
        return SDXLDepthPipeline(**tiny_components(directory))


def expand_input(pipe):
    # The real trainer expands 4 -> 12; reuse that implementation for inference.
    from src.trainer.marigold_trainer import MarigoldTrainer
    trainer = object.__new__(MarigoldTrainer)
    trainer.model = pipe
    trainer._replace_unet_conv_in()


class SDXLDepthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_dual_text_cache_scale_and_added_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = tiny_pipeline(directory).to("cpu")
            self.assertEqual(pipe.empty_text_embed.shape, (1, 8, 20))
            self.assertEqual(pipe.empty_pooled_embed.shape, (1, 8))
            self.assertIsNone(pipe.text_encoder)
            self.assertIsNone(pipe.text_encoder_2)
            self.assertIsInstance(pipe.scheduler, DDIMScheduler)
            self.assertFalse(pipe.scheduler.config.clip_sample)
            self.assertEqual(pipe.depth_latent_scale_factor, .13025)
            expand_input(pipe)
            captured = []
            hook = pipe.unet.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
            pred = pipe.predict_noise(torch.randn(2, 12, 4, 6), torch.tensor([1, 2]), pipe.empty_text_embed.expand(2, -1, -1))
            hook.remove()
            self.assertEqual(pred.shape, (2, 4, 4, 6))
            torch.testing.assert_close(captured[0]["added_cond_kwargs"]["time_ids"],
                                       torch.tensor([[32., 48., 0., 0., 32., 48.]]).expand(2, -1))
            image = torch.randn(1, 3, 32, 48)
            encoded = pipe.encode_rgb(image)
            expected = pipe.vae.encode(image).latent_dist.mode() * .13025
            torch.testing.assert_close(encoded, expected)
            expected_depth = pipe.vae.decode(encoded / .13025).sample.mean(1, keepdim=True)
            torch.testing.assert_close(pipe.decode_depth(encoded), expected_depth)

    def test_multistep_inference_is_seeded_and_uses_every_step(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = tiny_pipeline(directory).to("cpu")
            expand_input(pipe)
            pipe.unet.eval()
            image = torch.randn(1, 3, 32, 48)
            calls = []
            handle = pipe.unet.register_forward_hook(lambda *args: calls.append(1))
            a = pipe.single_infer(image, 4, generator=torch.Generator().manual_seed(42), show_pbar=False)
            self.assertEqual(len(calls), 4)
            b = pipe.single_infer(image, 4, generator=torch.Generator().manual_seed(42), show_pbar=False)
            handle.remove()
            torch.testing.assert_close(a, b)
            self.assertEqual(a.shape, (1, 1, 32, 48))
            self.assertTrue(torch.isfinite(a).all())

    def test_actual_from_pretrained_with_sdxl_component_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = tiny_components(root)
            index = {"_class_name": "StableDiffusionXLPipeline"}
            for name, component in parts.items():
                component.save_pretrained(root / name)
                library = "diffusers" if name in ("unet", "vae", "scheduler") else "transformers"
                index[name] = [library, type(component).__name__]
            (root / "model_index.json").write_text(json.dumps(index))
            with patch.object(MarigoldPipeline, "_load_depth_prior", return_value=DummyPrior()):
                pipe = load_depth_pipeline(root, backbone="sdxl", local_files_only=True)
            self.assertIsInstance(pipe, SDXLDepthPipeline)
            self.assertIsNone(pipe.text_encoder_2)
            pipe.to("cpu")
            expand_input(pipe)
            self.assertTrue(torch.isfinite(pipe.single_infer(torch.randn(1, 3, 32, 32), 2, generator=None, show_pbar=False)).all())

    def test_checkpointing_training_resume_and_inference_override(self):
        from src.trainer import marigold_trainer as trainer_module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = recursive_load_config("config/train_sdxl_demo.yaml")
            # Exercise real SDXL conditions and activation checkpointing on CPU;
            # CUDA BF16 and bitsandbytes are separately covered by the GPU probe.
            cfg.trainer.mixed_precision = "no"
            cfg.trainer.report_cuda_memory = False
            cfg.optimizer.name = "Adam"
            cfg.max_iter = cfg.max_epoch = 1
            cfg.trainer.save_period = 1
            cfg.trainer.backup_period = 0
            cfg.lr_scheduler.kwargs.warmup_steps = 0
            pipe = tiny_pipeline(root)
            scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            mask = torch.ones(1, 32, 32, dtype=torch.bool)
            mask[..., :16, :] = False
            sample = {"rgb_norm": torch.randn(3, 32, 32), "depth_raw_norm": torch.randn(1, 32, 32), "valid_mask_raw": mask}
            loader = torch.utils.data.DataLoader([sample] * 2, batch_size=1)
            before = pipe.unet.conv_out.weight.detach().clone()
            vae_before = pipe.vae.encoder.conv_in.weight.detach().clone()
            (root / "ckpt").mkdir()
            with patch.object(trainer_module.DDPMScheduler, "from_pretrained", return_value=scheduler), \
                    patch.object(trainer_module, "tb_logger", MagicMock()):
                trainer = trainer_module.MarigoldTrainer(cfg, pipe, loader, "cpu", ".", str(root / "ckpt"), ".", ".", 2)
                trainer.train()
                self.assertFalse(torch.equal(before, pipe.unet.conv_out.weight))
                torch.testing.assert_close(vae_before, pipe.vae.encoder.conv_in.weight)
                pipe2 = tiny_pipeline(root)
                trainer2 = trainer_module.MarigoldTrainer(cfg, pipe2, loader, "cpu", ".", str(root / "ckpt"), ".", ".", 2)
                trainer2.load_checkpoint(root / "ckpt/latest")
                self.assertEqual(trainer2.effective_iter, 1)
                torch.testing.assert_close(pipe.unet.conv_out.weight, pipe2.unet.conv_out.weight)
                self.assertEqual(trainer2.lr_scheduler.last_epoch, trainer.lr_scheduler.last_epoch)
                self.assertTrue(trainer2.optimizer.state)
                # Actually continue the restored optimizer, not just deserialize.
                cfg.max_iter = 2
                trainer2.max_iter = 2
                trainer2.max_epoch = 2
                trainer2.train()
                self.assertEqual(trainer2.effective_iter, 2)
            target = tiny_pipeline(root)
            apply_model_overrides(target, root / "ckpt/iter_000002")
            self.assertEqual(target.unet.config.in_channels, 12)
            self.assertTrue(torch.isfinite(target.single_infer(torch.randn(1, 3, 32, 32), 3, generator=None, show_pbar=False)).all())

    def test_backbone_mismatch_rejected_and_configs_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = tiny_pipeline(directory)
            expand_input(pipe)
            pipe.unet.save_pretrained(Path(directory) / "unet")
            pipe.unet.register_to_config(addition_embed_type=None)
            with self.assertRaisesRegex(ValueError, "backbone mismatch"):
                apply_model_overrides(pipe, directory)
        sd2 = recursive_load_config("config/train_marigold.yaml")
        demo = recursive_load_config("config/train_sdxl_demo.yaml")
        probe = recursive_load_config("config/train_sdxl_probe.yaml")
        self.assertEqual(sd2.dataloader.max_train_batch_size, 7)
        self.assertEqual(sd2.optimizer.name, "Adam")
        self.assertEqual(demo.dataloader.effective_batch_size, 42)
        self.assertEqual(demo.dataloader.max_train_batch_size, 1)
        self.assertEqual(demo.max_iter, 23000)
        self.assertTrue(demo.validity_guided_completion.enabled)
        self.assertEqual(demo.validity_guided_completion.mode, "legacy")
        self.assertEqual(demo.validity_guided_completion.anchor_weight, 0.02)
        self.assertEqual(demo.validity_guided_completion.smooth_weight, 0.005)
        self.assertEqual(probe.validity_guided_completion.mode, "legacy")
        self.assertEqual(probe.max_iter, 2)
        self.assertTrue(probe.trainer.memory_probe)


if __name__ == "__main__":
    unittest.main()
