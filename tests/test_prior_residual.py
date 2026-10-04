import importlib
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn

from src.util.config_util import recursive_load_config
from src.util.prior_residual import (
    ABSOLUTE, PRIOR_RESIDUAL, compose_depth_latent, diffusion_target,
    load_parameterization, save_parameterization,
)
from src.util.model_overrides import load_unet_checkpoint
from src.util.diffusion_training import (
    conservative_latent_valid_mask, fill_invalid_depth_with_prior,
    make_v_prediction_schedulers, min_snr_v_weight, terminal_noise_fade,
)


class PriorResidualTest(unittest.TestCase):
    def test_parameterization_round_trip_and_prior_anchor(self):
        prior = torch.randn(2, 4, 3, 5, requires_grad=True)
        depth = torch.randn_like(prior, requires_grad=True)
        state = diffusion_target(depth, prior, PRIOR_RESIDUAL, .5)
        restored = compose_depth_latent(state, prior.detach(), PRIOR_RESIDUAL, .5)
        torch.testing.assert_close(restored, depth)
        anchored = compose_depth_latent(torch.zeros_like(prior), prior, PRIOR_RESIDUAL, .5)
        torch.testing.assert_close(anchored, prior)
        state.sum().backward()
        self.assertIsNone(prior.grad)
        torch.testing.assert_close(
            compose_depth_latent(state, prior, ABSOLUTE, .5), state
        )

    def test_checkpoint_metadata_and_active_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth_parameterization.json"
            save_parameterization(path, PRIOR_RESIDUAL, .75)
            self.assertEqual(load_parameterization(path), (PRIOR_RESIDUAL, .75))
            self.assertEqual(load_parameterization(path.with_name("missing.json")), (ABSOLUTE, 1.0))
        cfg = recursive_load_config("config/train_marigold.yaml")
        self.assertEqual(cfg.depth_parameterization.mode, PRIOR_RESIDUAL)
        self.assertEqual(cfg.depth_parameterization.residual_scale, 1.0)
        self.assertTrue(cfg.validity_guided_completion.enabled)
        self.assertEqual(cfg.validity_guided_completion.mode, "target_censoring")
        self.assertEqual(cfg.validity_guided_completion.boundary_margin, 2)
        self.assertEqual(cfg.multi_res_noise.strength, 0.9)
        self.assertEqual(cfg.diffusion_schedule.prediction_type, "v_prediction")
        self.assertTrue(cfg.diffusion_schedule.rescale_betas_zero_snr)
        self.assertEqual(cfg.diffusion_schedule.timestep_spacing, "trailing")

    def test_v_schedule_and_invalid_depth_censoring(self):
        base = DDPMScheduler(num_train_timesteps=100, prediction_type="epsilon")
        train_scheduler, infer_scheduler = make_v_prediction_schedulers(base, base)
        self.assertEqual(train_scheduler.config.prediction_type, "v_prediction")
        self.assertEqual(train_scheduler.alphas_cumprod[-1].item(), 0)
        self.assertEqual(infer_scheduler.config.timestep_spacing, "trailing")
        infer_scheduler.set_timesteps(10)
        self.assertEqual(infer_scheduler.timesteps[0].item(), 99)
        endpoint_weight = min_snr_v_weight(
            train_scheduler.alphas_cumprod[-1:].reshape(1, 1, 1, 1)
        )
        torch.testing.assert_close(
            endpoint_weight, torch.full_like(endpoint_weight, 0.05)
        )
        fade = terminal_noise_fade(torch.tensor([0, 949, 999]), 1000, 0.1)
        self.assertEqual(fade[0].item(), 1.0)
        self.assertAlmostEqual(fade[1].item(), 0.5005, places=3)
        self.assertEqual(fade[2].item(), 0.0)

        depth = torch.full((1, 1, 16, 24), -1.0)
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[..., :8, :8] = False
        prior = torch.full((1, 3, 16, 24), 0.75)
        filled = fill_invalid_depth_with_prior(depth, valid, prior)
        self.assertTrue(torch.all(filled[..., :8, :8] == 0.75))
        self.assertTrue(torch.all(filled[..., 8:, 8:] == -1.0))

        no_margin = conservative_latent_valid_mask(valid, (2, 3), 0)
        self.assertFalse(no_margin[0, 0, 0, 0])
        self.assertTrue(no_margin[0, 0, 1, 1])
        with_margin = conservative_latent_valid_mask(valid, (2, 3), 1)
        self.assertLess(with_margin.sum(), no_margin.sum())

    def test_inference_loader_restores_parameterization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unet").mkdir()
            save_parameterization(
                root / "depth_parameterization.json", PRIOR_RESIDUAL, .75
            )
            DDIMScheduler(
                num_train_timesteps=100,
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
            ).save_pretrained(root / "scheduler")

            class FakeUnet:
                config = SimpleNamespace(in_channels=12)

                def eval(self):
                    return self

            class FakePipe:
                unet = None

                def set_depth_parameterization(self, mode, residual_scale):
                    self.mode = mode
                    self.scale = residual_scale

            pipe = FakePipe()
            fake_module = ModuleType("marigold.modules.unet_2d_condition")
            fake_class = MagicMock()
            fake_class.from_pretrained.return_value = FakeUnet()
            fake_module.UNet2DConditionModel = fake_class
            module_name = "marigold.modules.unet_2d_condition"
            previous = sys.modules.get(module_name)
            sys.modules[module_name] = fake_module
            try:
                mode, scale = load_unet_checkpoint(pipe, root, torch.float32)
            finally:
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous
            self.assertEqual((mode, scale), (PRIOR_RESIDUAL, .75))
            self.assertEqual((pipe.mode, pipe.scale), (PRIOR_RESIDUAL, .75))
            self.assertEqual(pipe.scheduler.config.prediction_type, "v_prediction")

    def test_real_trainer_uses_residual_state_and_safe_conv_initialization(self):
        fake_da2 = ModuleType("DA2.depth_anything_v2.dpt")
        fake_da2.DepthAnythingV2 = nn.Module
        module_name = "DA2.depth_anything_v2.dpt"
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = fake_da2
        try:
            trainer_module = importlib.import_module("src.trainer.marigold_trainer")
        finally:
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

        class TinyUnet(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = {"in_channels": 4}
                self.conv_in = nn.Conv2d(4, 4, 3, padding=1)

            def enable_xformers_memory_efficient_attention(self):
                pass

            def forward(self, x, timestep, text):
                return SimpleNamespace(sample=self.conv_in(x))

        class Prior(nn.Module):
            def infer_batch(self, rgb):
                return rgb.mean(1, keepdim=True).expand_as(rgb)

        class Pipeline(nn.Module):
            def __init__(self):
                super().__init__()
                self.unet = TinyUnet()
                self.vae = nn.Conv2d(3, 4, 8, stride=8)
                self.text_encoder = nn.Linear(1, 1)
                self.da2 = Prior()
                self.scheduler = DDPMScheduler(num_train_timesteps=100, prediction_type="epsilon")

            def encode_empty_text(self):
                self.empty_text_embed = torch.zeros(1, 1, 1)

            def encode_rgb(self, image):
                return self.vae(image)

            def set_depth_parameterization(self, mode, residual_scale):
                self.depth_prediction_mode = mode
                self.depth_residual_scale = residual_scale

        torch.manual_seed(3)
        cfg = recursive_load_config("config/train_marigold.yaml")
        cfg.max_iter = cfg.max_epoch = 1
        cfg.lr_scheduler.kwargs.warmup_steps = 0
        cfg.trainer.save_period = cfg.trainer.backup_period = 0
        cfg.validity_guided_completion.boundary_margin = 0
        pipe = Pipeline()
        original = pipe.unet.conv_in.weight.detach().clone()
        valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        valid[..., :16, :] = False
        sample = {
            "rgb_norm": torch.randn(3, 32, 32),
            "depth_raw_norm": torch.randn(1, 32, 32),
            "valid_mask_raw": valid[0],
        }
        loader = torch.utils.data.DataLoader([sample], batch_size=1)
        with patch.object(
            trainer_module.DDPMScheduler, "from_pretrained", return_value=pipe.scheduler
        ), patch.object(trainer_module, "tb_logger", MagicMock()):
            trainer = trainer_module.MarigoldTrainer(
                cfg, pipe, loader, "cpu", ".", ".", ".", ".", 1
            )
            torch.testing.assert_close(pipe.unet.conv_in.weight[:, :4], original * .5)
            self.assertEqual(pipe.unet.conv_in.weight[:, 4:8].abs().sum().item(), 0)
            torch.testing.assert_close(pipe.unet.conv_in.weight[:, 8:12], original * .5)
            self.assertEqual(pipe.depth_prediction_mode, PRIOR_RESIDUAL)
            self.assertFalse(any(parameter.requires_grad for parameter in pipe.da2.parameters()))
            self.assertEqual(trainer.prediction_type, "v_prediction")
            self.assertEqual(
                trainer.training_noise_scheduler.alphas_cumprod[-1].item(), 0
            )
            self.assertEqual(pipe.scheduler.config.timestep_spacing, "trailing")
            trainer.save_checkpoint = MagicMock()
            trainer.train()
        self.assertEqual(trainer.effective_iter, 1)
        self.assertTrue(torch.isfinite(pipe.unet.conv_in.weight).all())


if __name__ == "__main__":
    unittest.main()
