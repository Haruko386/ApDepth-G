import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from diffusers import DDPMScheduler

from src.util.config_util import recursive_load_config
from src.util.loss import LatentGradLoss
from src.util.lr_scheduler import IterExponential
from src.util.diffusion_training import (
    align_inference_scheduler, align_training_scheduler,
    latent_validity_masks, min_snr_weight, terminal_noise_fade,
)


class DiffusionTrainingTest(unittest.TestCase):
    def test_parameterizations_have_equal_x0_objective(self):
        alpha = torch.tensor([0.0001, 0.1, 0.5, 0.99]).reshape(-1, 1, 1, 1)
        x0 = torch.randn(4, 4, 3, 3)
        noise = torch.randn_like(x0)
        xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise
        estimate = x0 + torch.randn_like(x0) * 0.2
        expected = (alpha / (1 - alpha)).clamp(max=5) * (estimate - x0).square()
        for mode in ("epsilon", "v_prediction", "sample"):
            if mode == "epsilon":
                pred, target = (xt - alpha.sqrt() * estimate) / (1 - alpha).sqrt(), noise
            elif mode == "v_prediction":
                pred = (alpha.sqrt() * xt - estimate) / (1 - alpha).sqrt()
                target = alpha.sqrt() * noise - (1 - alpha).sqrt() * x0
            else:
                pred, target = estimate, x0
            weight = min_snr_weight(alpha, mode)
            torch.testing.assert_close(weight * (pred - target).square(), expected, atol=1e-5, rtol=1e-4)

    def test_zero_terminal_schedule_and_nonzero_endpoint_weight(self):
        config = {
            "prediction_type": "v_prediction",
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        }
        base = DDPMScheduler(num_train_timesteps=100, prediction_type="epsilon")
        training = align_training_scheduler(base, config)
        inference = align_inference_scheduler(base, config)
        self.assertEqual(training.config.prediction_type, "v_prediction")
        self.assertEqual(training.alphas_cumprod[-1].item(), 0)
        self.assertFalse(inference.config.clip_sample)
        inference.set_timesteps(10)
        self.assertEqual(inference.timesteps[0].item(), 99)
        endpoint_weight = min_snr_weight(
            training.alphas_cumprod[-1:].reshape(1, 1, 1, 1),
            "v_prediction", gamma=5, snr_floor=.05,
        )
        torch.testing.assert_close(endpoint_weight, torch.full_like(endpoint_weight, .05))
        with self.assertRaisesRegex(ValueError, "requires v_prediction"):
            align_training_scheduler(base, {**config, "prediction_type": "epsilon"})

    def test_mixed_cells_are_excluded(self):
        pixels = torch.ones(1, 1, 16, 24, dtype=torch.bool)
        pixels[..., :8, :8] = False
        pixels[..., 0, 8] = False
        valid, invalid = latent_validity_masks(pixels)
        self.assertTrue(invalid[0, 0, 0, 0])
        self.assertFalse(invalid[0, 0, 0, 1])
        self.assertFalse(valid[0, 0, 0, 1])
        self.assertTrue(valid[0, 0, 1, 1])

    def test_structured_noise_fades_only_near_terminal_endpoint(self):
        timesteps = torch.tensor([0, 899, 949, 999])
        fade = terminal_noise_fade(timesteps, 1000, .1)
        torch.testing.assert_close(fade[:2], torch.ones(2))
        self.assertAlmostEqual(fade[2].item(), 0.5005, places=3)
        self.assertEqual(fade[3].item(), 0)

    def test_valid_gradient_does_not_supervise_across_invalid_boundary(self):
        pred = torch.tensor([[[[1., 200.], [1., 200.]]]], requires_grad=True)
        mask = torch.tensor([[[[True, False], [True, False]]]])
        loss = LatentGradLoss()(pred, torch.ones_like(pred), mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(pred.grad.abs().sum().item(), 0)

    def test_config_sample_budget_and_lr_exposure(self):
        cfg = recursive_load_config("config/train_marigold.yaml")
        self.assertEqual(cfg.max_iter * cfg.dataloader.effective_batch_size, 966000)
        self.assertEqual(cfg.dataloader.effective_batch_size // cfg.dataloader.max_train_batch_size, 6)
        self.assertEqual(cfg.trainer.training_noise_scheduler.pretrained_path, cfg.model.pretrained_path)
        self.assertEqual(cfg.diffusion_schedule.prediction_type, "v_prediction")
        self.assertTrue(cfg.diffusion_schedule.rescale_betas_zero_snr)
        self.assertEqual(cfg.diffusion_schedule.timestep_spacing, "trailing")
        self.assertTrue(cfg.diffusion_schedule.terminal_noise_alignment)
        self.assertEqual(cfg.diffusion_schedule.terminal_noise_fade_fraction, .1)
        self.assertTrue(cfg.min_snr_enabled)
        self.assertEqual(cfg.offset_noise_strength, .1)
        self.assertEqual(cfg.latent_gradient_weight, .1)
        self.assertTrue(cfg.da2_conditioning_enabled)
        self.assertEqual(cfg.validity_guided_completion.smooth_weight, .005)
        self.assertEqual(cfg.validity_guided_completion.far_anchor_weight, .01)
        self.assertEqual(cfg.validity_guided_completion.far_depth_threshold, .5)
        self.assertNotIn("mode", cfg.validity_guided_completion)
        old = IterExponential(25000, .01, 100)
        new = IterExponential(cfg.lr_scheduler.kwargs.total_iter, .01, cfg.lr_scheduler.kwargs.warmup_steps)
        for step in (1000, 10000, 23000):
            self.assertAlmostEqual(old(step), new(step))

    def test_ablation_configs_change_only_declared_components(self):
        config_dir = Path("config/ablation")
        expected_files = {
            "00_full.yaml",
            "01_no_far_anchor.yaml",
            "02_no_vgc.yaml",
            "03_no_terminal_snr.yaml",
            "04_no_min_snr.yaml",
            "05_no_multi_res_noise.yaml",
            "06_no_offset_noise.yaml",
            "07_no_latent_gradient.yaml",
            "08_no_da2_prior.yaml",
        }
        self.assertEqual(
            {path.name for path in config_dir.glob("*.yaml")}, expected_files
        )

        configs = {
            path.stem: recursive_load_config(str(path))
            for path in config_dir.glob("*.yaml")
        }
        for cfg in configs.values():
            self.assertEqual(cfg.max_iter, 23000)
            self.assertEqual(cfg.dataloader.effective_batch_size, 42)
            self.assertEqual(cfg.dataloader.seed, 2024)
            self.assertEqual(cfg.diffusion_schedule.prediction_type, "v_prediction")

        self.assertEqual(
            configs["01_no_far_anchor"].validity_guided_completion.far_anchor_weight,
            0,
        )
        self.assertFalse(configs["02_no_vgc"].validity_guided_completion.enabled)
        terminal = configs["03_no_terminal_snr"].diffusion_schedule
        self.assertFalse(terminal.rescale_betas_zero_snr)
        self.assertEqual(terminal.timestep_spacing, "leading")
        self.assertEqual(terminal.min_snr_floor, 0)
        self.assertFalse(terminal.terminal_noise_alignment)
        self.assertFalse(configs["04_no_min_snr"].min_snr_enabled)
        self.assertIsNone(configs["05_no_multi_res_noise"].multi_res_noise)
        self.assertEqual(configs["06_no_offset_noise"].offset_noise_strength, 0)
        self.assertEqual(configs["07_no_latent_gradient"].latent_gradient_weight, 0)
        no_da2 = configs["08_no_da2_prior"]
        self.assertFalse(no_da2.da2_conditioning_enabled)
        self.assertFalse(no_da2.validity_guided_completion.enabled)

    def test_real_trainer_updates_unet_with_accumulation(self):
        self._run_trainer()

    def _run_trainer(self):
        # External DA2 code/weights are not in this checkout. Stub only its import;
        # execute the actual trainer loop with small differentiable modules.
        fake_da2 = ModuleType("DA2.depth_anything_v2.dpt")
        fake_da2.DepthAnythingV2 = nn.Module
        module_name = "DA2.depth_anything_v2.dpt"
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = fake_da2
        try:
            trainer_module = importlib.import_module("src.trainer.marigold_trainer")
        finally:
            # Restore only our stub; unloading unrelated imports can register
            # torchvision operators twice in the full test suite.
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous

        class TinyUnet(nn.Module):
            config = {"in_channels": 12}
            def __init__(self):
                super().__init__()
                self.conv_in = nn.Conv2d(12, 4, 1)
            def enable_xformers_memory_efficient_attention(self):
                pass
            def forward(self, x, t, text):
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
                self.scheduler = DDPMScheduler(num_train_timesteps=100, prediction_type="v_prediction")
            def encode_empty_text(self):
                self.empty_text_embed = torch.zeros(1, 1, 1)
            def encode_rgb(self, x):
                return self.vae(x)

        torch.manual_seed(7)
        cfg = recursive_load_config("config/train_marigold.yaml")
        cfg.max_iter = 1
        cfg.max_epoch = 1
        cfg.lr_scheduler.kwargs.warmup_steps = 0
        cfg.trainer.save_period = cfg.trainer.backup_period = 0
        pipeline = Pipeline()
        valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        # Rows 0:8 are fully invalid at latent scale, rows 8:16 are mixed, and
        # the remainder is fully valid. VGC must not treat the mixed row as sky.
        valid[..., :12, :] = False
        batch = {"rgb_norm": torch.randn(1, 3, 32, 32),
                 "depth_raw_norm": torch.randn(1, 1, 32, 32), "valid_mask_raw": valid}
        before = pipeline.unet.conv_in.weight.detach().clone()
        vae_before = pipeline.vae.weight.detach().clone()
        with patch.object(trainer_module.DDPMScheduler, "from_pretrained", return_value=pipeline.scheduler), \
                patch.object(trainer_module, "tb_logger", MagicMock()):
            loader = torch.utils.data.DataLoader(
                [{key: value[0] for key, value in batch.items()}] * 2, batch_size=1
            )
            trainer = trainer_module.MarigoldTrainer(cfg, pipeline, loader, "cpu", ".", ".", ".", ".", 2)
            self.assertEqual(trainer.prediction_type, "v_prediction")
            self.assertEqual(trainer.training_noise_scheduler.alphas_cumprod[-1].item(), 0)
            self.assertEqual(trainer.model.scheduler.config.timestep_spacing, "trailing")
            pred = torch.zeros(1, 4, 4, 4)
            prior = torch.ones_like(pred)
            invalid = torch.ones_like(pred, dtype=torch.bool)
            no_far = torch.zeros_like(invalid)
            some_far = no_far.clone()
            some_far[..., 0, :] = True
            loss_without_far = trainer._validity_guided_completion_loss(
                pred, prior, invalid, no_far
            )
            loss_with_far = trainer._validity_guided_completion_loss(
                pred, prior, invalid, some_far
            )
            self.assertGreater(loss_with_far.item(), loss_without_far.item())
            captured_invalid_masks = []
            captured_far_masks = []
            original_vgc = trainer._validity_guided_completion_loss

            def capture_vgc_mask(*args, **kwargs):
                captured_invalid_masks.append(kwargs["invalid_mask"].detach().clone())
                captured_far_masks.append(kwargs["far_mask"].detach().clone())
                return original_vgc(*args, **kwargs)

            trainer._validity_guided_completion_loss = capture_vgc_mask
            trainer.save_checkpoint = MagicMock()
            trainer.train()
        self.assertEqual(trainer.effective_iter, 1)
        self.assertEqual(trainer.n_batch_in_epoch, 2)
        self.assertFalse(torch.equal(before, pipeline.unet.conv_in.weight))
        torch.testing.assert_close(vae_before, pipeline.vae.weight)
        self.assertIsNone(pipeline.vae.weight.grad)
        self.assertTrue(torch.isfinite(pipeline.unet.conv_in.weight).all())
        self.assertTrue(captured_invalid_masks)
        self.assertTrue(captured_invalid_masks[0][..., 0, :].all())
        self.assertFalse(captured_invalid_masks[0][..., 1:, :].any())
        self.assertTrue(captured_far_masks)
        self.assertFalse(
            (captured_far_masks[0] & ~captured_invalid_masks[0]).any()
        )


if __name__ == "__main__":
    unittest.main()
