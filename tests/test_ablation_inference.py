import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import torch
from diffusers import DDIMScheduler

from src.util.model_overrides import (
    apply_model_overrides, checkpoint_inference_settings, checkpoint_pipeline_kwargs,
)


class AblationInferenceTest(unittest.TestCase):
    def make_checkpoint(self, directory, conditioning=True, zero_snr=False, spacing="leading"):
        checkpoint = Path(directory) / "checkpoint" / "iter_023000"
        checkpoint.mkdir(parents=True)
        settings = {"da2_conditioning_enabled": conditioning, "da2_config": None,
                    "diffusion_schedule": {"prediction_type": "v_prediction",
                                           "rescale_betas_zero_snr": zero_snr,
                                           "timestep_spacing": spacing}}
        (checkpoint / "inference_settings.json").write_text(json.dumps(settings), encoding="utf-8")
        DDIMScheduler(prediction_type="v_prediction", rescale_betas_zero_snr=zero_snr,
                      timestep_spacing=spacing).save_pretrained(checkpoint / "scheduler")
        return checkpoint

    def test_terminal_ablation_loads_its_own_schedule(self):
        for zero_snr, spacing in ((False, "leading"), (True, "trailing")):
            with self.subTest(zero_snr=zero_snr), tempfile.TemporaryDirectory() as directory:
                checkpoint = self.make_checkpoint(directory, zero_snr=zero_snr, spacing=spacing)
                loaded_unet = MagicMock(config=OmegaConf.create({"in_channels": 12, "addition_embed_type": None}))
                pipe = SimpleNamespace(unet=SimpleNamespace(dtype=torch.float32, config={"addition_embed_type": None}),
                                       vae=MagicMock(), da2=MagicMock(), register_to_config=MagicMock())
                with patch("diffusers.UNet2DConditionModel.from_pretrained", return_value=loaded_unet):
                    apply_model_overrides(pipe, checkpoint)
                self.assertEqual(pipe.scheduler.config.rescale_betas_zero_snr, zero_snr)
                self.assertEqual(pipe.scheduler.config.timestep_spacing, spacing)

    def test_no_da2_switch_restored_before_and_after_pipeline_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory, conditioning=False)
            self.assertFalse(checkpoint_pipeline_kwargs(checkpoint)["da2_conditioning_enabled"])
            pipe = SimpleNamespace(unet=SimpleNamespace(dtype=torch.float32, config={"addition_embed_type": None}),
                                   vae=MagicMock(), da2=MagicMock(), register_to_config=MagicMock())
            unet = MagicMock(config=OmegaConf.create({"in_channels": 12, "addition_embed_type": None}))
            with patch("diffusers.UNet2DConditionModel.from_pretrained", return_value=unet):
                apply_model_overrides(pipe, checkpoint)
            self.assertFalse(pipe.da2_conditioning_enabled)

    def test_legacy_run_config_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory)
            (checkpoint / "inference_settings.json").unlink()
            (Path(directory) / "config.yaml").write_text(
                "da2_conditioning_enabled: false\n", encoding="utf-8")
            self.assertFalse(checkpoint_inference_settings(checkpoint)["da2_conditioning_enabled"])

    def test_missing_settings_or_scheduler_and_mismatched_schedule_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                checkpoint_inference_settings(Path(directory) / "no_settings")
            checkpoint = self.make_checkpoint(directory)
            settings = checkpoint_inference_settings(checkpoint)
            settings["diffusion_schedule"]["timestep_spacing"] = "trailing"
            (checkpoint / "inference_settings.json").write_text(json.dumps(settings), encoding="utf-8")
            pipe = SimpleNamespace(unet=SimpleNamespace(dtype=torch.float32, config={"addition_embed_type": None}))
            unet = MagicMock(config=OmegaConf.create({"in_channels": 12, "addition_embed_type": None}))
            with patch("diffusers.UNet2DConditionModel.from_pretrained", return_value=unet):
                with self.assertRaisesRegex(ValueError, "timestep_spacing"):
                    apply_model_overrides(pipe, checkpoint)


if __name__ == "__main__":
    unittest.main()
