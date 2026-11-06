"""Preview cadence, reproducibility and isolation from the training loop."""

from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
import pytest
import torch

from src.util.training_preview import TrainingPreview


class PreviewPipeline:
    def __init__(self, fail=False):
        self.unet = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
        self.unet[1].eval()  # Check restoration of mixed submodule modes, too.
        self.vae = torch.nn.Linear(1, 1).eval()
        self.da2 = torch.nn.Linear(1, 1).eval()
        self.scheduler = SimpleNamespace(timesteps=[999])
        self.calls = []
        self.fail = fail

    def __call__(self, image, **kwargs):
        assert not torch.is_grad_enabled()
        assert not self.unet.training
        assert not self.vae.training
        assert not self.da2.training
        self.scheduler.timesteps = [3, 2, 1]
        # Even an inference implementation that uses global RNG must be isolated.
        random.random()
        np.random.rand()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("inference failed")
        noise = torch.rand(1, generator=kwargs["generator"]).item()
        self.calls.append((image.size, noise, kwargs))
        return SimpleNamespace(depth_colored=Image.new("RGB", image.size, (int(noise * 255), 0, 0)))


def image_folder(tmp_path):
    folder = tmp_path / "images"
    (folder / "nested").mkdir(parents=True)
    Image.new("RGB", (32, 48)).save(folder / "room.png")
    Image.new("RGB", (48, 32)).save(folder / "nested" / "room.JPG")
    (folder / "ignored.txt").write_text("not an image")
    return folder


@pytest.mark.parametrize("fail", [False, True])
def test_rng_scheduler_and_modes_are_restored(tmp_path, fail):
    folder = image_folder(tmp_path)
    preview = TrainingPreview({"input_dir": str(folder), "period": 2}, tmp_path / "out")
    model = PreviewPipeline(fail)
    original_scheduler = model.scheduler
    modes = [m.training for m in model.unet.modules()]
    random_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            preview.maybe_run(model, 2, "cpu")
    else:
        preview.maybe_run(model, 2, "cpu")
    assert model.scheduler is original_scheduler
    assert model.scheduler.timesteps == [999]
    assert [m.training for m in model.unet.modules()] == modes
    assert random.getstate() == random_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_periodic_images_and_fixed_seeds(tmp_path):
    folder = image_folder(tmp_path)
    preview = TrainingPreview({"input_dir": str(folder), "period": 2, "denoising_steps": 5}, tmp_path / "out")
    model = PreviewPipeline()
    for step in range(5):
        preview.maybe_run(model, step, "cpu")
    assert len(model.calls) == 4
    assert [call[1] for call in model.calls[:2]] == [call[1] for call in model.calls[2:]]
    assert model.calls[0][2]["denoising_steps"] == 5
    for step in (2, 4):
        output = tmp_path / "out/test_files" / f"iter_{step:06d}"
        assert (output / "room.png.depth.png").is_file()
        with Image.open(output / "nested/room.JPG.depth.png") as image:
            assert image.mode == "RGB" and image.size == (48, 32)
    assert not (tmp_path / "out/test_files/iter_000001").exists()


def test_disabled_and_bad_folders(tmp_path):
    model = PreviewPipeline()
    TrainingPreview({}, tmp_path).maybe_run(model, 2000, "cpu")
    TrainingPreview({"input_dir": "missing", "period": 0}, tmp_path).maybe_run(model, 2000, "cpu")
    assert not model.calls
    with pytest.raises(ValueError, match="does not exist"):
        TrainingPreview({"input_dir": str(tmp_path / "missing")}, tmp_path)
    with pytest.raises(ValueError, match="No supported"):
        TrainingPreview({"input_dir": str(tmp_path)}, tmp_path)


def test_real_sdxl_training_continues_after_preview_and_resume(tmp_path):
    from diffusers import DDIMScheduler
    from test_sdxl_depth import tiny_pipeline
    from src.trainer import marigold_trainer as trainer_module
    from src.util.config_util import recursive_load_config

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = recursive_load_config("config/train_sdxl_demo.yaml")
        # Explicit valid fixture: the user's working config may be an experiment.
        cfg.diffusion_schedule.prediction_type = "v_prediction"
        cfg.diffusion_schedule.rescale_betas_zero_snr = True
        cfg.diffusion_schedule.min_snr_floor = .05
        cfg.optimizer.name = "Adam"
        cfg.trainer.mixed_precision = "no"
        cfg.trainer.report_cuda_memory = False
        cfg.trainer.save_period = 0
        cfg.trainer.backup_period = 0
        cfg.max_iter = 2
        cfg.max_epoch = 2
        cfg.lr_scheduler.kwargs.warmup_steps = 0
        cfg.test_inference = dict(input_dir=str(image_folder(tmp_path)), period=1,
                                  denoising_steps=2, processing_res=32, seed=2024)
        sample = {"rgb_norm": torch.randn(3, 32, 32),
                  "depth_raw_norm": torch.randn(1, 32, 32),
                  "valid_mask_raw": torch.ones(1, 32, 32, dtype=torch.bool)}
        loader = torch.utils.data.DataLoader([sample] * 4, batch_size=1)
        pipe = tiny_pipeline(tmp_path)
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        with patch.object(trainer_module.DDPMScheduler, "from_pretrained", return_value=scheduler), \
                patch.object(trainer_module, "tb_logger", MagicMock()):
            trainer = trainer_module.MarigoldTrainer(
                cfg, pipe, loader, "cpu", ".", str(tmp_path / "ckpt"),
                str(tmp_path / "eval"), str(tmp_path / "vis"), 2)
            weights_before = pipe.unet.conv_out.weight.detach().clone()
            trainer.train()
            assert trainer.effective_iter == 2
            assert trainer.n_batch_in_epoch == 4  # Two microbatches per update.
            assert not torch.equal(weights_before, pipe.unet.conv_out.weight)
            assert pipe.unet.training
            assert len(list((tmp_path / "vis/test_files").rglob("*.png"))) == 4
            trainer.save_checkpoint("latest", save_train_state=True)
            resumed_pipe = tiny_pipeline(tmp_path)
            resumed = trainer_module.MarigoldTrainer(
                cfg, resumed_pipe, loader, "cpu", ".", str(tmp_path / "ckpt"),
                str(tmp_path / "eval"), str(tmp_path / "vis"), 2)
            resumed.load_checkpoint(tmp_path / "ckpt/latest")
            resumed.max_iter = 3
            resumed.train()
            assert resumed.effective_iter == 3
            assert (tmp_path / "vis/test_files/iter_000003/room.png.depth.png").is_file()
    finally:
        torch.set_num_threads(previous_threads)
