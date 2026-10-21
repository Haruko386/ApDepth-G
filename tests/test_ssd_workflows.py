"""SSD architecture, download plans and preserved main sky/depth evaluation."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from accelerate import init_empty_weights
from diffusers import UNet2DConditionModel

from script.download_ssd_weights import SSD_FILES, SSD_REVISION, download_ssd, main as download_main
from script.eval_ssd import commands
from marigold.marigold_pipeline import MarigoldPipeline
from src.trainer.marigold_trainer import MarigoldTrainer
from src.util.config_util import recursive_load_config
from src.util.model_overrides import apply_model_overrides
from src.util.diode_evaluation import refine_diode_valid_mask
from test_ssd_depth import tiny_pipeline, DummyPrior


def test_original_ssd_topology_builds_without_allocating_weights():
    # Fixture copied verbatim from segmind/SSD-1B at SSD_REVISION.
    config = json.loads(Path("tests/fixtures/ssd_unet_config.json").read_text())
    with init_empty_weights():
        unet = UNet2DConditionModel.from_config(config)
    assert unet.config.mid_block_type == "UNetMidBlock2D"
    assert len(unet.up_blocks[0].attentions[2].transformer_blocks) == 10
    assert len(unet.up_blocks[1].attentions[1].transformer_blocks) == 1
    assert unet.add_embedding.linear_1.in_features == 2816
    assert 1_000_000_000 < sum(p.numel() for p in unet.parameters()) < 1_500_000_000


def test_download_uses_exact_diffusers_files_and_detects_incomplete_snapshot(tmp_path):
    with patch("huggingface_hub.snapshot_download") as download:
        with pytest.raises(FileNotFoundError, match="Incomplete SSD"):
            download_ssd(tmp_path)
        assert download.call_args.kwargs["revision"] == SSD_REVISION
        assert download.call_args.kwargs["allow_patterns"] == SSD_FILES
    assert all("fp16" not in name for name in SSD_FILES)
    assert not any(name.endswith(".safetensors") and "/" not in name for name in SSD_FILES)
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("network")):
        assert download_main(["--base_ckpt_dir", str(tmp_path / "new"), "--dry-run"]) == 0
    assert not (tmp_path / "new").exists()


def test_eval_plan_keeps_sky_separate_and_preserves_main_benchmark_options(tmp_path):
    args = argparse.Namespace(datasets=None, sky_mask_name="mask2former_v1", eval_only=False,
                              unet_checkpoint="trained", checkpoint="base", base_data_dir="data",
                              output_dir=str(tmp_path), denoise_steps=50, ensemble_size=1,
                              processing_res=768, seed=2024, da2_config=None, fp16=False)
    sky = list(commands(args))
    assert len(sky) == 6
    assert all("--sky_mask_name" in command for command in sky)
    assert all("--alignment" not in command for command in sky)
    assert all(command[command.index("--backbone") + 1] == "ssd" for command in sky[::2])
    args.sky_mask_name = None
    depth = list(commands(args))
    assert len(depth) == 10
    assert all("--alignment" in command for command in depth[1::2])
    eth3d = next(command for command in depth[1::2] if "data_eth3d.yaml" in " ".join(command))
    assert eth3d[-2:] == ["--alignment_max_res", "1024"]
    args.eval_only = True
    args.unet_checkpoint = None
    assert len(list(commands(args))) == 5


@pytest.mark.parametrize("vgc", [False, True])
def test_eight_channel_ablations_train_and_restore_without_da2_input(tmp_path, vgc):
    from src.trainer import marigold_trainer as trainer_module
    from unittest.mock import MagicMock
    cfg = recursive_load_config(f"config/ablation_ssd/{'12_8ch_vgc' if vgc else '11_8ch_no_vgc'}.yaml")
    cfg.optimizer.name = "Adam"
    cfg.trainer.report_cuda_memory = False
    cfg.lr_scheduler.kwargs.warmup_steps = 0
    cfg.max_iter = cfg.max_epoch = 1
    cfg.trainer.save_period = cfg.trainer.backup_period = 0
    pipe = tiny_pipeline(tmp_path)
    old_input = pipe.unet.conv_in.weight.detach().clone()
    old_output = pipe.unet.conv_out.weight.detach().clone()
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    mask[:, :16] = False
    sample = {"rgb_norm": torch.randn(3, 32, 32), "depth_raw_norm": torch.randn(1, 32, 32),
              "valid_mask_raw": mask}
    loader = torch.utils.data.DataLoader([sample], batch_size=1)
    with patch.object(trainer_module.DDPMScheduler, "from_pretrained", return_value=pipe.scheduler), \
            patch.object(trainer_module, "tb_logger", MagicMock()):
        trainer = MarigoldTrainer(cfg, pipe, loader, "cpu", ".", str(tmp_path / "ckpt"), ".", ".", 1)
        assert pipe.unet.config.in_channels == 8
        torch.testing.assert_close(pipe.unet.conv_in.weight, old_input.repeat(1, 2, 1, 1) / 2)
        trainer.train()
    assert not torch.equal(old_output, pipe.unet.conv_out.weight)
    target = tiny_pipeline(tmp_path)
    target.depth_backbone = "ssd"
    apply_model_overrides(target, tmp_path / "ckpt/iter_000001")
    target.da2.infer_batch = MagicMock(side_effect=AssertionError("8ch inference ran DA2"))
    assert target.unet.config.in_channels == 8
    assert not target.da2_conditioning_enabled
    assert torch.isfinite(target.single_infer(torch.randn(1, 3, 32, 32), 2, generator=None, show_pbar=False)).all()


def test_ssd_rejects_sdxl_metadata_before_loading_weights(tmp_path):
    pipe = tiny_pipeline(tmp_path)
    pipe.depth_backbone = "ssd"
    (tmp_path / "inference_settings.json").write_text(json.dumps({
        "backbone": "sdxl", "da2_conditioning_enabled": True,
    }))
    with pytest.raises(ValueError, match="backbone mismatch"):
        apply_model_overrides(pipe, tmp_path)


def test_terminal_fade_ablation_remains_independent(tmp_path):
    from src.trainer import marigold_trainer as trainer_module
    cfg = recursive_load_config("config/ablation_ssd/N2_no_terminal_noise_fade.yaml")
    cfg.optimizer.name = "Adam"
    pipe = tiny_pipeline(tmp_path)
    with patch.object(trainer_module.DDPMScheduler, "from_pretrained", return_value=pipe.scheduler):
        trainer = MarigoldTrainer(cfg, pipe, [], "cpu", ".", ".", ".", ".", 1)
    assert trainer.terminal_noise_alignment
    assert not trainer.terminal_noise_fade_enabled


def test_diode_filter_preserves_gt_and_filters_discontinuities():
    depth = np.full((32, 32), 2.)
    depth[:, 16:] = 4.
    mask = np.ones_like(depth, dtype=bool)
    before = depth.copy()
    refined = refine_diode_valid_mask(depth, mask)
    assert not refined[:, 15:17].any()
    assert refined[:, :14].all()
    np.testing.assert_array_equal(depth, before)
    assert mask.all()


def test_full_frame_kitti_sky_inference_and_eval_cli(tmp_path):
    # A frame smaller than the benchmark crop exposes accidental crop changes.
    rgb = tmp_path / "rgb"
    rgb.mkdir()
    Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)).save(rgb / "00000.png")
    split = tmp_path / "split.txt"
    split.write_text("00000.png unused_depth.png\n00001.png unused_depth.png\n")
    config = tmp_path / "dataset.yaml"
    config.write_text(f"name: kitti\ndisp_name: tiny_kitti\ndir: rgb\nfilenames: {split.as_posix()}\n"
                      "kitti_bm_crop: true\nvalid_mask_crop: eigen\n")
    sky_root = rgb / "sky_masks/mask2former_v1"
    sky_root.mkdir(parents=True)
    mask = np.zeros((128, 256), dtype=np.uint8)
    mask[:64] = 255
    Image.fromarray(mask).save(sky_root / "00000.png.sky.png")
    pipe = tiny_pipeline(tmp_path)
    trainer = object.__new__(MarigoldTrainer)
    trainer.model = pipe
    trainer._replace_unet_conv_in()
    common = ["--dataset_config", str(config), "--base_data_dir", str(tmp_path),
              "--sky_mask_name", "mask2former_v1"]
    predictions = tmp_path / "predictions"
    args = ["infer.py", "--backbone", "ssd", "--checkpoint", "unused", *common,
            "--output_dir", str(predictions), "--processing_res", "32", "--denoise_steps", "2",
            "--ensemble_size", "1", "--seed", "2024"]
    with patch("src.util.pipeline_loader.load_depth_pipeline", return_value=pipe), \
            patch.object(torch.cuda, "is_available", return_value=False), patch.object(sys, "argv", args):
        runpy.run_path("infer.py", run_name="__main__")
    files = list(predictions.rglob("*.npy"))
    assert len(files) == 1
    assert np.load(files[0]).shape == mask.shape
    # Known prediction: sky is constant and farther than varying foreground.
    depth = np.tile(np.linspace(0, 1, 256), (128, 1))
    depth[:64] = 2
    np.save(files[0], depth)
    output = tmp_path / "metrics"
    args = ["eval.py", *common, "--prediction_dir", str(predictions), "--output_dir", str(output)]
    with patch.object(sys, "argv", args), pytest.raises(SystemExit) as result:
        runpy.run_path("eval.py", run_name="__main__")
    assert result.value.code == 0
    report = (output / "eval_metrics-sky.txt").read_text()
    assert "evaluated images: 1" in report
    for metric in ("sky_iqr", "sky_p95_p5", "sfoe"):
        row = next(line for line in report.splitlines() if line.startswith(metric))
        assert float(row.split()[1]) == 0


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
