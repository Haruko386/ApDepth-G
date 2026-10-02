import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from diffusers import AutoencoderKL, DDIMScheduler

from src.util.decoder_calibration import (
    calibration_loss, capture_terminal_prediction, decode_raw, eligible_checkpoint,
    read_labels, read_manifest, region_metrics, supervision_masks, train_decoder_only,
)
from src.util.model_overrides import apply_model_overrides
from train_decoder_calibration import fit


def tiny_vae():
    return AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
                         block_out_channels=(8, 16), norm_num_groups=4,
                         down_block_types=("DownEncoderBlock2D",) * 2,
                         up_block_types=("UpDecoderBlock2D",) * 2)


class DecoderLossTest(unittest.TestCase):
    def test_far_direction_and_no_clamp_dead_gradient(self):
        labels = torch.full((1, 1, 8, 8), 255, dtype=torch.uint8)
        pred = torch.full((1, 1, 8, 8), -3.0, requires_grad=True)
        loss, _ = calibration_loss(pred, torch.zeros_like(pred), labels, erosion=0)
        loss.backward()
        self.assertTrue((pred.grad < 0).all())  # Gradient descent increases depth.
        far, _ = calibration_loss(torch.ones_like(pred), torch.zeros_like(pred), labels, erosion=0)
        self.assertEqual(far.item(), 0)

    def test_unknown_pixels_ignored_and_non_sky_preserved(self):
        labels = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
        labels[..., :4, :] = 255
        labels[..., 4, :] = 128
        reference = torch.randn(1, 1, 8, 8)
        target = reference.clone()
        target[..., :4, :] = 1
        target[..., 4, :] = 77
        target.requires_grad_()
        loss, _ = calibration_loss(target, reference, labels, erosion=0)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue((target.grad[..., 4, :] == 0).all())
        changed = target.detach().clone()
        changed[..., 5:, :] += .2
        loss, _ = calibration_loss(changed, reference, labels, erosion=0)
        self.assertGreater(loss.item(), 0)

    def test_erosion_respects_objects_and_keeps_image_border(self):
        labels = torch.full((1, 1, 8, 8), 255, dtype=torch.uint8)
        labels[..., 4:, :] = 0
        sky, keep = supervision_masks(labels, erosion=1)
        self.assertTrue(sky[..., 0, :].all())
        self.assertFalse(sky[..., 3, :].any())
        self.assertTrue(keep[..., 4, :].all())

    def test_no_sky_replay_and_empty_regions_are_finite(self):
        for label in (0, 128, 255):
            pred = torch.randn(1, 1, 1, 1, requires_grad=True)
            labels = torch.full_like(pred, label, dtype=torch.uint8)
            loss, _ = calibration_loss(pred, torch.zeros_like(pred), labels, erosion=0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_selection_rejects_foreground_damage(self):
        baseline = {"sky_below_09": .9}
        candidate = {"sky_mae": .02, "sky_below_09": .1, "non_sky_mae": .01, "non_sky_p95": .03}
        self.assertTrue(eligible_checkpoint(candidate, baseline, .5))
        candidate["non_sky_p95"] = .1
        self.assertFalse(eligible_checkpoint(candidate, baseline, .5))
        candidate["non_sky_p95"] = .03
        candidate["sky_below_09"] = 1
        self.assertFalse(eligible_checkpoint(candidate, baseline, .5))

    def test_metric_direction_and_overshoot_visible(self):
        labels = torch.full((1, 1, 8, 8), 255, dtype=torch.uint8)
        metrics = region_metrics(torch.full((1, 1, 8, 8), -1.), torch.zeros(1, 1, 8, 8), labels)
        self.assertEqual(metrics["sky_mae"], 1)
        self.assertEqual(metrics["sky_below_09"], 1)
        metrics = region_metrics(torch.full((1, 1, 8, 8), 1.2), torch.zeros(1, 1, 8, 8), labels)
        self.assertAlmostEqual(metrics["sky_mae"], .1, places=5)


class ManifestTest(unittest.TestCase):
    def test_registered_masks_and_split_leak_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "a.png")
            Image.fromarray(np.ones((8, 8, 3), np.uint8)).save(root / "b.png")
            Image.fromarray(np.full((8, 8), 255, np.uint8)).save(root / "mask.png")
            labels = read_labels(root / "mask.png", (8, 8))
            self.assertEqual(labels.shape, (1, 1, 8, 8))
            with self.assertRaises(ValueError):
                read_labels(root / "mask.png", (9, 8))
            rows = [{"image": "a.png", "mask": "mask.png", "split": "train"},
                    {"image": "b.png", "mask": "mask.png", "split": "val"}]
            path = root / "manifest.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(len(read_manifest(path)), 2)
            rows[1]["image"] = "a.png"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_manifest(path)


class CacheCaptureTest(unittest.TestCase):
    def test_full_ddim_capture_and_hook_cleanup(self):
        class Pipe:
            def __init__(self):
                self.unet = nn.Conv2d(12, 4, 1)
                self.steps_seen = 0
                self.fail = False

            def decode_depth(self, latent):
                return latent.mean(1, keepdim=True)

            def __call__(self, image, **kwargs):
                scheduler = DDIMScheduler(num_train_timesteps=100, prediction_type="v_prediction")
                scheduler.set_timesteps(kwargs["denoising_steps"])
                latent = torch.randn(1, 4, 4, 4, generator=kwargs["generator"])
                condition = torch.zeros(1, 8, 4, 4)
                for timestep in scheduler.timesteps:
                    prediction = self.unet(torch.cat([condition, latent], 1))
                    latent = scheduler.step(prediction, timestep, latent).prev_sample
                    self.steps_seen += 1
                if self.fail:
                    raise RuntimeError("inference failed")
                return self.decode_depth(latent)

        pipe = Pipe()
        original = pipe.decode_depth
        latent, ref = capture_terminal_prediction(pipe, None, 5, 512, 2024, "cpu")
        self.assertEqual(pipe.steps_seen, 5)
        torch.testing.assert_close(ref, latent.mean(1, keepdim=True))
        self.assertEqual(pipe.decode_depth, original)
        self.assertFalse(pipe.unet._forward_hooks)
        pipe.fail = True
        with self.assertRaises(RuntimeError):
            capture_terminal_prediction(pipe, None, 5, 512, 2024, "cpu")
        self.assertEqual(pipe.decode_depth, original)
        self.assertFalse(pipe.unet._forward_hooks)


class DecoderTrainingTest(unittest.TestCase):
    def test_only_decoder_changes_and_inference_load_preserves_encoder(self):
        torch.manual_seed(9)
        source = tiny_vae()
        vae = copy.deepcopy(source)
        before = {k: v.clone() for k, v in vae.state_dict().items()}
        parameters = train_decoder_only(vae)
        optimizer = torch.optim.Adam(parameters, lr=.001)
        latent = torch.randn(1, 4, 4, 4)
        with torch.no_grad():
            reference = decode_raw(vae, latent, .18215)
        labels = torch.full_like(reference, 255, dtype=torch.uint8)
        loss, _ = calibration_loss(decode_raw(vae, latent, .18215), reference, labels, erosion=0)
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in vae.state_dict().items() if k.startswith("decoder.")))
        for k, value in vae.state_dict().items():
            if not k.startswith("decoder."):
                torch.testing.assert_close(value, before[k], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            vae.save_pretrained(Path(directory) / "vae")
            class Pipe:
                pass
            pipe = Pipe()
            pipe.vae, pipe.da2 = source, None
            pipe.unet = nn.Linear(1, 1)
            pipe.unet.config = type("Config", (), {"in_channels": 12})()
            apply_model_overrides(pipe, decoder_checkpoint=directory)
            torch.testing.assert_close(decode_raw(pipe.vae, latent, .18215), decode_raw(vae, latent, .18215))
            with torch.no_grad():
                pipe.vae.encoder.conv_in.weight.add_(1)
            with self.assertRaisesRegex(ValueError, "source VAE mismatch"):
                apply_model_overrides(pipe, decoder_checkpoint=directory)

    def test_real_fit_checkpoint_and_exact_resume(self):
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            vae = tiny_vae().eval()
            vae.save_pretrained(cache / "source_vae")
            records = []
            for i, split in enumerate(["train", "train", "val"]):
                latent = torch.randn(1, 4, 4, 4)
                with torch.no_grad():
                    reference = decode_raw(vae, latent, .18215)
                labels = torch.zeros_like(reference, dtype=torch.uint8)
                if i != 1:
                    labels[..., :4, :] = 255
                name = f"{i}.pt"
                torch.save({"latent": latent, "reference": reference, "labels": labels}, cache / name)
                records.append({"file": name, "identity": str(i), "split": split})
            metadata = {"version": 1, "steps": 50, "depth_latent_scale_factor": .18215, "records": records}
            (cache / "index.json").write_text(json.dumps(metadata), encoding="utf-8")
            cfg = {"max_steps": 2, "learning_rate": .0001, "warmup_steps": 0, "accumulation_steps": 2,
                   "seed": 7, "evaluate_every": 1, "save_every": 1, "max_grad_norm": 1., "erosion": 0,
                   "sky_weight": 1., "preserve_weight": 5., "gradient_weight": .5,
                   "max_non_sky_mae": .02, "max_non_sky_p95": .05}
            fit(cache, root / "run", cfg, "cpu")
            fit(cache, root / "resumed", cfg, "cpu", root / "run/step_000001")
            full = AutoencoderKL.from_pretrained(root / "run/step_000002/vae")
            resumed = AutoencoderKL.from_pretrained(root / "resumed/step_000002/vae")
            for key, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
                if not key.startswith("decoder."):
                    torch.testing.assert_close(value, vae.state_dict()[key], rtol=0, atol=0)
            self.assertTrue((root / "run/metrics.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
