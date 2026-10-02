# Last modified: 2026-06-16
#
# Copyright 2026 Jiawei Wang, SJZU. All rights reserved.
#
# This file has been modified from the original version.
# Original copyright (c) 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
import shutil
import json
from contextlib import nullcontext
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from marigold.marigold_pipeline import MarigoldPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.loss import LatentGradLoss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence
from src.util.residual_completion import (
    latent_validity_masks, min_snr_weights, residual_completion_loss,
)


class MarigoldTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Adapt input layers. The method uses fixed 12-channel conditioning:
        # RGB latent (4) + DA2 prior latent (4) + noisy depth latent (4).
        if 12 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        if cfg.trainer.get("use_xformers", True):
            self.model.unet.enable_xformers_memory_efficient_attention()
        if cfg.trainer.get("gradient_checkpointing", False):
            self.model.unet.enable_gradient_checkpointing()
        self.use_bf16 = cfg.trainer.get("mixed_precision", "no") == "bf16"
        if cfg.trainer.get("mixed_precision", "no") not in ("no", "bf16"):
            raise ValueError("Training supports FP32 or BF16 autocast with FP32 weights")
        if self.use_bf16 and (torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 demo training requires a BF16-capable CUDA GPU")
        self.memory_probe = bool(cfg.trainer.get("memory_probe", False))
        self.report_cuda_memory = bool(cfg.trainer.get("report_cuda_memory", False))

        # Trainability
        self.model.vae.requires_grad_(False)
        if self.model.text_encoder is not None:
            self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)
        self.model.da2.requires_grad_(False)
        self.model.da2.eval()

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        if cfg.optimizer.name == "Adam8bit":
            if torch.device(device).type != "cuda":
                raise ValueError("Adam8bit demo requires CUDA")
            try:
                from bitsandbytes.optim import Adam8bit
            except ImportError as exc:
                raise ImportError("SDXL demo requires bitsandbytes: pip install bitsandbytes") from exc
            self.optimizer = Adam8bit(self.model.unet.parameters(), lr=lr)
        elif cfg.optimizer.name == "Adam":
            self.optimizer = Adam(self.model.unet.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)
        self.latent_grad_loss = LatentGradLoss()

        # Validity-Guided Completion (VGC)
        # This is a mask-free sky-collapse regularizer: it does not use sky masks
        # or extra input channels. It only acts on regions without valid depth
        # supervision, which are common in outdoor sky / out-of-range areas.
        vgc_cfg = self.cfg.get("validity_guided_completion", {})
        self.vgc_enabled = bool(vgc_cfg.get("enabled", False))
        self.vgc_min_invalid_ratio = float(vgc_cfg.get("min_invalid_ratio", 0.08))
        self.vgc_anchor_weight = float(vgc_cfg.get("anchor_weight", 0.02))
        self.vgc_smooth_weight = float(vgc_cfg.get("smooth_weight", 0.005))
        self.vgc_eps = float(vgc_cfg.get("eps", 1e-6))
        self.vgc_mode = vgc_cfg.get("mode", "legacy")
        if self.vgc_mode not in ("legacy", "residual_snr"):
            raise ValueError(f"Unknown completion mode: {self.vgc_mode}")
        self.vgc_gradient_weight = float(vgc_cfg.get("gradient_weight", 0.005))
        self.min_snr_gamma = float(self.cfg.get("min_snr_gamma", 5.0))
        if self.min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be positive")

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(
            "loss", "diffusion_loss", "latent_gradient_loss", "completion_loss",
            "completion_anchor", "completion_gradient", "completion_coverage",
            "completion_snr_weight",
        )
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    # 12 channels
    def _replace_unet_conv_in(self):
        # replace the first layer to accept 12 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()      # [320]
        _weight = _weight.repeat((1, 3, 1, 1))  

        _weight *= (1.0 / 3.0)
        
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        
        logging.info("Unet conv_in layer is replaced to accept 12 channels")
        
        # replace config
        self.model.unet.config["in_channels"] = 12
        logging.info("Unet config is updated to 12 channels")
        return

    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)
        if self.report_cuda_memory and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb = batch["rgb_norm"].to(device)
                depth_gt_for_latent = batch[self.gt_depth_type].to(device)
                with torch.no_grad():
                    da2_depth = self.model.da2.infer_batch(rgb).to(device)

                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    valid_mask_down, invalid_interior_down = latent_validity_masks(
                        valid_mask_for_latent
                    )
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                    invalid_mask_down = ~valid_mask_down
                else:
                    raise NotImplementedError

                batch_size = rgb.shape[0]

                with torch.no_grad():
                    # Encode image
                    rgb_latent = self.model.encode_rgb(rgb)  # [B, 4, h, w]
                    # Encode GT depth
                    gt_depth_latent = self.encode_depth(
                        depth_gt_for_latent
                    )  # [B, 4, h, w]
                    # Encode DA2 depth
                    da2_depth_latent = self.model.encode_rgb(da2_depth)  # [B, 4, h, w]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        gt_depth_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        gt_depth_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]
                
                offset_noise_strength = 0.1
                offset_noise = torch.randn(
                    batch_size, gt_depth_latent.shape[1], 1, 1, 
                    device=device, 
                    generator=rand_num_generator
                ) * offset_noise_strength
                noise = noise + offset_noise

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    gt_depth_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and depth latents
                cat_latents = torch.cat(
                    [rgb_latent, da2_depth_latent, noisy_latents], dim=1
                )  # [B, 12, h, w]
                cat_latents = cat_latents.float()

                # Predict the noise residual
                amp_context = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
                with amp_context:
                    if hasattr(self.model, "predict_noise"):
                        model_pred = self.model.predict_noise(cat_latents, timesteps, text_embed)
                    else:
                        model_pred = self.model.unet(cat_latents, timesteps, text_embed).sample
                model_pred = model_pred.float()
                
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = gt_depth_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        gt_depth_latent, noise, timesteps
                    )  # [B, 4, h, w]
                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")
                
                alphas_cumprod = self.training_noise_scheduler.alphas_cumprod.to(device)
                alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                beta_prod_t = 1 - alpha_prod_t

                if "v_prediction" == self.prediction_type:
                    pred_x0 = (alpha_prod_t ** 0.5) * noisy_latents - (beta_prod_t ** 0.5) * model_pred
                elif "epsilon" == self.prediction_type:
                    pred_x0 = (noisy_latents - (beta_prod_t ** 0.5) * model_pred) / (alpha_prod_t ** 0.5)
                else:
                    pred_x0 = model_pred # sample type

            
                snr_weight, completion_snr_weight = min_snr_weights(
                    alpha_prod_t, self.prediction_type, self.min_snr_gamma
                )

                diff = model_pred.float() - target.float()
                if "l1" in self.cfg.loss.name.lower():
                    unreduced_loss = torch.abs(diff)
                else:
                    unreduced_loss = torch.square(diff)
                
                unreduced_loss = unreduced_loss * snr_weight

                if self.gt_mask_type is not None:
                    latent_loss = torch.where(valid_mask_down, unreduced_loss, 0.0).sum() / valid_mask_down.sum().clamp_min(1)
                    grad_loss = self.latent_grad_loss(
                        pred_x0.float(), 
                        gt_depth_latent.float(), 
                        valid_mask_down
                    )
                else:
                    latent_loss = unreduced_loss
                    grad_loss = self.latent_grad_loss(pred_x0.float(), gt_depth_latent.float())

                loss = latent_loss.mean() + 0.1 * grad_loss.mean()
                vgc_loss = loss.new_zeros(())
                completion_stats = {}

                if self.vgc_enabled and self.vgc_mode == "residual_snr":
                    vgc_loss, completion_stats = residual_completion_loss(
                        pred_x0, da2_depth_latent, invalid_interior_down,
                        completion_snr_weight,
                        anchor_weight=self.vgc_anchor_weight,
                        gradient_weight=self.vgc_gradient_weight,
                        min_invalid_ratio=self.vgc_min_invalid_ratio,
                    )
                    loss = loss + vgc_loss
                elif self.vgc_enabled:
                    # VGC only supervises invalid-depth regions. This prevents the
                    # regularizer from directly supervising valid pixels. Shared
                    # U-Net parameters still require indoor regression evaluation.
                    vgc_loss = self._validity_guided_completion_loss(
                        pred_x0=pred_x0.float(),
                        prior_latent=da2_depth_latent.float().detach(),
                        invalid_mask=invalid_mask_down,
                    )
                    loss = loss + vgc_loss

                self.train_metrics.update("loss", loss.item())
                if self.cfg.trainer.get("check_finite_loss", False) and not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; stop before corrupting the optimizer")
                self.train_metrics.update("diffusion_loss", latent_loss.mean().item())
                self.train_metrics.update("latent_gradient_loss", grad_loss.mean().item())
                self.train_metrics.update("completion_loss", vgc_loss.item())
                for name, value in completion_stats.items():
                    self.train_metrics.update(name, value.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1
                    if self.report_cuda_memory and torch.device(device).type == "cuda":
                        self._report_memory()

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        if not self.memory_probe:
                            self.save_checkpoint(
                                ckpt_name=self._get_backup_ckpt_name(),
                                save_train_state=False,
                            )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def _report_memory(self):
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        gib = 1024 ** 3
        record = {
            "effective_iter": self.effective_iter,
            "gpu": torch.cuda.get_device_name(self.device),
            "allocated_gib": torch.cuda.memory_allocated(self.device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / gib,
            "device_free_gib": free / gib, "device_total_gib": total / gib,
            "backbone": self.cfg.model.get("backbone", "sd2"),
            "optimizer": self.cfg.optimizer.name,
            "unet_parameters": sum(p.numel() for p in self.model.unet.parameters()),
            "mixed_precision": self.cfg.trainer.get("mixed_precision", "no"),
            "gradient_checkpointing": self.cfg.trainer.get("gradient_checkpointing", False),
            "microbatch": self.cfg.dataloader.max_train_batch_size,
            "accumulation_steps": self.accumulation_steps,
        }
        logging.info("CUDA memory: %s", json.dumps(record))
        path = os.path.join(self.out_dir_ckpt, "..", "memory_profile.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

    def _validity_guided_completion_loss(self, pred_x0, prior_latent, invalid_mask):
        """Weakly complete invalid-depth regions using the detached DA2 prior.

        Outdoor sky and out-of-range regions are often excluded by the valid-depth
        mask, so the normal supervised loss gives them no training signal. VGC
        adds a small training-only loss on those invalid regions, without using a
        sky segmentation mask and without changing the 12-channel input.
        """
        if invalid_mask is None:
            return pred_x0.new_tensor(0.0)

        invalid_mask = invalid_mask.to(device=pred_x0.device, dtype=pred_x0.dtype)
        if invalid_mask.shape != pred_x0.shape:
            invalid_mask = invalid_mask.expand_as(pred_x0)

        # Avoid applying the loss to tiny missing-depth holes. This is important
        # for indoor datasets such as NYU, where most pixels are valid and small
        # invalid holes should not dominate training.
        invalid_ratio = invalid_mask.flatten(1).mean(dim=1)  # [B]
        sample_gate = (invalid_ratio >= self.vgc_min_invalid_ratio).to(pred_x0.dtype)
        if sample_gate.sum() <= 0:
            return pred_x0.new_tensor(0.0)

        sample_gate = sample_gate.view(-1, 1, 1, 1)
        mask = invalid_mask * sample_gate
        denom = mask.sum().clamp_min(self.vgc_eps)

        # Region-level anchor: align only the mean latent response in invalid
        # regions to the prior. This prevents copying high-frequency DA2 artifacts
        # into sky while still preventing free collapse.
        pred_region_mean = (pred_x0 * mask).sum(dim=(2, 3), keepdim=True) / (
            mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.vgc_eps)
        )
        prior_region_mean = (prior_latent * mask).sum(dim=(2, 3), keepdim=True) / (
            mask.sum(dim=(2, 3), keepdim=True).clamp_min(self.vgc_eps)
        )
        anchor_loss = ((pred_region_mean - prior_region_mean) ** 2 * sample_gate).sum() / (
            sample_gate.sum().clamp_min(self.vgc_eps) * pred_x0.shape[1]
        )

        # Smooth invalid regions only. This suppresses sky-like texture collapse but
        # does not smooth valid object boundaries.
        grad_x = torch.abs(pred_x0[..., 1:] - pred_x0[..., :-1])
        grad_y = torch.abs(pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :])
        mask_x = mask[..., 1:] * mask[..., :-1]
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        smooth_x = (grad_x * mask_x).sum() / mask_x.sum().clamp_min(self.vgc_eps)
        smooth_y = (grad_y * mask_y).sum() / mask_y.sum().clamp_min(self.vgc_eps)
        smooth_loss = 0.5 * (smooth_x + smooth_y)

        return self.vgc_anchor_weight * anchor_loss + self.vgc_smooth_weight * smooth_loss

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]
            # GT depth
            depth_raw_ts = batch["depth_raw_linear"].squeeze()
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["rgb_relative_path"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location="cpu", weights_only=True)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"), map_location="cpu", weights_only=False)
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
