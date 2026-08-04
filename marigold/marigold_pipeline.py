# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from DA2.depth_anything_v2.dpt import DepthAnythingV2

class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

        # Optional inference-time Prior-guided Far-field Rectification (PFR).
        # These attributes let external scripts or trainer validation enable PFR
        # without changing model weights or the 12-channel denoising path.
        self.pfr_enabled = False
        self.pfr_kwargs = {}

        da2_config = {
            'encoder':'vitg',
            'features': 384,
            'out_channels': [1536, 1536, 1536, 1536]
        }

        if da2_config is not None:
            self.da2 = DepthAnythingV2(**da2_config)
            self.da2.load_state_dict(torch.load(f'DA2/checkpoints/depth_anything_v2_{da2_config["encoder"]}.pth', map_location='cpu'))
            self.da2.to(device="cuda")
        else:
            self.da2 = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        pfr_enabled: Optional[bool] = None,
        pfr_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
            pfr_enabled (`bool`, *optional*, defaults to `None`):
                Enable prior-guided far-field rectification. `None` uses the
                value stored on the pipeline, which defaults to `False`.
            pfr_kwargs (`dict`, *optional*, defaults to `None`):
                Keyword arguments forwarded to prior-guided far-field
                rectification. `None` uses the settings stored on the pipeline.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        if pfr_enabled is None:
            pfr_enabled = getattr(self, "pfr_enabled", False)
        if pfr_kwargs is None:
            pfr_kwargs = getattr(self, "pfr_kwargs", {})

        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                pfr_enabled=pfr_enabled,
                pfr_kwargs=pfr_kwargs,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        pfr_enabled: Optional[bool] = None,
        pfr_kwargs: Dict = None,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
            pfr_enabled (`bool`, *optional*, defaults to `None`):
                Enable prior-guided far-field rectification.
            pfr_kwargs (`dict`, *optional*, defaults to `None`):
                Keyword arguments forwarded to prior-guided far-field
                rectification.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        if pfr_enabled is None:
            pfr_enabled = getattr(self, "pfr_enabled", False)
        if pfr_kwargs is None:
            pfr_kwargs = getattr(self, "pfr_kwargs", {})

        device = self.device
        rgb_in = rgb_in.to(device)
        da2_depth = self.da2.infer_batch(rgb_in).to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
        da2_depth_latent = self.encode_rgb(da2_depth)

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, da2_depth_latent, depth_latent], dim=1
            )  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        if pfr_enabled:
            depth = self._prior_guided_farfield_rectification(
                depth=depth,
                prior_depth=da2_depth,
                **(pfr_kwargs or {}),
            )

        return depth

    @torch.no_grad()
    def _prior_guided_farfield_rectification(
        self,
        depth: torch.Tensor,
        prior_depth: torch.Tensor,
        top_ratio: float = 0.55,
        prior_far_quantile: float = 0.70,
        ref_near_quantile: float = 0.45,
        min_candidate_ratio: float = 0.015,
        margin: float = 0.03,
        strength: float = 0.25,
        smooth_kernel: int = 21,
        prior_far_is_larger: bool = True,
        depth_far_is_larger: bool = True,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Rectify rare sky-near failures using prior-depth ordering.

        The denoising model already uses DA2 as a 12-channel latent condition.
        PFR does not add any channel and does not require a sky mask. Instead,
        after decoding, it checks the upper image region where the DA2 prior is
        confidently far. If the prediction makes that region closer than lower
        near-reference regions, PFR softly moves only the selected upper-far
        region back to the far side.

        This is intentionally conservative: it is a post-decoding safety guard
        for occasional sky-depth reversals rather than a replacement for the
        learned depth prediction.
        """
        if depth is None or prior_depth is None or strength <= 0:
            return depth

        if prior_depth.shape[-2:] != depth.shape[-2:]:
            prior_depth = F.interpolate(
                prior_depth.float(),
                size=depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        prior_depth = prior_depth.to(device=depth.device, dtype=depth.dtype)

        # Use a single-channel prior if the upstream model returns repeated maps.
        if prior_depth.shape[1] != 1:
            prior_depth = prior_depth.mean(dim=1, keepdim=True)

        bsz, _, h, w = depth.shape
        out = depth.clone()
        top_h = max(1, min(h, int(round(h * float(top_ratio)))))
        min_pixels = max(1, int(round(h * w * float(min_candidate_ratio))))

        top_mask = torch.zeros((h, w), device=depth.device, dtype=torch.bool)
        top_mask[:top_h, :] = True
        lower_mask = ~top_mask

        if smooth_kernel is None:
            smooth_kernel = 1
        smooth_kernel = int(smooth_kernel)
        if smooth_kernel < 1:
            smooth_kernel = 1
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1

        for bi in range(bsz):
            prior = prior_depth[bi, 0]
            pred = depth[bi, 0]

            # Robustly normalize the prior. After this step, larger means farther.
            prior_flat = prior.flatten()
            p_low = torch.quantile(prior_flat, 0.02)
            p_high = torch.quantile(prior_flat, 0.98)
            prior_norm = (prior - p_low) / (p_high - p_low).clamp_min(eps)
            prior_norm = prior_norm.clamp(0.0, 1.0)
            if not prior_far_is_larger:
                prior_norm = 1.0 - prior_norm

            top_values = prior_norm[top_mask]
            if top_values.numel() == 0:
                continue
            far_thr = torch.quantile(top_values, float(prior_far_quantile))
            candidate = top_mask & (prior_norm >= far_thr)
            if int(candidate.sum().item()) < min_pixels:
                continue

            ref_values = prior_norm[lower_mask]
            if ref_values.numel() == 0:
                continue
            near_thr = torch.quantile(prior_norm.flatten(), float(ref_near_quantile))
            reference = lower_mask & (prior_norm <= near_thr)
            if int(reference.sum().item()) < min_pixels:
                # Fallback: use the lower image region when there are too few
                # near-prior reference pixels.
                reference = lower_mask
            if int(reference.sum().item()) < min_pixels:
                continue

            cand_depth = torch.median(pred[candidate])
            ref_depth = torch.median(pred[reference])

            if depth_far_is_larger:
                violation = cand_depth < (ref_depth + margin)
                if not bool(violation.item()):
                    continue
                target = torch.maximum(pred, torch.full_like(pred, ref_depth + margin))
            else:
                violation = cand_depth > (ref_depth - margin)
                if not bool(violation.item()):
                    continue
                target = torch.minimum(pred, torch.full_like(pred, ref_depth - margin))

            candidate_f = candidate.to(dtype=depth.dtype).view(1, 1, h, w)
            if smooth_kernel > 1:
                candidate_f = F.avg_pool2d(
                    candidate_f,
                    kernel_size=smooth_kernel,
                    stride=1,
                    padding=smooth_kernel // 2,
                ).clamp(0.0, 1.0)
            blend = (float(strength) * candidate_f[0, 0]).clamp(0.0, 1.0)
            out[bi, 0] = pred * (1.0 - blend) + target * blend

        return out.clamp(0.0, 1.0)

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
