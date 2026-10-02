"""SDXL base adaptation: RGB + DA2 + noisy depth, with multi-step DDIM."""

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from .marigold_pipeline import MarigoldPipeline


class SDXLDepthPipeline(MarigoldPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        scale_invariant: bool = True,
        shift_invariant: bool = True,
        default_denoising_steps: int = 50,
        default_processing_resolution: int = 768,
    ):
        # The upstream SDXL checkpoint uses Euler; our diffusion objective uses
        # its alpha schedule with multi-step DDIM, with no latent x0 clipping.
        scheduler = DDIMScheduler.from_config(scheduler.config, clip_sample=False)
        super().__init__(unet, vae, scheduler, text_encoder, tokenizer,
                         scale_invariant, shift_invariant,
                         default_denoising_steps, default_processing_resolution)
        self.register_modules(text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2)
        self.rgb_latent_scale_factor = float(vae.config.scaling_factor)
        self.depth_latent_scale_factor = self.rgb_latent_scale_factor
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        if vae.config.latent_channels != 4 or self.vae_scale_factor != 8:
            raise ValueError("This demo requires SDXL's four-channel, 8x VAE")
        if unet.config.addition_embed_type != "text_time":
            raise ValueError("Expected SDXL base U-Net with text_time conditioning")
        # Original SDXL VAE is not fp16-safe. Keep its encoder and decoder FP32.
        self.vae.to(dtype=torch.float32).requires_grad_(False).eval()
        self.empty_pooled_embed = None
        self.encode_empty_text()
        # Only the empty prompt is used. Cache it once, then release both CLIPs
        # before pipeline.to(cuda); text weights consume no training VRAM.
        self.register_modules(text_encoder=None, text_encoder_2=None)

    @torch.no_grad()
    def encode_empty_text(self):
        if self.empty_text_embed is not None:
            return
        embeddings = []
        for tokenizer, encoder in ((self.tokenizer, self.text_encoder),
                                   (self.tokenizer_2, self.text_encoder_2)):
            encoder.requires_grad_(False).eval()
            inputs = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length,
                               truncation=True, return_tensors="pt")
            output = encoder(inputs.input_ids.to(encoder.device), output_hidden_states=True)
            embeddings.append(output.hidden_states[-2].detach().float().cpu())
        self.empty_text_embed = torch.cat(embeddings, dim=-1)
        self.empty_pooled_embed = output.text_embeds.detach().float().cpu()

    def predict_noise(self, latents, timesteps, text_embed):
        batch, _, height, width = latents.shape
        # We condition on the actual resized training/inference canvas. No crop
        # beyond that canvas is claimed. Keep identical convention at inference.
        height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        dtype, device = self.unet.dtype, latents.device
        time_ids = torch.tensor([height, width, 0, 0, height, width],
                                dtype=dtype, device=device).unsqueeze(0).expand(batch, -1)
        pooled = self.empty_pooled_embed.to(device=device, dtype=dtype).expand(batch, -1)
        return self.unet(
            latents.to(dtype=dtype), timesteps,
            encoder_hidden_states=text_embed.to(device=device, dtype=dtype),
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample

    def encode_rgb(self, rgb_in):
        with torch.autocast(device_type=rgb_in.device.type, enabled=False):
            return super().encode_rgb(rgb_in.float())

    def decode_depth(self, depth_latent):
        with torch.autocast(device_type=depth_latent.device.type, enabled=False):
            return super().decode_depth(depth_latent.float())
