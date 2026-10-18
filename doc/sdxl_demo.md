# SDXL branch migration and validation

The current nightly workflow is documented in [README.md](../README.md).
The historical `train_sdxl_demo.yaml` name is retained; it now inherits the
current main objective rather than the old epsilon/legacy-VGC configuration.

## Comparison against local main

Migration source: local `main` at `4c9d3dd`.
Starting nightly: fetched `origin/nightly` at `9aeb19e`.

| Main change | Integration on nightly |
|---|---|
| `463b8b5`: terminal diffusion alignment | Shared schedule utilities/trainer; checkpoint DDIM saved and restored |
| `7e09644`: corrected DA2 far-field anchor | Shared corrected DA2 preprocessing and depth convention; VGC far-cell selection and loss |
| `73f4d78`: ablation matrix | Original SD2 configs plus 11 SDXL configurations with identical component switches |
| `45ef3fc`: progressive sampling and standalone DA2 | Shared sampler, prefetch-safe update position, resume and TensorBoard logging; SDXL constructor forwards DA2 arguments |
| `9b29f4b`: complete ablations/inference settings | Region/smoothness ablations, checkpoint metadata, scheduler and DA2 restoration; SDXL-aware evaluation runners |
| `4c9d3dd`: resolution notation | Shared dataset configs; no change to image preprocessing |

The previous residual-SNR and decoder-calibration experiments were removed in
main and are removed here too. Main's historical experiment notes are retained;
they do not establish SDXL accuracy. Uncommitted local paper deletions, distillation
work and supervision-coverage scripts are not part of this migration.

## SDXL-specific behavior retained

- SDXL Base, no refiner, full U-Net tuning, 12 input channels.
- Two CLIP hidden-state embeddings plus pooled text and actual-canvas time IDs.
  Empty prompt embeddings are cached on CPU; text encoders are released.
- VAE scaling read from its config (official base: 0.13025), FP32 VAE and DA2.
- FP32 U-Net parameters, BF16 autocast, gradient checkpointing, SDPA, Adam8bit.
- Effective batch 42 with microbatch 1; LR 1e-5, 23,000 optimizer updates.
- Fixed two-dataset GPU memory probe in `config/train_sdxl_probe.yaml`.
- Backbone mismatch detection when applying a U-Net checkpoint.

## Verification scope

Local validation on 2026-10-18: **37 tests and 6 subtests passed**. The `--help`
entry points for training, image inference, dataset inference, evaluation and
both evaluation runners also passed. CPU test runtime: PyTorch 2.4.1,
Diffusers 0.35.2, Transformers 4.57.0 and Accelerate 1.10.1.
The real image/dataset CLI tests generate depth arrays using tiny SDXL weights;
the evaluation CLI test reads synthetic NYUv2 GT and writes metrics JSON.
Metric reset also supports pandas copy-on-write (tested with pandas 3.0.6).
Inference defaults to FP32; explicit FP16 loads ordinary checkpoint files and
keeps the SDXL VAE in FP32. This is separate from training's BF16 autocast setting.

Tests use real small Diffusers SDXL U-Net, two CLIP encoders and an 8x VAE, with
a synthetic frozen DA2 prior instead of external weights. They cover conditioning,
latent scaling, DA2 configuration, gradient checkpointing, optimizer updates,
checkpoint restore, matching v-prediction/DDIM and multi-step seeded inference.
Shared tests cover conservative validity masks, far-region loss, noise alignment,
dynamic sampling across accumulation/epoch boundaries and schedule position after
resume. Dataset CLI tests check command forwarding and evaluation on synthetic GT.

Full SDXL/DA2 weights and datasets are not present locally. BF16 CUDA, Adam8bit
CUDA kernels, full-resolution peak memory, runtime and accuracy still require
server validation. Run the GPU probe in the README first, then start a new run.
The probe uses both shapes after optimizer allocation, but does not establish the
worst-case memory for all images. CPU tests are not a claim of GPU validation.

Native-resolution evaluation can exceed memory for large images even with
ensemble 1. A reduced processing resolution is an explicit experimental setting
and should be recorded rather than silently changed.
