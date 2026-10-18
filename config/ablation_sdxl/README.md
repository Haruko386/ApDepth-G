# SDXL ablations

These eleven configurations mirror `config/ablation/`, but inherit
`config/train_sdxl_demo.yaml`. The full model shares all main training objectives,
progressive dataset sampling and DA2 configuration. SDXL-specific optimizer,
precision, microbatch, learning rate and backbone settings remain unchanged
across the ablations.

| Config | Disabled component |
|---|---|
| `00_full` | None |
| `01_no_far_anchor` | DA2 far-region anchor |
| `02_no_vgc` | All VGC losses |
| `03_no_terminal_snr` | Terminal alignment (keeps v-prediction) |
| `04_no_min_snr` | Min-SNR weighting |
| `05_no_multi_res_noise` | Multi-resolution noise |
| `06_no_offset_noise` | Offset noise |
| `07_no_latent_gradient` | Latent gradient loss |
| `08_no_da2_prior` | DA2 conditioning and VGC together |
| `09_no_region_anchor` | Invalid-region mean anchor |
| `10_no_interior_smoothness` | Invalid-region smoothness |

From the repository root, train one run:

```bash
python train.py --config config/ablation_sdxl/01_no_far_anchor.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir output/ablation_sdxl --no_wandb
```

Run directory: `output/ablation_sdxl/01_no_far_anchor/`.
Repeat with each desired config. After training:

```bash
python script/eval_ablations.py --backbone sdxl \
  --run_root output/ablation_sdxl --output_dir output/ablation_sdxl_eval \
  --base_checkpoint /root/Marigold/pretrained_checkpoint/stable-diffusion-xl-base-1.0 \
  --base_data_dir /root/Dataset --ids 00_full 01_no_far_anchor --main_extra
```

Omit `--ids` only after all eleven runs exist. Each is evaluated on NYUv2 and
KITTI; `--main_extra` adds ETH3D, ScanNet and DIODE for `00_full`. `--dry_run`
prints commands, and `--collect_only` rebuilds the CSV from completed metrics.
Use each run's own `scheduler/` and `inference_settings.json` during inference.
See the [main README](../../README.md) for environment, probe and resume commands.
