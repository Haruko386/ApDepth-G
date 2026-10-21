# SSD ablations

Each YAML inherits `config/train_ssd.yaml` and applies the same change as its
counterpart in `config/ablation`. All retain main's 8,000 updates and effective
batch 42. Tests compare all resolved leaves, allowing only SSD-specific settings.

| Config | Change from full model |
| --- | --- |
| `00_full` | Full method |
| `01_no_far_anchor` | Disable corrected DA2 far-region anchor |
| `02_no_vgc` | Disable VGC |
| `03_no_terminal_snr` | Disable terminal SNR alignment, matching leading DDIM |
| `04_no_min_snr` | Disable Min-SNR weighting |
| `05_no_multi_res_noise` | Disable multi-resolution noise |
| `06_no_offset_noise` | Disable offset noise |
| `07_no_latent_gradient` | Disable latent gradient loss |
| `08_no_da2_prior` | Zero DA2 input and disable VGC; retain 12 channels |
| `09_no_region_anchor` | Disable region mean anchor |
| `10_no_interior_smoothness` | Disable interior smoothness |
| `11_8ch_no_vgc` | Genuine 8-channel input, no VGC |
| `12_8ch_vgc` | Genuine 8-channel input, DA2 only for training-time VGC |
| `N1_no_snr_floor` | Disable terminal Min-SNR floor |
| `N2_no_terminal_noise_fade` | Disable fade independently of terminal alignment |

```bash
python train.py --config config/ablation_ssd/01_no_far_anchor.yaml \
  --base_data_dir "$BASE_DATA_DIR" --base_ckpt_dir "$BASE_CKPT_DIR" \
  --output_dir output/ssd_ablations --no_wandb
```

Infer using that run's matching U-Net and scheduler: `--backbone ssd` and
`--unet_checkpoint output/ssd_ablations/01_no_far_anchor/checkpoint/iter_008000`.
Conditioning and scheduler settings restore from checkpoint metadata.
