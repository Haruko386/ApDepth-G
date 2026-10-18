# Main-training ablation experiments

All 11 configurations inherit `config/train_marigold.yaml`. Train each one
independently from the same SD2 base model for 23,000 optimizer updates, with an
effective batch size of 42 and seeds of 2024. All runs share progressive dataset
sampling (VKITTI 10% to 20%). There is no sampling-strategy ablation.

## Configuration matrix

| ID | Configuration | Change from the full model | Paper table |
|---|---|---|---|
| 00 | `00_full.yaml` | None; shared reference | Main, core, supporting |
| 01 | `01_no_far_anchor.yaml` | Set only the far-field anchor weight to zero | Core |
| 02 | `02_no_vgc.yaml` | Disable all three VGC terms; keep prior conditioning | Core |
| 03 | `03_no_terminal_snr.yaml` | Disable the endpoint-alignment package; keep velocity prediction | Core |
| 04 | `04_no_min_snr.yaml` | Disable Min-SNR weighting | Supporting |
| 05 | `05_no_multi_res_noise.yaml` | Disable multi-resolution noise; retain offset noise | Supporting |
| 06 | `06_no_offset_noise.yaml` | Set offset-noise strength to zero | Supporting |
| 07 | `07_no_latent_gradient.yaml` | Set valid-region gradient weight to zero | Supporting |
| 08 | `08_no_da2_prior.yaml` | Zero prior-conditioning channels and disable VGC | Core |
| 09 | `09_no_region_anchor.yaml` | Set only the mean-anchor weight over U to zero | Core |
| 10 | `10_no_interior_smoothness.yaml` | Set only U-U smoothness weight to zero | Core |

Rows 01, 09, and 10 each remove one of VGC's three constraints while retaining
the other two. Comparing 02 with 08 isolates prior conditioning when VGC is absent
in both. Comparing 00 with 08 measures all teacher information together.
Row 03 changes zero-terminal-SNR, timestep spacing, the SNR floor, and terminal
noise alignment as a group; it does not identify any one of those changes alone.
All rows use corrected DA2 preprocessing and near/far ordering. Those correctness
fixes are shared infrastructure, not optional ablation components.

## Train all configurations

Run from the repository root in the training environment. Set the paths for your
server before starting. The DA2 encoder and weights are configured in
`config/da2.yaml`; a custom checkpoint must match the encoder.

```bash
export BASE_DATA_DIR=/root/Dataset
export BASE_CKPT_DIR=/root/Marigold/pretrained_checkpoint

for cfg in config/ablation/*.yaml; do
  python train.py \
    --config "$cfg" \
    --base_data_dir "$BASE_DATA_DIR" \
    --base_ckpt_dir "$BASE_CKPT_DIR" \
    --output_dir output/ablation \
    --no_wandb || break
done
```

The output layout is `output/ablation/<config_stem>/`. Do not start one ablation
from another ablation's weights. Do not resume an old fixed-mixture run and label
it as the new progressive-mixture reference. Existing output directories are not
overwritten. To resume an interrupted matching run, use its
`checkpoint/latest` with `--resume_run`, then continue the remaining configs.

Prioritize 00, 01, 02, 09, and 10 for the VGC mechanism; then 08 and 03, followed
by the supporting components 04-07. All eleven are required to fill every
currently specified ablation row.

## Evaluate ablations on NYUv2 and KITTI

Use the final `iter_023000` checkpoint for every row. `latest` may precede the
final update; do not choose different iterations by test performance. The runner
reads a run's saved `max_iter` to locate its final checkpoint.

Preview commands without loading models or data:

```bash
python script/eval_ablations.py \
  --run_root output/ablation \
  --base_checkpoint "$BASE_CKPT_DIR/sd2-1" \
  --base_data_dir "$BASE_DATA_DIR" \
  --output_dir output/ablation_eval \
  --dry_run
```

Remove `--dry_run` to execute. Add `--ids 00_full 09_no_region_anchor` to run a
subset. The runner performs inference and least-squares scale/shift evaluation
on NYUv2 and KITTI for each selected configuration, with 50 DDIM steps, one
ensemble member, seed 2024, and native processing resolution. Dataset-specific
crops and validity masks are supplied by the existing dataset YAML files.

Every checkpoint now includes `inference_settings.json`. `run.py` and `infer.py`
automatically restore its DA2 configuration and conditioning switch. For old
checkpoints, keep the original `<run>/config.yaml` beside the `checkpoint/`
directory, or copy it into the selected checkpoint directory. Do not discard
`scheduler/`: row 03 intentionally uses its own nonzero-terminal-SNR, leading
schedule, and row 08 must use zero prior channels during inference as in training.

Results are written to:

```text
output/ablation_eval/<config_stem>/<dataset>/prediction/
output/ablation_eval/<config_stem>/<dataset>/eval/metrics.json
output/ablation_eval/paper_results.csv
```

`metrics.json` contains the raw metrics. `paper_results.csv` contains
`AbsRel_x100` and `delta1_percent`, already scaled for the paper. Do not multiply
these CSV columns by 100 again. To regenerate the CSV from completed evaluations:

```bash
python script/eval_ablations.py \
  --output_dir output/ablation_eval \
  --collect_only
```

The runner refuses to overwrite prediction/evaluation directories. For additional
subsets use the same `--ids` in evaluation and collection, or use a new output
directory for a rerun.

## Which results fill which tables?

| Paper table | Training configurations | Evaluation datasets |
|---|---|---|
| Main comparison | 00 for the ApDepth-G row | NYUv2, KITTI, ETH3D, ScanNet, DIODE |
| Core ablations | 00, 01, 02, 03, 08, 09, 10 | NYUv2 and KITTI only |
| Supporting components | 00, 04, 05, 06, 07 | NYUv2 and KITTI only |

Thus, training all eleven configurations and evaluating them on NYUv2/KITTI fills
both ablation tables. Filling the complete model's main-table row also requires
ETH3D, ScanNet, and DIODE. Reuse 00's NYUv2/KITTI results across all three tables;
the shared reference is trained only once. Published baseline rows and the
original ApDepth row are separate comparisons, not outputs of these configs.

Add `--main_extra` to evaluate those three additional datasets for 00 only:

```bash
python script/eval_ablations.py \
  --run_root output/ablation \
  --base_checkpoint "$BASE_CKPT_DIR/sd2-1" \
  --base_data_dir "$BASE_DATA_DIR" \
  --output_dir output/paper_eval \
  --main_extra
```

This performs 25 dataset evaluations: eleven times two, plus three for the full
model. Use `--main_extra --collect_only` when collecting this complete plan.
Sky comparisons use fixed KITTI test images and are qualitative; no separate
sky-mask dataset or sky-score table is required. Valid-depth benchmark metrics
alone do not establish the quality of predictions in unobserved sky regions.
