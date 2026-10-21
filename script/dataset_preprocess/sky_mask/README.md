# Sky mask quick start

Prepare VKITTI2, KITTI, and DIODE and put them under`BASE_DATA_DIR`

If the masks already exist, evaluate existing depth predictions directly:

```bash
export BASE_DATA_DIR=/path/to/your/DATA
export SKY_PREDICTION_ROOT=output/sky_predictions/celeste_full
bash script/eval/62_eval_vkitti_sky.sh
```

To run both inference and evaluation for all datasets:

```bash
export BASE_DATA_DIR=/path/to/your/DATA
export CHECKPOINT=/path/to/local/base/sd2-1
export CELESTE_CHECKPOINT=/path/to/local/trained/checkpoint
bash script/eval/0_infer_eval_all_sky.sh
```

Only if masks have **not** been generated, use a local Mask2Former checkpoint once:

For the SSD branch, also set `BACKBONE=ssd`, point `CHECKPOINT` to the local
`SSD-1B` base and `CELESTE_CHECKPOINT` to its trained depth checkpoint. Use separate
`SKY_PREDICTION_ROOT` and `SKY_EVAL_ROOT` directories for SSD results. Alternatively,
use `python script/eval_ssd.py --sky_mask_name mask2former_v1`; see the
[SSD README](../../../README.md#sky-masks-and-sky-metrics) for complete commands.

```bash
python -B script/dataset_preprocess/sky_mask/segment_sky.py \
  --base_data_dir "$BASE_DATA_DIR" \
  --checkpoint /path/to/local/mask2former \
  --datasets vkitti2 kitti diode \
  --data_split data_split/vkitti/vkitti_val_sampled.txt \
               data_split/kitti/eigen_test_files_with_gt.txt \
               data_split/diode/diode_val_all_filename_list.txt \
  --output_dir output/sky_segmentation/mask2former_v1
```

Masks are saved along with each dataset's source as `sky_masks/mask2former_v1/<RGB relative path>.sky.png`.
