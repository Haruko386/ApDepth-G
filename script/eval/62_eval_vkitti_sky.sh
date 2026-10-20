#!/usr/bin/env bash
set -euo pipefail

BASE_DATA_DIR="${BASE_DATA_DIR:-/root/Dataset}"
SKY_MASK_NAME="${SKY_MASK_NAME:-mask2former_v1}"
SKY_PREDICTION_ROOT="${SKY_PREDICTION_ROOT:-output/sky_predictions/celeste_full}"
SKY_EVAL_ROOT="${SKY_EVAL_ROOT:-output/sky_eval/celeste_full}"

python -B eval.py \
    --base_data_dir "$BASE_DATA_DIR" \
    --dataset_config config/dataset/data_vkitti_val_sampled.yaml \
    --sky_mask_name "$SKY_MASK_NAME" \
    --prediction_dir "$SKY_PREDICTION_ROOT/vkitti2" \
    --output_dir "$SKY_EVAL_ROOT/vkitti2"
