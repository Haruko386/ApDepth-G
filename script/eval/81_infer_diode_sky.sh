#!/usr/bin/env bash
set -euo pipefail

BASE_DATA_DIR="${BASE_DATA_DIR:-/root/Dataset}"
SKY_MASK_NAME="${SKY_MASK_NAME:-mask2former_v1}"
SKY_PREDICTION_ROOT="${SKY_PREDICTION_ROOT:-output/sky_predictions/celeste_full}"
CHECKPOINT="${CHECKPOINT:-${BASE_CKPT_DIR:+$BASE_CKPT_DIR/sd2-1}}"
: "${CHECKPOINT:?Set CHECKPOINT or BASE_CKPT_DIR to an existing local model}"
model_args=(--checkpoint "$CHECKPOINT" --backbone "${BACKBONE:-sd2}")
if [[ -n "${CELESTE_CHECKPOINT:-}" ]]; then model_args+=(--unet_checkpoint "$CELESTE_CHECKPOINT"); fi
if [[ -n "${DA2_CONFIG:-}" ]]; then model_args+=(--da2_config "$DA2_CONFIG"); fi

python -B infer.py "${model_args[@]}" \
    --base_data_dir "$BASE_DATA_DIR" \
    --dataset_config config/dataset/data_diode_all.yaml \
    --sky_mask_name "$SKY_MASK_NAME" \
    --output_dir "$SKY_PREDICTION_ROOT/diode" \
    --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
