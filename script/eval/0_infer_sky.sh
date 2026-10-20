#!/usr/bin/env bash
# Manual sky inference using the existing per-dataset configs.
export BASE_DATA_DIR="${BASE_DATA_DIR:-/root/Dataset}"
export MASK_RUN="${MASK_RUN:-output/sky_segmentation/mask2former_v1}"

for dataset in vkitti2 kitti diode; do
    python -B infer.py \
        --dataset_config "$MASK_RUN/$dataset/depth_inference.yaml" \
        --base_data_dir "$BASE_DATA_DIR" \
        --output_dir "output/sky_infer/$dataset" \
        --denoise_steps 50 \
        --ensemble_size 5 \
        --processing_res 0 \
        --seed 1234
done
