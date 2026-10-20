#!/usr/bin/env bash
set -euo pipefail

# Each dataset uses its own split and masks already stored under BASE_DATA_DIR.
bash script/eval/61_infer_vkitti_sky.sh
bash script/eval/62_eval_vkitti_sky.sh
bash script/eval/71_infer_kitti_sky.sh
bash script/eval/72_eval_kitti_sky.sh
bash script/eval/81_infer_diode_sky.sh
bash script/eval/82_eval_diode_sky.sh
