# ApDepth-G: SDXL depth estimation (`nightly`)

ApDepth-G adapts diffusion models for monocular depth estimation, with
validity-guided completion (VGC) for regions lacking depth supervision.
Based on [Marigold](https://marigoldmonodepth.github.io/).

[Jiawei Wang](https://haruko386.github.io/),
[Shuai Yuan](https://syjz.teacher.360eol.com/teacherBasic/preview?teacherId=23776),
[Mingbo Lei](https://github.com/Ltohka).

This branch fine-tunes **SDXL Base**. `main` uses SD2. The current `main`
training features are shared through `config/train_marigold.yaml`; SDXL adds
its dual-text/size conditioning, VAE scaling and memory settings.
Run every command below from the repository root.

## What is synchronized from main

| Component | SDXL behavior |
|---|---|
| Progressive dataset sampling | Hypersim/VKITTI 90/10 to 80/20, indexed by optimizer updates, with resume support |
| Standalone DA2 | `config/da2.yaml`: vits/vitb/vitl/vitg and checkpoint path; frozen FP32 |
| Corrected DA2 depth | ImageNet input normalization; near = -1, far = +1 |
| VGC | Region mean anchor, interior smoothness, corrected DA2 far-region anchor |
| Terminal SNR alignment | v-prediction, zero terminal SNR, trailing DDIM, structured-noise fade near the endpoint |
| Training losses/noise | Min-SNR with endpoint floor, multi-resolution noise, offset noise, latent gradient loss |
| Inference consistency | Matching scheduler, DA2 configuration and conditioning switch restored from checkpoints |
| Ablations | All 11 SD2 configurations retained; matching SDXL variants in `config/ablation_sdxl/` |

Only the U-Net is optimized. SDXL keeps the 12-channel RGB/DA2/noisy-depth input,
FP32 VAE, cached dual-CLIP empty prompt, actual-canvas size conditioning,
BF16 autocast, gradient checkpointing, PyTorch SDPA and Adam8bit.
The removed `residual_snr` and decoder-calibration experiments are not part of
current main training. See [the migration/validation notes](doc/sdxl_demo.md).

## Environment and model files

Update the server checkout to this branch before starting a new job:

```bash
git fetch origin nightly
git switch nightly
git pull --ff-only origin nightly
```

Use a Linux CUDA environment with a working PyTorch/torchvision pair. If your
existing SD2 environment already works, clone it before updating dependencies:

```bash
conda create -n apdepth-sdxl --clone marigold
conda activate apdepth-sdxl
python -m pip install -r requirements-sdxl.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
python -m bitsandbytes
```

Replace `marigold` with your existing environment name. For a fresh environment,
install a matching CUDA PyTorch/torchvision pair first, then `requirements-sdxl.txt`.
The SDXL requirements include the runtime packages and do not pin or replace your
CUDA PyTorch build deliberately. No xformers installation is required for SDXL.
The memory configuration requires a BF16-capable GPU; measure its peak with the
probe below before full training.

```bash
export BASE_DATA_DIR=/root/Dataset
export BASE_CKPT_DIR=/root/Marigold/pretrained_checkpoint
export SDXL_BASE="${BASE_CKPT_DIR}/stable-diffusion-xl-base-1.0"
```

Download the base model (no refiner):

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=os.environ["SDXL_BASE"],
    allow_patterns=[
        "model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors", "text_encoder_2/model.safetensors",
    ],
)
PY
```

Place the matching DA2 weights at `DA2/checkpoints/depth_anything_v2_vitg.pth`,
or change `encoder` and `checkpoint` in `config/da2.yaml`. Relative DA2 paths
are resolved from the repository root. Changing `encoder` selects the matching
architecture profile; the weight file must match it. DA2 is not included in a
U-Net checkpoint, so install the same weights on the inference server.

Dataset directories and split files remain in `config/dataset/`. The training
server only needs Hypersim and VKITTI; the inference server needs the five test
datasets. Paths in the existing YAML files are used directly, including their
configured crops and resize settings. No data needs to move between servers.

## GPU probe and training

First run two updates using one sample from each training dataset per update:

```bash
python train.py \
  --config config/train_sdxl_probe.yaml \
  --base_data_dir "${BASE_DATA_DIR}" --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --output_dir output/sdxl_probe_v2 --no_wandb
```

Check successful exit and `effective_iter: 2` in
`output/sdxl_probe_v2/train_sdxl_probe/memory_profile.json`.
The probe measures both training shapes after optimizer states are allocated;
it uses a fixed alternating mixture and saves no model checkpoints.
It is not a full-dataset worst-case memory measurement.

Start **fresh from SDXL Base** with the updated objective:

```bash
python train.py \
  --config config/train_sdxl_demo.yaml \
  --base_data_dir "${BASE_DATA_DIR}" --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --output_dir output/sdxl_main_v2 --no_wandb
```

The config filename is retained for compatibility. This is full U-Net training:
23,000 optimizer updates, microbatch 1, accumulation 42, effective batch 42,
learning rate 1e-5. The scheduler is configuration state, not an optimized model.
Training validation loaders are disabled; evaluate separately as shown below.

Sampling follows `config/dataset/dataset_train.yaml`: 90% Hypersim / 10% VKITTI
through 4,600 completed updates, linearly changes to 80% / 20% by update 18,400,
then stays fixed. At 11,500 completed updates, the next update uses 85% / 15%.
Each microbatch samples one dataset. These are probabilities, not fixed quotas.
Set `probability_schedule.enabled: false` for fixed sampling.
TensorBoard logs target `train_sampling/probability/*` and observed
`train_sampling/observed_ratio/*`. Worker prefetching does not advance the schedule.

Resume **only a run started with this updated configuration**:

```bash
python train.py \
  --resume_run output/sdxl_main_v2/train_sdxl_demo/checkpoint/latest \
  --base_data_dir "${BASE_DATA_DIR}" --base_ckpt_dir "${BASE_CKPT_DIR}" --no_wandb
```

Resume loads that run's saved `config.yaml`, optimizer state and schedule
position; editing repository YAML does not update an existing run. Do not reuse
an SD2 checkpoint or an old epsilon-schedule nightly run for this experiment.
The final inference checkpoint is `checkpoint/iter_023000`; `latest` also stores
training state (saved every 250 updates), and backups are saved every 2,000.
A full FP32 SDXL U-Net is about 9.56 GiB; reserve ample checkpoint disk space.

## Inference on your images

Copy the entire iteration directory to the inference server, including `unet/`,
`scheduler/` and `inference_settings.json`. Keep the run's `config.yaml` as well.
Both servers need the same SDXL base and DA2 weights.

```bash
export TRAINED=output/sdxl_main_v2/train_sdxl_demo/checkpoint/iter_023000
python run.py \
  --backbone sdxl --checkpoint "${SDXL_BASE}" --unet_checkpoint "${TRAINED}" \
  --input_rgb_dir input/in-the-wild_example --output_dir output/sdxl_images_v2 \
  --denoise_steps 50 --ensemble_size 1 --batch_size 1 --processing_res 768 --seed 2024
```

`run.py` processes all supported images in the input directory. The example uses
FP32 inference and the ordinary safetensors files downloaded above. Only an
explicit `--fp16` (alias `--half_precision`) selects FP16; the loader casts the
ordinary weights, so separate fp16 variant files are not required.
The SDXL VAE and DA2 remain FP32. DA2 and ablation switches restore automatically
from checkpoint metadata. If the DA2 path differs between servers, pass
`--da2_config config/da2.yaml` with the same encoder/weights at their new location.

## Dataset inference and evaluation

Use the dataset runner for SDXL; the old numbered shell scripts retain their
legacy SD2 defaults. All five datasets, in the main-table order:

```bash
python script/eval_datasets.py \
  --backbone sdxl --checkpoint "${SDXL_BASE}" --unet_checkpoint "${TRAINED}" \
  --base_data_dir "${BASE_DATA_DIR}" --output_dir output/sdxl_benchmark_v2 \
  --datasets nyu kitti eth3d scannet diode
```

Defaults: 50 DDIM steps, ensemble 1, seed 2024, native processing resolution
(`--processing_res 0`) and least-square scale/shift alignment on valid GT.
Both evaluation runners also default to FP32; pass `--fp16` explicitly to
forward FP16 to their inference commands. Metric evaluation is unchanged.
The runner forwards the matching checkpoint and stops on failed commands.
Use `--dry_run` to inspect commands without loading data or weights.
Use `--datasets nyu kitti` for a subset. A fresh output directory avoids mixing
predictions from different models. If native-resolution images exceed memory,
choose a lower `--processing_res` and report that setting with your results.

To split inference and evaluation into separate jobs:

```bash
python script/eval_datasets.py \
  --phase infer --backbone sdxl \
  --checkpoint "${SDXL_BASE}" --unet_checkpoint "${TRAINED}" \
  --base_data_dir "${BASE_DATA_DIR}" --output_dir output/sdxl_benchmark_split_v2

python script/eval_datasets.py \
  --phase eval --base_data_dir "${BASE_DATA_DIR}" \
  --output_dir output/sdxl_benchmark_split_v2
```

The evaluation phase needs predictions and GT, not model weights.
Results are `<output>/<dataset>/prediction/*.npy` (with dataset subdirectories),
`<output>/<dataset>/eval/metrics.json` and `<output>/paper_results.csv`.
The CSV reports AbsRel x 100 and delta1 in percent; raw metrics stay in JSON.
Benchmark valid-depth metrics do not measure unlabeled sky accuracy.

The equivalent direct NYUv2 commands are:

```bash
python infer.py \
  --backbone sdxl --checkpoint "${SDXL_BASE}" --unet_checkpoint "${TRAINED}" \
  --dataset_config config/dataset/data_nyu_test.yaml --base_data_dir "${BASE_DATA_DIR}" \
  --output_dir output/sdxl_nyu_direct/prediction \
  --denoise_steps 50 --ensemble_size 1 --processing_res 0 --seed 2024

python eval.py \
  --dataset_config config/dataset/data_nyu_test.yaml --base_data_dir "${BASE_DATA_DIR}" \
  --prediction_dir output/sdxl_nyu_direct/prediction \
  --output_dir output/sdxl_nyu_direct/eval --alignment least_square
```

`eval.py` is backbone-independent. Other configs are `data_kitti_eigen_test.yaml`,
`data_eth3d.yaml`, `data_scannet_val.yaml`, `data_diode_all.yaml` in `config/dataset/`.

## SDXL ablations

Use the matching [SDXL ablation configs](config/ablation_sdxl/README.md):

```bash
python train.py --config config/ablation_sdxl/00_full.yaml \
  --base_data_dir "${BASE_DATA_DIR}" --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --output_dir output/ablation_sdxl --no_wandb

python script/eval_ablations.py --backbone sdxl \
  --run_root output/ablation_sdxl --output_dir output/ablation_sdxl_eval \
  --base_checkpoint "${SDXL_BASE}" --base_data_dir "${BASE_DATA_DIR}" \
  --ids 00_full --main_extra
```

Train the other config stems separately. Omitting `--ids` evaluates all eleven
runs on NYUv2/KITTI; `--main_extra` adds ETH3D/ScanNet/DIODE for `00_full` only.
Every ablation keeps its own saved scheduler and inference switches, including
`03_no_terminal_snr` and `08_no_da2_prior`.

## Validation and license

Offline CPU tests use real tiny SDXL U-Net, dual CLIP and VAE components, with
a synthetic DA2 prior. They check configuration parity, training/backpropagation,
resume, scheduler/DA2 restoration, multi-step inference and dataset commands.
Full SDXL weights, BF16/Adam8bit CUDA execution and depth quality must still be
validated on the server; CPU tests cannot establish those results.

```bash
python -m pip install pytest
python -m pytest tests -q
```

Apache 2.0; see [LICENSE.txt](LICENSE.txt). DA2 has its own
[license](DA2/LICENSE); pretrained weights retain their upstream licenses.
