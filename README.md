# ApDepth: Aiming for Precise Monocular Depth Estimation Based on Diffusion Models

This repository is based on [Marigold](https://marigoldmonodepth.github.io), CVPR 2024 Best Paper: [**Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation**](https://arxiv.org/abs/2312.02145)

[![Website](doc/badges/badge-website.svg)](https://haruko386.github.io/research)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/developy/ApDepth)

[**Jiawei Wang**](https://haruko386.github.io/),
[Shuai Yuan](https://syjz.teacher.360eol.com/teacherBasic/preview?teacherId=23776)
[Mingbo Lei](https://github.com/Ltohka)

![cover](doc/cover.png)

> [!NOTE]
>
> This project follows the same training methodology as [**ApDepth**](https://github.com/cvrs-ys801/ApDepth) and serves as an extension of its content. It is provided for reference only.

> [!IMPORTANT]
>
> The active experiment on the `nightly` branch uses **SDXL Base 1.0** and retains
> multi-step DDIM depth inference. It trains a 12-channel U-Net conditioned on RGB,
> a frozen DA2-Giant prior, and noisy depth. The default completion objective is the
> newer `residual_snr` mode; the original VGC implementation remains available as
> `legacy`, but the two modes are not applied together. Decoder calibration is a
> separate optional post-training workflow and is not run by the SDXL training config.

## 🛠️ Setup

The model was trained on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, GeForce RTX 6000 Ada Generation

The inference code was tested on:

- Ubuntu 22.04 LTS, Python 3.12.9,  CUDA 11.8, GeForce RTX 4090

### 🪧 A Note for Windows users

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/Haruko386/ApDepth-G.git
cd ApDepth-G
git switch --track origin/nightly
```

### 💻 Dependencies

**Using Conda:** create an environment and install the base dependencies:

```bash
conda create -n apdepth python==3.12.9
conda activate apdepth
pip install -r requirements.txt
```

For the SDXL experiment, install its additional dependencies without replacing the
CUDA-enabled PyTorch installation:

```bash
python -m pip install -r requirements-sdxl.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
python -m bitsandbytes
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing on your images

### 📷 Prepare images

+ Use selected images under `input`. Or place your images in it, for example, under `input/test-image`, and run the following inference command.

### 🎮 Run inference with paper setting

This setting corresponds to our paper. For academic comparison, please run with this setting.

```bash
python run.py \
    --checkpoint prs-eth/marigold-v1-0 \
    --ensemble_size 1 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

You can find all results in `output/in-the-wild_example`. Enjoy!

### ⚙️ Inference settings

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)
  - `--ensemble_size`: Number of inference passes in the ensemble. For LCM `ensemble_size` is more important than `denoise_steps`. Default: 10

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  
  
  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: ~~768~~ `None`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.
- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.

### ⬇ Checkpoint cache

By default, the [checkpoint](https://huggingface.co/prs-eth/marigold-v1-0) is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally:

```bash
bash script/download_weights.sh apdepth-G
```

At inference, specify the checkpoint path:

```bash
python run.py \
    --checkpoint checkpoint/marigold-v1-0 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example\
    --output_dir output/in-the-wild_example
```

## 🦿 Evaluation on test datasets <a name="evaluation"></a>

Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```

Set data directory variable (also needed in evaluation scripts) and download [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset) into corresponding subfolders:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

Run inference and evaluation scripts, for example:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

Note: although the seed has been set, the results might still be slightly different on different hardware.

## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR  # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
into `${BASE_CKPT_DIR}/stable-diffusion-xl-base-1.0`. The refiner and duplicate
single-file checkpoints are not used:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

base_dir = os.environ["BASE_CKPT_DIR"]
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir=os.path.join(base_dir, "stable-diffusion-xl-base-1.0"),
    allow_patterns=[
        "model_index.json",
        "*/config.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
    ],
)
PY
```

The project also expects the existing DA2-Giant checkpoint at
`DA2/checkpoints/depth_anything_v2_vitg.pth`.

The SDXL main-training stack enabled by `config/train_sdxl_demo.yaml` is:

- 12-channel RGB + DA2 prior + noisy-depth conditioning;
- annealed multi-resolution noise and channel-wise offset noise;
- prediction-type-correct Min-SNR weighting and valid-region latent-gradient loss;
- `residual_snr` invalid-region completion, with original VGC retained only as the
  selectable `legacy` mode.

Removed failed experiments are not re-enabled. Decoder calibration remains in the
repository as an independent post-training experiment and is not invoked here.

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

Before a full run, stop other GPU jobs and run the two-update memory probe. It uses
one real HyperSim and one real Virtual KITTI sample per optimizer update, allocates
the 8-bit Adam state, and writes the measured CUDA usage to
`output/sdxl_probe_v1/train_sdxl_probe/memory_profile.json`:

```bash
python train.py \
    --config config/train_sdxl_probe.yaml \
    --base_data_dir "${BASE_DATA_DIR}" \
    --base_ckpt_dir "${BASE_CKPT_DIR}" \
    --output_dir output/sdxl_probe_v1 \
    --no_wandb
```

Only start the full 23,000-update run after the probe finishes at
`effective_iter: 2` with safe VRAM headroom:

```bash
python train.py \
    --config config/train_sdxl_demo.yaml \
    --base_data_dir "${BASE_DATA_DIR}" \
    --base_ckpt_dir "${BASE_CKPT_DIR}" \
    --output_dir output/sdxl_demo_v1 \
    --no_wandb
```

The SDXL config trains the full U-Net with microbatch 1, gradient accumulation 42,
BF16 autocast, activation checkpointing, and 8-bit Adam. It keeps FP32 master
weights and computes the losses in FP32. See [the SDXL demo notes](doc/sdxl_demo.md)
for the measured-vs-estimated memory distinction and current validation scope.

Resume from the matching SDXL run directory:

```bash
python train.py \
    --resume_run output/sdxl_demo_v1/train_sdxl_demo/checkpoint/latest \
    --base_data_dir "${BASE_DATA_DIR}" \
    --base_ckpt_dir "${BASE_CKPT_DIR}" \
    --no_wandb
```

Run 50-step inference with the trained U-Net. The base SDXL checkpoint alone is not
a depth estimator and must not be used without `--unet_checkpoint`:

```bash
python run.py \
    --backbone sdxl \
    --checkpoint "${BASE_CKPT_DIR}/stable-diffusion-xl-base-1.0" \
    --unet_checkpoint output/sdxl_demo_v1/train_sdxl_demo/checkpoint/iter_023000 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/sdxl_demo_eval \
    --denoise_steps 50 \
    --ensemble_size 1 \
    --batch_size 1 \
    --processing_res 768 \
    --seed 2024
```

Only the U-Net is updated by this training command. The repository still contains
the optional decoder-calibration workflow, but it is not part of the command above.
If it is used later, its full multi-step latent cache must be regenerated from the
selected SDXL U-Net; SD2 caches and calibrated decoders are incompatible with SDXL.

> [!IMPORTANT]
>
> Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.



## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🤔 Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (Windows) Invalid DOS bash script on WSL                                                                                                     | Run `dos2unix <script_name>` to convert script format          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |


<!-- ## 🎓 Citation
Please cite our paper:

```bibtex
@InProceedings{haruko26apdepth,
      title={ApDepth: Aiming for Precise Monocular Depth Estimation Based on Diffusion Models},
      author={Haruko386 and Yuan Shuai},
      booktitle = {Under review},
      year={2026}
}
``` -->

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
