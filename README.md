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
```

### 💻 Dependencies

 **Using Conda:** 
    Alternatively, create a Python native virtual environment and install dependencies into it:

    conda create -n apdepth python==3.12.9
    conda activate apdepth
    pip install -r requirements.txt

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

DIODE evaluation follows PPD's released mask refinement: valid GT must be finite
and within [0.6, 350] metres, and pixels with an unnormalized Sobel response
`abs(Gx) + abs(Gy) > 0.3` are excluded before alignment and metric calculation.
`metrics.json` records this protocol. Existing predictions can be reused, but
metrics from the previous unfiltered protocol must be recomputed.

For per-image supervision coverage records and local bar-chart generation, see
[the coverage guide](doc/supervision_coverage.md).

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

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2)
into `${BASE_CKPT_DIR}/sd2-1`.

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

The current `main` experiment keeps the original VGC objective and replaces the failed
`residual_snr` objective with terminal-SNR-aligned diffusion training. It uses
v-prediction, zero terminal SNR and trailing DDIM timesteps; it does not add sky labels
or another invalid-region loss. See [the experiment note](doc/terminal_snr_alignment.md).

Start a new training run from SD2. Do not resume a checkpoint trained with the old
epsilon schedule:

```bash
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir "${BASE_DATA_DIR}" \
  --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --output_dir output/unet_dynamic_mix_v1 \
  --no_wandb
```

Resume only the matching run:

```bash
python train.py \
  --resume_run output/unet_dynamic_mix_v1/train_marigold/checkpoint/latest \
  --base_data_dir "${BASE_DATA_DIR}" \
  --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --no_wandb
```

The checkpoint contains both `unet/` and its matching `scheduler/`. Keep them together
and pass the checkpoint parent directory at inference:

```bash
python run.py \
  --checkpoint "${BASE_CKPT_DIR}/sd2-1" \
  --unet_checkpoint output/unet_dynamic_mix_v1/train_marigold/checkpoint/iter_008000 \
  --input_rgb_dir output/out \
  --output_dir output/unet_terminal_snr_eval \
  --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
```

Only the U-Net is optimized. The scheduler is configuration state saved to keep
training and multi-step inference mathematically consistent.

### Progressive dataset sampling

Set the initial and final sampling probabilities in
`config/dataset/dataset_train.yaml`, following the order of `dataset_list`
(Hypersim, Virtual KITTI):

```yaml
dataset:
  train:
    prob_ls: [0.9, 0.1]
    probability_schedule:
      enabled: true
      start_fraction: 0.2
      end_fraction: 0.8
      final_prob_ls: [0.8, 0.2]
```

The default schedule spans `max_iter: 23000` **optimizer updates**. It keeps the
90%/10% mixture for the first 20% of training, linearly changes it to 80%/20%
during the middle 60%, and holds 80%/20% for the final 20%. After 4,600, 11,500,
and 18,400 completed updates, the VKITTI probability for the next update is
10%, 15%, and 20%, respectively. Each microbatch comes from one dataset. These
values are sampling probabilities rather than required sample quotas per update.
This configuration does not change the learning rate, loss weights, or total
number of training updates.

Set `enabled: false` to use fixed probabilities. Legacy configurations without
`probability_schedule` also retain fixed sampling. The paper's ablations all use
the default progressive schedule.

In TensorBoard, `train_sampling/probability/<dataset>` records the target
probability used for each optimizer update, while
`train_sampling/observed_ratio/<dataset>` records the proportion of samples
actually consumed in that update. The observed ratio fluctuates with random
sampling; use smoothing to inspect its trend. The schedule uses each batch's
logical training position so that worker prefetching does not advance it early.
It also supports gradient accumulation across epoch boundaries.

`--resume_run` reads `config.yaml` from the original run directory and restores
the global schedule position using the checkpoint's `effective_iter` and
`n_batch_in_epoch`, rather than restarting at 10%. Editing the repository's
configuration does not automatically change an existing run. Runs without a
schedule configuration continue with fixed probabilities. Restoring schedule
progress does not guarantee bitwise reproducibility of the entire training run.
Use a new output directory for a new control experiment, and keep `max_iter` and
the effective batch size unchanged when resuming.

### Standalone DA2 configuration

`config/da2.yaml` defines the DA2 encoder, its architecture parameters, and the
checkpoint path. The default encoder is `vitg`. The main training configuration
includes this file through `base_config`.

```yaml
da2:
  encoder: vitg  # Choices: vits / vitb / vitl / vitg
  checkpoint: null
  # Keep the four architecture profiles supplied under model_configs.
```

Change `encoder` to select another architecture profile. With `checkpoint: null`,
the checkpoint path is `DA2/checkpoints/depth_anything_v2_<encoder>.pth`.
You can also specify a custom path. Relative checkpoint paths are resolved from
the repository root; absolute paths are used directly. The weights must match
the selected encoder. DA2 remains frozen, in evaluation mode and FP32, and follows
the pipeline's execution device.

Both `run.py` and `infer.py` accept `--da2_config`. For example:

```bash
python run.py \
  --checkpoint "${BASE_CKPT_DIR}/sd2-1" \
  --unet_checkpoint output/unet_dynamic_mix_v1/train_marigold/checkpoint/iter_008000 \
  --da2_config output/unet_dynamic_mix_v1/train_marigold/config.yaml \
  --input_rgb_dir input/in-the-wild_example \
  --output_dir output/dynamic_mix_eval \
  --denoise_steps 50 --ensemble_size 1 --seed 2024
```

Pass the training run's saved `config.yaml` to reuse its DA2 settings. If this
argument is omitted, the DA2 configuration saved in a complete pipeline takes
precedence. Legacy models without that configuration use the repository's
`config/da2.yaml`. A U-Net checkpoint does not include DA2 weights; use the same
DA2 encoder and weights for training and inference. These settings do not change
DA2 input normalization or the conversion to the project's depth range.

### Paper ablations and evaluation

The [ablation guide](config/ablation/README.md) maps all eleven configurations to
the paper tables and provides training and evaluation commands. All ablations
share progressive sampling and are evaluated on NYUv2 and KITTI. Only the full
model is additionally evaluated on ETH3D, ScanNet, and DIODE for the main table.
`script/eval_ablations.py` runs this plan and exports paper-scale metrics to CSV.

New checkpoints include `inference_settings.json`, which restores DA2 settings
and the conditioning switch automatically during inference. Existing checkpoints
can use their original run's `config.yaml`. Keep the matching `scheduler/` with
each checkpoint, including the terminal-alignment ablation's own scheduler.

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
