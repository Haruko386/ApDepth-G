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
git clone https://github.com/cvrs-ys801/ApDepth-G.git
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
    --config config/run.yaml \
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
- `--config`: Runtime YAML configuration shared by `run.py` and `infer.py`. Default: `config/run.yaml`.

### Prior-guided far-field rectification (PFR)

Both inference entry points load PFR settings from `config/run.yaml`. PFR is a conservative post-processing guard for occasional far-field or sky depth reversals. It is disabled by default, so adding `--config config/run.yaml` does not change predictions until `enabled` is set to `true`.

```yaml
prior_guided_farfield_rectification:
  enabled: true
  top_ratio: 0.50
  prior_far_quantile: 0.75
  ref_near_quantile: 0.45
  min_candidate_ratio: 0.015
  margin: 0.02
  strength: 0.15
  smooth_kernel: 21
  prior_far_is_larger: true
  depth_far_is_larger: true
  eps: 1.0e-06
```

The fields have the following meanings:

- `enabled`: Enables or disables PFR.
- `top_ratio`: Fraction of the upper image searched for far-field candidates.
- `prior_far_quantile`: Prior-depth quantile used to select confident far-field pixels in the upper region.
- `ref_near_quantile`: Prior-depth quantile used to select near-reference pixels in the lower region.
- `min_candidate_ratio`: Minimum candidate/reference area relative to the full image.
- `margin`: Minimum separation enforced between far candidates and near-reference depth.
- `strength`: Blend strength of the rectification; `0` leaves the prediction unchanged.
- `smooth_kernel`: Smoothing-kernel size for the candidate mask; even values are promoted to the next odd value.
- `prior_far_is_larger`: Set according to whether larger prior values mean farther depth.
- `depth_far_is_larger`: Set according to whether larger output values mean farther depth.
- `eps`: Numerical stability constant used during prior normalization.

To use another runtime file, pass it explicitly:

```bash
python run.py \
    --config path/to/run.yaml \
    --checkpoint prs-eth/marigold-v1-0 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

Dataset inference uses the same runtime configuration independently of its dataset configuration:

```bash
python infer.py \
    --config config/run.yaml \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --base_data_dir /path/to/datasets \
    --output_dir output/nyu
```

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

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

Run training script

```bash
python train.py --config config/train_marigold.yaml --no_wandb
```

Resume from a checkpoint, e.g.

```bash
python train.py --resume_run output/train_marigold/checkpoint/latest --no_wandb
```

Evaluating results

Only the U-Net is updated and saved during training. To use the inference pipeline with your training result, replace `unet` folder in Marigold checkpoints with that in the `checkpoint` output folder. Then refer to [this section](#evaluation) for evaluation.

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
