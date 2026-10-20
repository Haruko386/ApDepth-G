<div align="center"><h1>Mind The Sky: Resolving Sky-Depth Inconsistency with Validity-Guided Diffusion</h1></div>

<div align=center>
<a href="https://haruko386.github.io/research"><img src="assets/badges/badge-website.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache--2.0-929292"></a>
<a href="https://huggingface.co/developy/Celeste"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green"></a>
</div>

![cover](assets\badges\banner-hor.png)


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
git clone https://github.com/cvrs-yx801/Celeste.git
cd Celeste
```

### 💻 Dependencies

 **Using Conda:** 

```bash
conda create -n celeste python==3.12.9
conda activate celeste
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing on your images

### 📷 Prepare images

+ Use our selected images under `input`. Or place your images in it, for example, under `input/example-1`, and run the following inference command.

### 🎮 Run inference with paper setting

This setting corresponds to our paper. For academic comparison, please run with this setting.

```bash
python run.py \
    --checkpoint path/to/model/checkpoint \
    --denoise_steps 50 \
    --processing_res 0 \
    --ensemble_size 10 \
    --input_rgb_dir input/example-1 \
    --output_dir output/example-1
```

You can find all results in `output/example-1`. Enjoy!

### ⚙️ Inference settings

The default settings are optimized for the best result. However, the behavior of the code can be customized:

- Trade-offs between the **accuracy** and **speed** (for both options, larger values result in better accuracy at the cost of slower inference.)
  - `--ensemble_size`: Number of inference passes in the ensemble. Default: 5.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution. This gives the best quality, as Stable Diffusion, from which Marigold is derived, performs best at 768x768 resolution.  

  - `--processing_res`: the processing resolution; set as 0 to process the input resolution directly. When unassigned (`None`), will read default setting from model config. Default: `0`.
  - `--output_processing_res`: produce output at the processing resolution instead of upsampling it to the input resolution. Default: False.
  - `--resample_method`: the resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic`, or `nearest`. Default: `bilinear`.
- `--denoise_steps`: Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps. Default: 50.
- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to have faster speed and reduced VRAM usage, but might lead to suboptimal results.
- `--seed`: Random seed can be set to ensure additional reproducibility. Default: None (unseeded). Note: forcing `--batch_size 1` helps to increase reproducibility. To ensure full reproducibility, [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms) needs to be used.
- `--batch_size`: Batch size of repeated inference. Default: 0 (best value determined automatically).
- `--color_map`: [Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) used to colorize the depth prediction. Default: Spectral. Set to `None` to skip colored depth map generation.
- `--apple_silicon`: Use Apple Silicon MPS acceleration.

### ⬇ Checkpoint cache

By default, the [checkpoint](https://huggingface.co/developy/Celeste) is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally:

```bash
bash script/download_weights.sh Celeste
```

At inference, specify the checkpoint path:

```bash
python run.py \
    --checkpoint checkpoint/Celeste \
    --ensemble_size 10 \
    --denoise_steps: 50 \
    --processing_res: 0 \
    --input_rgb_dir input/example-1 \
    --output_dir output/example-1
```

## 🦿 Evaluation on test datasets

Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
```

Set data directory variable (also needed in evaluation scripts) and download [evaluation datasets](https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset) into corresponding subfolders:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory

wget -r -np -nH --cut-dirs=4 -R "index.html*" -P ${BASE_DATA_DIR} https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/
```

### Evaluate on standard depth mask

Run inference and evaluation scripts, for example:

```bash
# Run inference
bash script/eval/11_infer_nyu.sh

# Evaluate predictions
bash script/eval/12_eval_nyu.sh
```

Or run all dataset test:
```bash
bash script/eval/0_infer_eval_all.sh
```

### Evaluate on our Sky mask

Generate sky masks for the evaluation splits with **Mask2Former**, then run depth inference and evaluation on **VKITTI2**, **KITTI**, or **DIODE**. See the [readme](script/dataset_preprocess/sky_mask/README.md) for the commands.

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

### Fine-tune Checkpoint:
 We support three fine-tune backbone:
  1. Stable Diffusion v2 [checkpoint](https://huggingface.co/sd2-community/stable-diffusion-2)
  2. Segmind Stable Diffusion 1B [checkpoint](https://huggingface.co/segmind/SSD-1B)
  3. Stable-Diffusion-XL-Base-1.0 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

 Download one of them into `${BASE_CKPT_DIR}/`

Prepare for [Hypersim](https://github.com/apple/ml-hypersim) and [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) datasets and save into `${BASE_DATA_DIR}`. Please refer to [this README](script/dataset_preprocess/hypersim/README.md) for Hypersim preprocessing.

```bash
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir "${BASE_DATA_DIR}" \
  --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --output_dir output/ \
  --no_wandb
```

Resume only the matching run:

```bash
python train.py \
  --resume_run /path/to/your/CKPT \
  --base_data_dir "${BASE_DATA_DIR}" \
  --base_ckpt_dir "${BASE_CKPT_DIR}" \
  --no_wandb
```

> [!IMPORTANT]
>
> Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

<!-- ## 🤔 Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (Windows) Invalid DOS bash script on WSL                                                                                                     | Run `dos2unix <script_name>` to convert script format          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` | -->


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
