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

### Mask2Former 天空分割：VKITTI2 / KITTI / DIODE

使用 [script/segment_sky.py](script/segment_sky.py) 为评估 RGB 生成天空二值掩码。
脚本从 `--base_data_dir` 或环境变量 `BASE_DATA_DIR` 读取数据，支持已解压目录和 `.tar`，
不需要 depth GT、官方天空标签或预先解压归档。配置见
[config/sky_segmentation.yaml](config/sky_segmentation.yaml)。

通过必填的 `--data_split` 传入已有名单，**分割和后续 eval 使用同一份 split**。
名单第一列为相对数据集 RGB 根目录的图片路径，其余 GT 路径、焦距等列原样保留；支持空行和 `#` 注释。
不会改写你的 split，也不会自动换回 dataset config 中的默认名单。
以下是项目已有划分与默认 RGB 寻址配置，可直接传入相应名单：

| 名称 | 默认配置 / 划分 | 默认数据位置（相对 BASE_DATA_DIR） |
|---|---|---|
| `vkitti2`（也接受 `vkitti`） | `data_vkitti_val.yaml`，仓库 Scene01 验证集 | `vkitti/vkitti.tar`，或 `vkitti/vkitti_rgb` / `vkitti` |
| `kitti` | `data_kitti_eigen_test.yaml`，Eigen test | `kitti/kitti_eigen_split_test.tar`，或 `kitti/kitti_eigen_split_test` / `kitti` |
| `diode` | `data_diode_all.yaml`，发布的全部验证集，包含 indoors/outdoors | `diode/diode_val.tar`，或 `diode/diode_val` / `diode` |

**VKITTI 小规模评估：** 使用 `data_split/vkitti/vkitti_val_sampled_540.txt`，
仅从原 5,364 张 `vkitti_val.txt` 抽取 540 张，不引入 train。
按 6 种 variation × 2 个 camera 均衡取样，每组 45 张，覆盖原序列不同时间位置。
使用时将该路径传给 `--data_split`，并选择新的 output_dir 运行名；原 train/val 均保留不变。
详细抽样规则、配置和交集审计见 [VKITTI split 说明](data_split/vkitti/README.md)。

例如 `BASE_DATA_DIR=/data` 时，VKITTI 解压后的第一张 RGB 应位于
`/data/vkitti/vkitti_rgb/rgb/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg`。
目录结构不同时，在分割配置的 `datasets.<name>.data_dir` 中填写相对 `BASE_DATA_DIR` 的目录或归档路径；
该路径下必须能直接找到名单第一列所写的相对路径。指定 `data_dir` 后不再自动回退。
数据读取按原配置路径、列出的解压目录依次尝试；不会扫描训练集扩大样本范围。
例如 DIODE 只需要 outdoors 时，直接传
`--dataset diode --data_split data_split/diode/diode_val_outdoor_filename_list.txt`。
`--dataset` 与 `--datasets` 等价；标准 `data_split/kitti/...` 等路径可以省略 dataset 参数自动识别。
自定义位置的名单请明确传 dataset。批量处理时，每个 dataset 对应一个 split，按参数顺序配对。

**本地模型准备。** 使用 Transformers 格式的完整 Mask2Former 模型目录，包含 `config.json`、
`preprocessor_config.json` 和权重文件。三个数据集应固定使用同一个分割 checkpoint。
可选用 [facebook/mask2former-swin-large-ade-semantic](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic)
作为统一的 ADE20K 语义分割模型；这是覆盖不同场景的起点，实际天空标签质量仍需人工检查。
运行环境需要项目已有的 PyTorch / Transformers，以及 NumPy、Pillow、PyYAML、SciPy。
脚本只接受**已存在的本地模型目录**，强制离线加载，不会自动下载模型或数据集。

先检查 RGB 路径（不加载模型、不写输出）：

```bash
export BASE_DATA_DIR=/path/to/datasets
python -B script/segment_sky.py \
    --base_data_dir "$BASE_DATA_DIR" \
    --dataset kitti \
    --data_split data_split/kitti/eigen_test_files_with_gt.txt \
    --dry-run
```

生成全部天空掩码：

```bash
python -B script/segment_sky.py \
    --config config/sky_segmentation.yaml \
    --base_data_dir "$BASE_DATA_DIR" \
    --checkpoint /path/to/local/mask2former-swin-large-ade-semantic \
    --datasets vkitti2 kitti diode \
    --data_split data_split/vkitti/vkitti_val.txt \
                 data_split/kitti/eigen_test_files_with_gt.txt \
                 data_split/diode/diode_val_all_filename_list.txt \
    --device cuda \
    --output_dir output/sky_segmentation/mask2former_v1
```

Windows PowerShell 使用同一入口，例如只处理本地 VKITTI：

```powershell
$env:BASE_DATA_DIR = 'F:\'
python -B script/segment_sky.py --dataset vkitti2 --data_split data_split/vkitti/vkitti_val.txt --dry-run
python -B script/segment_sky.py --dataset vkitti2 `
    --data_split data_split/vkitti/vkitti_val.txt `
    --checkpoint 'D:\models\mask2former-swin-large-ade-semantic' `
    --device cuda --output_dir output/sky_segmentation/vkitti_mask2former_v1
```

`--device cpu` 可用于 CPU；默认 `auto` 优先 CUDA。
`--limit-samples 5` 可以生成每个数据集前 5 张的预览，但结果会明确标记为 limited。
实际分割始终逐张处理，使用 checkpoint 自带 processor，并通过官方
[`post_process_semantic_segmentation(..., target_sizes=[(H,W)])`](https://huggingface.co/docs/transformers/en/model_doc/mask2former#transformers.Mask2FormerImageProcessor.post_process_semantic_segmentation)
在原 RGB 尺寸上得到类别图。从模型 `id2label` 查找 `sky`，不硬编码 ADE20K/Cityscapes 类别编号。
输出 `255=sky`、`0=non-sky`；不做后续膨胀、填洞、深度有效性筛选或额外裁剪。
全黑掩码也是有效输出；无天空样本仍保留在生成名单中。

**掩码直接保存到数据集文件夹下**，按 RGB 相对路径组织，运行名取 `--output_dir` 的最后一级目录名。
目录数据写到 `<RGB根目录>/sky_masks/<运行名>/`；tar 数据写到 `<tar所在目录>/sky_masks/<运行名>/`。
例如 KITTI 使用 `BASE_DATA_DIR/kitti/kitti_eigen_split_test.tar` 时：

```text
<BASE_DATA_DIR>/kitti/sky_masks/mask2former_v1/
  <RGB相对路径>.sky.png
```

`--output_dir` 只存放运行记录、索引、预览和后续 eval 配置，配置中自动引用数据集内的掩码：

```text
output/sky_segmentation/mask2former_v1/
  run_metadata.json               # checkpoint/processor/权重哈希、环境、分割规则
  summary.json                    # 各数据集成功、失败、无天空、>=5% 天空数量
  sky_eval.generated.yaml         # 仅在所有选中图像分割成功后导出
  vkitti2/                        # kitti/、diode/ 同样组织
    manifest.jsonl                # sample_id、RGB/mask 路径、尺寸、哈希、伪标签来源
    per_image.jsonl               # 每个输入的状态和失败原因
    segmentation_report.json
    previews/                     # 最多 50 张均匀选取的 RGB+天空叠加图
    depth_inference.yaml          # 复用 infer.py 的完整图像配置
    depth_inference_files.txt      # 此次选中的原始名单行
```

运行输出目录和数据集内的掩码运行目录必须是新目录；同一个运行名不覆盖旧掩码。
换 split 或分割模型时请同时使用新的 `--output_dir` 运行名。
退出码 `0` 表示完成，`2` 表示错误/不完整；
发生逐图失败时记录原因，不导出可用于评估的生成配置。修正问题后使用新的输出目录重新运行。

**接入天空深度评估。** 这些掩码来自 **Mask2Former 伪语义标签**，与官方 sky GT 的结果应区分。
先检查叠加图，再使用生成的 `sky_eval.generated.yaml`。每个深度模型共用同一套冻结的分割结果。
`eval.py` 的常规 GT 深度指标仍使用原协议；天空一致性和排序使用独立的 `script/sky_eval.py`。

KITTI/VKITTI 原 benchmark 推理会裁剪，旧裁剪预测不能和完整天空图直接配对。
生成的 `depth_inference.yaml` 关闭 benchmark crop，标准名单的 `filenames` 直接指向你传入的原始 `--data_split`，
SCOE 配置中的 `data_split` 也指向同一文件。任何 split 修改都会被检查出来，不能悄悄变更评估集合。
使用 `--limit-samples` 时，推理配置引用对应前 N 张的名单快照，并将结果标记为 limited。
名单含空行/注释时，为兼容原 `infer.py`，推理配置引用移除这些非数据行的快照，图像集合和顺序保持一致。
要复用深度推理，split 的其余列仍须满足原 dataset loader 的格式；仅生成天空掩码时只读第一列。
可用原入口重新生成完整图预测：

```bash
# BASE_CKPT_DIR 和 CELESTE_CHECKPOINT 必须指向现有本地深度模型及匹配的 scheduler/config。
for dataset in vkitti2 kitti diode; do
  python infer.py \
      --checkpoint "$BASE_CKPT_DIR/sd2-1" \
      --unet_checkpoint "$CELESTE_CHECKPOINT" \
      --base_data_dir "$BASE_DATA_DIR" \
      --dataset_config "output/sky_segmentation/mask2former_v1/$dataset/depth_inference.yaml" \
      --output_dir "output/sky_predictions/celeste_full/$dataset" \
      --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
done
```

在 `sky_eval.generated.yaml` 的 `models` 中填写实际深度模型记录，可参考
[config/sky_eval_mask2former_models.example.yaml](config/sky_eval_mask2former_models.example.yaml)。
`layout: infer_split` 自动读取 `<root>/<dataset>/<RGB相对父目录>/pred_<RGB stem>.npy`。
先填写并核实 checkpoint、输出表示、方向和推理元数据，再执行 audit；之后不修改该配置。

```bash
python -B script/sky_eval.py audit --config output/sky_segmentation/mask2former_v1/sky_eval.generated.yaml
# audit 对伪标签诊断协议返回 2 并记录原因；检查无数据失败后，继续下面各步。
python -B script/sky_eval.py prepare --config output/sky_segmentation/mask2former_v1/sky_eval.generated.yaml
python -B script/sky_eval.py evaluate --config output/sky_segmentation/mask2former_v1/sky_eval.generated.yaml --model celeste_full
python -B script/sky_eval.py report --config output/sky_segmentation/mask2former_v1/sky_eval.generated.yaml
```

诊断报告分别导出 VKITTI2 / KITTI / DIODE 的指标和有效样本数；5% 天空筛选在 audit 阶段完成。
`--formal` 对这套伪标签协议保持阻断，避免混同原先 HRWSI+官方 VKITTI 标签协议的主表。
完整指标定义、失败规则与模型表示要求见 [SCOE 指南](doc/sky_eval.md)。

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

The fifteen YAML files in `config/ablation/` map to Tables III and IV for manual
experiments. N2 (`N2_no_terminal_noise_fade.yaml`) sets
`diffusion_schedule.terminal_noise_fade_enabled=false`: structured noise no
longer fades near the endpoint, while exact final-step Gaussian matching,
zero-terminal-SNR and trailing DDIM remain enabled. When this flag is omitted,
fading follows `terminal_noise_alignment`, preserving the Full and ID 03 settings.
The two 8-channel configs require manual architecture changes; the table runners
and automatic evaluation remain removed.

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
