# nightly：SDXL Base 多步深度估计 demo

2026-10-02。状态：真实小型 SDXL 组件的 CPU 集成测试通过；尚未在用户的
RTX 6000 Ada 上测试完整权重、BF16 或 bitsandbytes CUDA 内核，也没有深度效果结论。
本次是独立底座实验，main 的 SD2 配置及已提交权重不受影响。

## 显存判断

用户提供的 nvidia-smi 显示总量 46,068 MiB（44.99 GiB），当前 SD2 训练占用
38,073 MiB（37.18 GiB），余量 7.81 GiB。不能与当前训练同时启动 SDXL。

按官方 SDXL U-Net 配置在 meta device 实例化、将输入扩展为 12 通道后，参数量为
2,567,486,724。以下只是静态状态估算，不是峰值显存测量：

| U-Net 方案 | 每参数粗估 | 静态显存 |
|---|---:|---:|
| FP32 权重、梯度、Adam 两个 FP32 状态 | 16 bytes | 38.26 GiB |
| FP32 权重、梯度、两个 8-bit Adam 状态 | 约 10 bytes | 约 23.91 GiB |

还需加上冻结 DA2-Giant、FP32 VAE、量化元数据、激活、BF16 权重转换缓存、
优化器临时空间、CUDA context 和 allocator 缓存。bitsandbytes 的小张量也可能保留 FP32 状态。
因此原样使用 FP32 Adam 很紧；8-bit Adam 配合 batch 1 和梯度检查点有合理的适配空间，
值得做 demo，但这里不承诺具体峰值或必定不 OOM。BF16 autocast 不会把 FP32 主权重和梯度减半。

本实现参考 [Diffusers SDXL 训练说明](https://huggingface.co/docs/diffusers/training/sdxl)
与 [bitsandbytes 8-bit optimizer](https://huggingface.co/docs/bitsandbytes/optimizers)。
原始模型配置见 [SDXL U-Net](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/unet/config.json)。

## 本版行为

- 使用 `stabilityai/stable-diffusion-xl-base-1.0`，不使用 refiner；训练完整 U-Net。
- RGB / DA2 prior / noisy depth 仍为 4+4+4 通道。多分辨率退火噪声、offset noise、
  Min-SNR、latent 梯度继续保留；补全目标显式使用原始 VGC（`mode: legacy`），
  SDXL 训练不启用 `residual_snr`。
- 补齐 SDXL 双编码器的倒数第二层 hidden states、第二编码器的 pooled embedding 和尺寸条件。
  两个文本编码器在 CPU 编码空提示后释放，只保存空提示 embedding。
- 尺寸条件统一使用实际输入画布 `[H,W,0,0,H,W]`，训练与推理一致。
- VAE 缩放从其配置读取（官方值 0.13025）；VAE 始终使用 FP32，DA2 也保持原 FP32。
- FP32 U-Net 主权重，BF16 autocast 前向，梯度检查点，PyTorch SDPA，8-bit Adam。
  损失在 FP32 中计算；BF16 不使用 FP16 GradScaler。
- 主配置为 23,000 次更新，单次 batch 1、累积 42 次、effective batch 42。
  demo 学习率取 1e-5；LR 调度 25,000 / warmup 100。保留原始两类训练图像尺寸。
- 原 checkpoint 的 alpha/noise schedule 用于训练；推理显式转换为 DDIM、关闭 latent clipping，
  默认 50 步，仍由纯噪声开始。没有改成单步回归。

## 获取分支和依赖

先正常保存并结束当前训练，再切换代码和环境，避免运行中的任务遇到依赖变化。
如果 main 有本地修改，先保留它们；下面命令不会强制清理工作区。

```bash
cd /root/ApDepth-G
git fetch origin
git switch --track origin/nightly  # 首次；已有本地 nightly 时使用 git switch nightly
git pull --ff-only origin nightly
python -m pip install -r requirements-sdxl.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
python -m bitsandbytes
```

`nvidia-smi` 的 CUDA 13.0 表示驱动能力，不代表当前 PyTorch 的 CUDA runtime 版本。
新增依赖文件不主动指定新的 torch 版本。BF16 检查应返回 True，bitsandbytes 应能识别 CUDA。

下载 SDXL 的组件权重（无需下载 refiner 或仓库根目录的重复单文件权重）：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="/root/Marigold/pretrained_checkpoint/stable-diffusion-xl-base-1.0",
    allow_patterns=[
        "model_index.json", "*/config.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors", "text_encoder_2/model.safetensors",
    ],
)
PY
```

沿用项目原有 DA2-Giant 代码和 `DA2/checkpoints/depth_anything_v2_vitg.pth`。

## 先做实际显存探测

```bash
python train.py \
  --config config/train_sdxl_probe.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/sdxl_probe_v1 \
  --no_wandb
```

探测固定取 Hypersim 和 VKITTI 各一张真实训练样本，交替运行两个优化器更新，
每次累积两张。第二次更新时优化器状态已经分配；数据按原来的尺寸处理。
这里只把累积次数从 42 减到 2，单次 batch 与正式训练一样，验证梯度累积和状态常驻时的峰值。
它不代表所有图像／尺寸的最坏情况，也不保存训练权重。失败会返回非零退出码，不会伪装完成。

结果文件：

```text
output/sdxl_probe_v1/train_sdxl_probe/memory_profile.json
```

必须确认命令成功退出、文件中 `effective_iter` 为 **2**，再检查：

- `peak_allocated_gib`：PyTorch 实际分配峰值。
- `peak_reserved_gib`：allocator 预留峰值，包含缓存。
- `device_free_gib`：记录时设备空闲量，不是峰值时的最小空闲量。

可同时观察 nvidia-smi。建议留出几 GiB 余量，实际数值接近设备上限时先调整尺寸再训练。
只有完成第一步的 JSON 不能说明探测通过。重跑时使用新的输出根目录，例如 `sdxl_probe_v2`。

## 完整训练

探测通过后另起新训练；不加载 SD2 checkpoint，不从探测目录 resume：

```bash
python train.py \
  --config config/train_sdxl_demo.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/sdxl_demo_v1 \
  --no_wandb
```

最终 checkpoint：`output/sdxl_demo_v1/train_sdxl_demo/checkpoint/iter_023000/`。
中断恢复使用同目录下的 `checkpoint/latest`：

```bash
python train.py \
  --resume_run /root/ApDepth-G/output/sdxl_demo_v1/train_sdxl_demo/checkpoint/latest \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --no_wandb
```

独立备份每 2,000 步，latest 每 250 步。完整 FP32 U-Net 单份权重约 9.56 GiB，
latest 还包含优化器状态；保留所有备份需要约 130 GiB 的输出磁盘空间，另计基础模型与数据。
梯度检查点和较多累积次数以计算时间换显存，不保证比 SD2 更快。

## 50 步推理

```bash
python run.py \
  --backbone sdxl \
  --checkpoint /root/Marigold/pretrained_checkpoint/stable-diffusion-xl-base-1.0 \
  --unet_checkpoint /root/ApDepth-G/output/sdxl_demo_v1/train_sdxl_demo/checkpoint/iter_023000 \
  --input_rgb_dir /root/ApDepth-G/output/out \
  --output_dir /root/ApDepth-G/output/sdxl_demo_eval \
  --denoise_steps 50 --ensemble_size 1 --batch_size 1 --processing_res 768 --seed 2024
```

早期检查可以用 `iter_002000` 等备份。默认推理为 FP32；在单次 batch 1 下没有训练梯度和
优化器状态，预算比训练宽裕。这里下载的是普通 safetensors 文件，不含 fp16 variant 文件，
上述命令不要额外添加 `--half_precision`。

`infer.py`、`script.trace_depth_denoising`、`script.cache_decoder_latents` 也支持
`--backbone sdxl`。decoder 后训练代码和配置继续保留，等主 U-Net 训练完成后再决定是否使用；
使用时必须从新的 SDXL U-Net 重新生成缓存，
它会记录匹配的 VAE 和 latent scale。SD2 的 decoder 权重或缓存不能用于 SDXL。
缓存、校准与恢复流程见 [decoder_calibration.md](decoder_calibration.md)。

## 验证范围与效果评估

CPU 测试使用真实小型 SDXL U-Net（含 text_time）、双 CLIP 和 8x VAE，覆盖本地组件加载、
4→12 通道训练、激活检查点反传、checkpoint 保存与实际恢复更新、推理权重覆盖、
VAE 缩放、尺寸条件、固定 seed 和完整多步采样。它们不能代替完整 SDXL/DA2 的 GPU 验证。

先观察 GPU 探测，再用固定图片及同一 seed 比较天空、细枝和室内场景；换底座本身不能保证
解决天空塌陷。当前 train.py 的验证 loader 仍未启用，应独立跑验证，不能仅看训练 loss。
因为底座、学习率和优化器都变化，这个 demo 不是严格隔离模型容量影响的消融实验。
