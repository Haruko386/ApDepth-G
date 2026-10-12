# 主 U-Net 新实验：Terminal-SNR Alignment

2026-10-04。该方案从 SD2 初始化完整主训练，保留 50 步 DDIM，不是后训练。
目前只有实现和数值测试，尚无真实数据效果，不能称为已经解决天空塌陷。

## 为什么换到扩散端点

`residual_snr` 以及此前多种 sky / far-field 辅助监督均未得到稳定收益，部分结果反而出现
天空斑点和远近反向。因此本轮不再定义“哪里是天空”或“天空应当是多少深度”，只处理训练与
推理的扩散端点是否一致。

原 SD2 schedule 在最后一个训练 timestep 仍保留少量 clean latent 信号，而推理从纯高斯噪声
开始；默认 `leading` DDIM 子序列也未必从最后一个训练 timestep 起步。模型训练时可能依靠
这部分残余的全局低频信息，推理第一步却得不到它。天空是面积大、纹理弱、主要依赖全局结构的
区域，因此可能对这种错配敏感。这是实验假设，不是已证明的天空塌陷根因。

该问题及通用修复来自
[Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891)：
将末端 SNR 重标定为 0、使用 v-prediction，并让采样从末端 timestep 开始。
[CVPR 2025 的 diffusion perception scaling 工作](https://openaccess.thecvf.com/content/CVPR2025/html/Ravishankar_Scaling_Properties_of_Diffusion_Models_For_Perceptual_Tasks_CVPR_2025_paper.html)
也报告了多步迭代和早期去噪计算对感知任务的价值，但两篇论文都没有验证本项目的天空问题。

## 实现

`config/train_marigold.yaml` 现在启用：

```yaml
diffusion_schedule:
  prediction_type: v_prediction
  rescale_betas_zero_snr: true
  timestep_spacing: trailing
  min_snr_floor: 0.05
  terminal_noise_fade_fraction: 0.10
```

- `v_prediction` 避免在 `alpha_bar=0` 时通过除以 `sqrt(alpha_bar)` 恢复 `x0`。
- zero-terminal-SNR 使最后一个训练状态成为真正的纯噪声端点。
- `trailing` 使 50 步 DDIM 的第一步使用最后一个训练 timestep。
- 标准 v-prediction Min-SNR 权重在 SNR=0 时也是 0。`min_snr_floor=0.05` 让高噪声端保留
  很小的训练权重；相对 `gamma=5` 的最大 x0 权重为 1%。它不是天空权重。
- 原多分辨率噪声和 offset noise 在最高 timestep 仍会造成训练—推理分布差异。它们在前 90%
  timestep 保持原设置，在最后 10% 线性衰减，并在末端强制换成标准高斯；推理仍从标准高斯开始。
- checkpoint 在 `unet/` 旁保存匹配的 `scheduler/`。`run.py`、`infer.py`、轨迹记录和 decoder
  cache 加载 U-Net 时会同步加载该 scheduler，避免拿新 U-Net 配旧 SD2 epsilon schedule。

仍保留：

- RGB + DA2 prior + noisy depth 的 12 通道输入；
- 多分辨率退火噪声和 channel offset noise；
- Min-SNR、有效区域 latent gradient；
- 原 VGC 的弱区域均值锚点和 invalid-region smoothness；
- 23,000 optimizer steps、effective batch 42、完整多步推理。

明确移除：`residual_snr`、额外 sky mask、远端伪标签、扩大 GT mask、语义关系以及新增的
invalid-region loss。decoder calibration 后训练代码仍保留，但本轮主训练不执行它。

## 训练

必须从原始 SD2 开始。prediction target 和 noise schedule 都已改变，不能 resume
`residual_snr`、语义关系或旧 VGC 的 trainer state。

```bash
cd /root/ApDepth-G
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/unet_terminal_snr_v1 \
  --no_wandb
```

中断后只能恢复同一次新实验：

```bash
python train.py \
  --resume_run /root/ApDepth-G/output/unet_terminal_snr_v1/train_marigold/checkpoint/latest \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --no_wandb
```

## 50 步推理

必须传 checkpoint 的父目录，使加载器同时找到 `unet/` 和 `scheduler/`：

```bash
python run.py \
  --checkpoint /root/Marigold/pretrained_checkpoint/sd2-1 \
  --unet_checkpoint /root/ApDepth-G/output/unet_terminal_snr_v1/train_marigold/checkpoint/iter_023000 \
  --input_rgb_dir /root/ApDepth-G/output/out \
  --output_dir /root/ApDepth-G/output/unet_terminal_snr_eval \
  --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
```

不要只复制 `unet/` 到旧 pipeline；那会丢失 scheduler。建议在 2,000、4,000 步先用固定图片、
固定 seed 检查天空顺序、天空内部方差、树枝边界和 NYUv2，最终再判断是否继续到 23,000 步。
训练总 loss 因 epsilon → v prediction 已改变，数值不能与旧日志直接比较。

如果本方案仍失败，应先用 `script.trace_depth_denoising` 对比首个 timestep 和前几步的低频漂移，
不要继续追加天空监督。后训练保留待用；若以后启用 decoder calibration，必须使用这个 checkpoint
重新生成完整多步 cache，旧 schedule 的 cache 不可复用。
