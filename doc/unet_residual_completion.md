# 主 U-Net 重新训练：噪声感知的先验残差补全

2026-10-02 更新：训练量恢复为 23,000 × 42。独立于 decoder calibration 后训练的新实验；后训练代码、配置和已有结果均保留。
目前只有实现与数值测试，尚未完成真实数据训练，不能宣称已解决天空塌陷。

## 改动依据

原 VGC 的补全目标为“无效区 latent 均值接近 DA2 + 预测 latent 的 TV”。
它只约束区域均值，区域内远近坡度仍有自由度；直接压低 latent 的空间变化，
也不等于解码后的天空深度恒定。冻结 VAE 会产生自身的空间编码结构。

本次把补全项换为预测与先验的**残差均值和带符号梯度**，同时按扩散时间步加权。
令 `r = pred_x0 - stop_grad(z_DA2)`，`s = alpha_bar / (1 - alpha_bar)`：

```text
w(t) = min(s, 5) / 5
L_completion = w(t) * [0.02 * mean_channels(mean_invalid(r)^2)
                       + 0.005 * 0.5 * (mean_edges_x((dx r)^2)
                                      + mean_edges_y((dy r)^2))]
```

每张图先独立计算区域均值、边损失，再对符合原有无效面积门槛 0.08 的图求平均。
不会除以权重之和，否则高噪声衰减会被抵消。两个端点都在无效内部时才使用该边。
每个 8×8 像素块全部无效才进入补全集合；混合块排除。这不是语义天空 mask，
也不是完整 VAE 感受野的隔离，树枝边界仍需观察。

与直接 TV 相比，`pred_x0 == z_DA2` 时新目标严格为零，即使 DA2 latent 本身有空间变化。
与特征关系矩阵对齐相比，带符号梯度会惩罚非恒定先验的方向反转；它并不能单独确定
每个不连通区域的绝对偏移，因此保留区域均值锚点，也不保证解码像素的远近次序。
这里没有增加 affine 拟合、置信度筛选、低通蒸馏、天空通道或额外去噪轨迹。

本方案是本项目的待验证设计，不声称论文级原创性。时间步权重依据
[Min-SNR 论文](https://arxiv.org/abs/2303.09556)，主损失参数化参照
[Diffusers 官方训练实现](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)。
论文没有验证这里的天空补全项。

## 同时修复的训练问题

- 主扩散 MSE 原来无论预测类型都乘 `min(s,5)/s`。现在 epsilon 使用该公式，
  v_prediction 使用 `min(s,5)/(s+1)`，sample 使用 `min(s,5)`。
  三者换算到 x0 空间都是 `min(s,5) * MSE(x0_pred,x0)`。
  使用多分辨率/offset 噪声时，s 是调度器定义的名义 SNR，并非空间相关噪声的实测 SNR。
- 有效区梯度损失原来仅检查边的一端，现在要求两端有效，空掩码返回可反传的零。
  原有效区梯度损失的幅值形式及系数 0.1 保留。
- DA2 显式冻结并设为 eval，生成先验使用 no_grad。
- 日志分开记录 diffusion_loss、latent_gradient_loss、completion_loss、
  completion_anchor、completion_gradient、completion_coverage、completion_snr_weight。
  coverage 是 latent 无效内部占比，不是天空识别率。

这些是可以确认的实现差异，尚不能据此确定旧模型天空塌陷的唯一原因。
`min_snr_gamma` 的换算针对默认 MSE，非 MSE 损失不能套用上述等价证明。

## 训练量

主训练仍保留 DA2-Giant 先验、RGB/prior/noisy-depth 的 12 通道输入、冻结 VAE 编解码、
多分辨率退火噪声（strength 0.9）、通道 offset noise（0.1）、Min-SNR 主扩散损失、
latent 梯度一致性（0.1）和多步 DDIM 推理。旧 VGC 仍可通过 `mode: legacy` 启用，
默认 `mode: residual_snr` 启用本次改进；两者是可切换实现，不同时叠加。
decoder calibration 后训练保持独立。此前明确失败并移除的实验仍记录在 failed.md，未重新启用。

| 项目 | 原配置 | 本次配置 |
|---|---:|---:|
| effective batch | 32 | 42 |
| 单次 batch / 梯度累积 | — | 7 / 6 |
| optimizer steps | 23,000 | 23,000 |
| 样本呈现次数 | 736,000 | 966,000 |
| LR 调度 total_iter | 25,000 | 25,000 |
| warmup | 100 | 100 |
| latest 保存间隔 | 250 | 250 |
| 独立备份间隔 | 2,000 | 2,000 |

学习率仍为 3e-5。按用户决定撤回按 batch 缩短步数的设置，恢复原先按优化器步数定义的
学习率调度和保存间隔。更新次数仍为 23,000，样本呈现次数较旧 32 batch 增加 31.25%。
当前 train.py 注释掉了验证/可视化 loader 的创建，所以相应周期显式设为 0，
不能把训练日志当成室内不退化的证明。

## 从 SD2 开始新训练

项目目录中应已有原来使用的 DA2-Giant 代码和 `DA2/checkpoints/depth_anything_v2_vitg.pth`。
SD2 完整目录位于 `/root/Marigold/pretrained_checkpoint/sd2-1`，数据位于 `/root/Dataset`。
本次无需天空标签，也无需 decoder 后训练缓存。

```bash
cd /root/ApDepth-G
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/unet_residual_snr_v1 \
  --no_wandb
```

不传旧 VGC checkpoint，也不传 `--resume_run`。训练器从 SD2 的 4 通道卷积扩展为 12 通道，
同时训练完整 U-Net；VAE、DA2、文本编码器冻结。仍然是常规扩散训练和多步 DDIM 推理。

实际运行目录是 `output/unet_residual_snr_v1/train_marigold/`。
新训练中断后，可继续**该次新实验**：

```bash
python train.py \
  --resume_run /root/ApDepth-G/output/unet_residual_snr_v1/train_marigold/checkpoint/latest \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --no_wandb
```

恢复读取运行目录内保存的 config.yaml，修改根配置不会改变恢复任务。
完成后的最终权重在 `checkpoint/iter_023000/unet/`；`latest` 按 250 步保存，
因此正常完成时 latest 也对应第 23,000 步，并包含恢复训练所需的状态。

## 检查新 U-Net，再决定是否做后训练

```bash
python run.py \
  --checkpoint /root/Marigold/pretrained_checkpoint/sd2-1 \
  --unet_checkpoint /root/ApDepth-G/output/unet_residual_snr_v1/train_marigold/checkpoint/iter_023000 \
  --input_rgb_dir /root/ApDepth-G/output/out \
  --output_dir /root/ApDepth-G/output/unet_residual_snr_eval \
  --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
```

先使用原 VAE 比较新旧 U-Net，固定图片、分辨率、50 步及多个相同 seed，检查天空、
树枝边界和室内图。可从第 2,000 / 4,000 步的独立备份提前检查，勿凭训练总 loss 下降判断成功。
如果 DA2 自身天空先验错误，新方法可能继承错误；它没有绝对语义天空监督，
并不能承诺无限远或彻底消除斑点。共享 U-Net 参数也可能影响室内结果。

为了单独评估新补全项，可复制本配置，将 `validity_guided_completion.mode` 设为 `legacy`、
添加 `smooth_weight: 0.005`，在另一输出根目录训练相同步数；保留其余训练修复。
该比较才隔离新补全项的效果，旧 23,000 步 checkpoint 同时含有 batch 和实现差异。

现有 decoder calibration 仍可继续用于旧 VGC U-Net。新 U-Net 选定后，若要运行该后训练，
必须用新 U-Net **重新缓存完整多步最终 latent 并重新校准 decoder**，详见
[decoder_calibration.md](decoder_calibration.md)。旧缓存不能代表新 U-Net 的采样分布。
