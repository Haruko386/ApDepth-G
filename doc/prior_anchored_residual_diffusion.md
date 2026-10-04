# master 实验：Masked Prior-Residual v-Diffusion

日期：2026-10-04。状态：实现完成，尚未进行真实 SD2 + DA2-Giant 训练，不能提前宣称已解决天空塌陷。

## VGC 的改动

VGC 把无效深度区域近似成天空/远场，并在该区域施加 DA2 区域均值锚点和平滑损失。此前多个失败实验都出现了相同规律：对 sky、far-field 或 invalid region 增加直接监督后，天空更容易出现斑点、错误坡度或远近反转。

`invalid depth` 还包括越界深度、遮挡、传感器空洞等内容，并不等价于天空。因此本实验保留 validity-guided 的思想，但把 VGC 从额外监督改成 `target_censoring`：

- 不计算 invalid-region anchor；
- 不计算 invalid-region smoothness；
- 不使用 sky mask 或天空伪标签；
- 不扩大有效 GT 范围。

VGC 现在只负责在 VAE 编码前清理无效深度值，并隔离可能受无效值污染的边界 latent，不再要求 U-Net 在无效区域拟合某个伪目标。

## 新的扩散变量

模型仍使用 DA2-Giant，但扩散过程生成相对于冻结 DA2 的修正量：

```text
r_gt = (z_gt - stop_grad(z_DA2)) / s
r_T ~ N(0, I)
r_T -> ... -> r_0
z_depth = z_DA2 + s * r_0
```

当前 `s=1.0`。DA2 因而是最终预测的显式基准，不再只是一个可能被 U-Net 忽略的输入条件。

## 无效 GT 的处理

旧数据流程会把无效原始深度归一化成 `-1`，对应近端。之后即使在 latent loss 上屏蔽无效 cell，VAE 编码器的感受野仍可能把这个错误近端值传播到天空边界附近的有效 latent。

新流程在 VAE 编码前执行：

```text
d_encode = valid * d_gt + (1 - valid) * stop_grad(d_DA2)
```

这一步只为消除编码污染。无效位置依然不进入 diffusion loss 或 gradient loss。像素 mask 映射到 latent 后，再向有效区域内部收缩 2 个 latent cell，避免边界混合 latent 参与监督。

## 统一 v-prediction

本分支没有 epsilon 训练路径：

- U-Net 只预测 velocity；
- 训练 scheduler 使用 zero terminal SNR；
- 50 步 DDIM 使用同一份 v-prediction scheduler 和 `trailing` timestep；
- checkpoint 必须包含 `scheduler/`，推理缺失时直接报错；
- Min-SNR 使用 v-prediction 对应权重，纯噪声端点保留 `0.05` 权重。

## 原有训练项的保留与适配

Multi-resolution noise、channel-wise offset noise 和 latent gradient loss 均保留。为适配 zero-terminal-SNR，前两种结构化噪声在最后 10% timestep 逐渐衰减，并在最后一个 timestep 完全切换成标准高斯，使多步推理起点与训练端点一致。

Latent gradient loss 仍只作用于有效区域；其边掩码改为要求梯度两端都有效，避免跨越天空/无效区边界计算梯度。

## 保留内容

- SD2 与冻结的 DA2-Giant；
- 12 通道 `[RGB latent, DA2 latent, noisy residual latent]`；
- 有效 GT 上的 Min-SNR v-prediction MSE；
- multi-resolution noise、offset noise 和有效区域 latent gradient；
- 23,000 optimizer steps；
- 50 步 DDIM 推理。

12 通道首层初始化为：

```text
RGB branch            = 0.5 * pretrained weight
DA2 condition branch  = 0
residual branch       = 0.5 * pretrained weight
```

## 训练

扩散变量和 prediction type 都已改变，必须从原始 SD2 开始，不能加载旧 VGC、epsilon 或上一版 residual checkpoint：

```bash
cd /root/ApDepth-G
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/unet_masked_residual_vpred_v2 \
  --no_wandb
```

同一次实验中断后可以恢复：

```bash
python train.py \
  --resume_run /root/ApDepth-G/output/unet_masked_residual_vpred_v2/train_marigold/checkpoint/latest \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --no_wandb
```

## 推理

`--unet_checkpoint` 必须指向同时包含 `unet/`、`scheduler/` 和 `depth_parameterization.json` 的 checkpoint 根目录：

```bash
python run.py \
  --checkpoint /root/Marigold/pretrained_checkpoint/sd2-1 \
  --unet_checkpoint /root/ApDepth-G/output/unet_masked_residual_vpred_v2/train_marigold/checkpoint/iter_004000 \
  --input_rgb_dir /root/ApDepth-G/output/out \
  --output_dir /root/ApDepth-G/output/masked_residual_vpred_004000 \
  --denoise_steps 50 \
  --ensemble_size 1 \
  --processing_res 768 \
  --seed 2024
```

建议先比较 2k 和 4k checkpoint，并固定输入、seed、分辨率。重点检查天空—地平线次序、天空内部方差、边界光环、非天空远景结构和 NYUv2 非退化。

## 已知风险

- 若 DA2 本身在天空区域判断错误，显式 prior 基准可能继承该错误；
- latent 加法不等价于像素深度线性相加；
- 收缩监督 mask 会减少天空边界附近的训练样本；
- 这是一项待验证实验，真实效果必须由 checkpoint 对比决定。
