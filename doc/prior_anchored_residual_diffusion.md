# master 实验：Prior-Anchored Residual Diffusion

2026-10-04。状态：已实现并通过小模型训练测试，尚未运行真实 SD2 + DA2-Giant 训练，
不能宣称已经解决天空塌陷。

## 核心变化

原模型让 DDIM 从噪声直接生成绝对深度 latent：

```text
noise -> z_depth
```

在天空和超量程区域没有有效 GT 时，绝对 latent 可以沿多步轨迹自由漂移。即使 DA2 latent
作为输入条件存在，U-Net 仍然可以忽略它或把它解释成普通纹理条件。

本实验把扩散变量改成相对 DA2 prior 的修正量：

```text
r_gt = (z_gt - stop_grad(z_DA2)) / s
r_T ~ N(0, I)
r_T -> ... -> r_0
z_depth = z_DA2 + s * r_0
```

当前 `s = 1.0`。主 diffusion loss 仍只在有效 GT latent cell 上计算；latent gradient 和原 VGC
作用于组合后的绝对深度 `z_DA2 + r_0`。没有增加 sky mask、天空伪标签或远端主监督。

这样做的目标不是复制 DA2，而是把 DA2 从“可被忽略的输入”提升为显式基准：有可靠 GT 的区域
学习修正 DA2；没有可靠 GT 的天空区域默认围绕零修正建模，并由原 VGC 弱约束区域均值和光滑性。

该方向受到 [Lotus-2](https://arxiv.org/abs/2512.01030) 中“在核心预测器定义的流形内进行受约束
多步 refinement”的启发，但本实现仍是 SD2 DDIM epsilon diffusion，不是 Lotus-2 的单步核心模型
或 rectified flow，不能把论文结论直接当作本项目结果。

## 安全初始化

12 通道顺序保持：

```text
[RGB latent, DA2 latent, noisy residual latent]
```

旧初始化把 SD2 的 4 通道卷积权重复制三份后各除以 3，使 DA2 在第 0 步就与 RGB、扩散状态等权。
新初始化使用：

```text
RGB branch            = 0.5 * pretrained weight
DA2 condition branch  = 0
residual branch       = 0.5 * pretrained weight
```

因此初始激活保持标准 8 通道 Marigold 形式，DA2 条件权重由训练逐渐学出；同时最终深度仍显式加上
DA2 prior。这样可减少同一 prior 在输入和输出两处同时产生强捷径的风险。

## 保留内容

- SD2、DA2-Giant、12 通道条件；
- multi-resolution noise、offset noise、Min-SNR、latent gradient；
- 原始 VGC；
- 23,000 optimizer steps；
- 50 步 DDIM 推理；
- decoder 后训练与本分支无关，未加入此旧分支。

训练器现在显式冻结 DA2 并在 `no_grad` 下生成 prior。每个 checkpoint 除了 `unet/`，还保存
`depth_parameterization.json`；推理必须同时加载两者，不能只复制 U-Net 权重。

## 训练

该实验改变了扩散目标，必须从原始 SD2 开始，不能 resume 绝对深度 VGC checkpoint：

```bash
cd /root/ApDepth-G
python train.py \
  --config config/train_marigold.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/unet_prior_residual_v1 \
  --no_wandb
```

同一次训练中断后恢复：

```bash
python train.py \
  --resume_run /root/ApDepth-G/output/unet_prior_residual_v1/train_marigold/checkpoint/latest \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --no_wandb
```

## 推理

```bash
python run.py \
  --checkpoint /root/Marigold/pretrained_checkpoint/sd2-1 \
  --unet_checkpoint /root/ApDepth-G/output/unet_prior_residual_v1/train_marigold/checkpoint/iter_004000 \
  --input_rgb_dir /root/ApDepth-G/output/out \
  --output_dir /root/ApDepth-G/output/prior_residual_004000 \
  --denoise_steps 50 --ensemble_size 1 --processing_res 768 --seed 2024
```

建议先比较 2k、4k checkpoint。除普通 depth 指标外，应单独测：天空平均深度相对地平线的次序、
天空内部标准差、天空边界光环以及非天空漂移。若 DA2 自身把天空判断为近处，这一参数化可能直接
继承错误；latent 加法也不等于像素深度线性相加。这些是本实验必须通过真实结果回答的风险。
