# ApDepth-G 主训练消融实验

本目录中的配置都继承 `config/train_marigold.yaml`。除被消融的组件外，数据、随机种子、学习率、有效 batch size（42）和训练步数（23000）完全相同。所有实验继续使用 `v_prediction`；终端 SNR 消融也不会退回 epsilon prediction。

DA2 的输入归一化和输出方向/范围修复属于接口正确性修复，不是可选创新点。所有实验都必须使用修复后的 `DA2/depth_anything_v2/dpt.py`，不能用错误的 DA2 输出做消融，否则实验比较无效。

## 实验列表

| 配置 | 仅改变的内容 | 用途 |
|---|---|---|
| `00_full.yaml` | 无 | 完整方法对照组 |
| `01_no_far_anchor.yaml` | `far_anchor_weight=0` | 验证修正 DA2 的远场弱锚点 |
| `02_no_vgc.yaml` | 关闭全部 VGC | 验证无效深度区域监督整体作用 |
| `03_no_terminal_snr.yaml` | 关闭 zero-terminal-SNR、trailing spacing、SNR floor 和终端噪声对齐 | 验证终端噪声分布对齐；仍使用 v-prediction |
| `04_no_min_snr.yaml` | 关闭 Min-SNR 重加权 | 验证时间步损失重加权 |
| `05_no_multi_res_noise.yaml` | 关闭多分辨率噪声 | 验证低频噪声增强 |
| `06_no_offset_noise.yaml` | offset noise 强度设为 0 | 验证逐通道噪声偏移 |
| `07_no_latent_gradient.yaml` | latent gradient 权重设为 0 | 验证潜空间梯度一致性 |
| `08_no_da2_prior.yaml` | DA2 条件置零并关闭 VGC | 验证全部 DA2 信息的整体贡献 |

其中 `01_no_far_anchor` 与 `02_no_vgc` 可以组成递进比较：

- `00_full - 01_no_far_anchor`：远场弱锚点的贡献。
- `01_no_far_anchor - 02_no_vgc`：原始 VGC 区域均值与平滑约束的贡献。

`08_no_da2_prior` 是组合消融，因为 VGC 的教师信号来自 DA2。关闭 DA2 条件后仍保留 VGC 会混入 DA2 信息，无法代表真正的“无 DA2”。其 U-Net 仍保持 12 通道，原 DA2 的 4 个通道用零填充，因此不会因修改网络形状而引入额外变量。

## 运行前约定

以下命令假定：

- 项目目录：`/root/ApDepth-G`
- 数据目录：`/root/Dataset`
- SD2 基础模型：`/root/Marigold/pretrained_checkpoint/sd2-1`
- 输出根目录：`/root/ApDepth-G/output/ablation`

每个配置都必须从 SD2 基础模型独立训练，不要从另一个消融实验续训。配置文件名会成为输出子目录名；同名输出目录已存在时，`train.py` 会拒绝覆盖，应先改名保存旧实验或更换 `--output_dir`。

## 00 完整方法

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/00_full.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 01 去掉 DA2 远场弱锚点

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/01_no_far_anchor.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 02 去掉全部 VGC

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/02_no_vgc.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 03 去掉终端 SNR 对齐

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/03_no_terminal_snr.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 04 去掉 Min-SNR

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/04_no_min_snr.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 05 去掉多分辨率噪声

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/05_no_multi_res_noise.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 06 去掉通道 offset noise

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/06_no_offset_noise.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 07 去掉 latent gradient consistency

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/07_no_latent_gradient.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

## 08 去掉全部 DA2 信息

```bash
cd /root/ApDepth-G
python train.py \
  --config config/ablation/08_no_da2_prior.yaml \
  --base_data_dir /root/Dataset \
  --base_ckpt_dir /root/Marigold/pretrained_checkpoint \
  --output_dir /root/ApDepth-G/output/ablation \
  --no_wandb
```

建议优先运行 `00_full`、`01_no_far_anchor`、`02_no_vgc`、`08_no_da2_prior`，这四组直接回答 DA2 与天空塌陷修复各部分的贡献；其余配置用于拆解训练策略。
