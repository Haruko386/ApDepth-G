# 冻结多步 VGC 的深度解码器后训练

更新：2026-10-01。状态：代码与小模型检查通过，真实天空效果尚未验证。

当前语义关系主训练已失败并撤下，具体图像观察见 [failed.md 第 13 节](../failed.md)。
本方案从**原先相对有效的 VGC U-Net**出发。请勿使用语义关系、已知远端补监督或轨迹后训练的失败权重。

## 为什么这次改解码器

之前关系矩阵的目标不确定深度远近方向；latent 上的相似或平滑，也不保证最终天空像素为统一远端。
因此此次将监督直接放在最后解码结果上，并把优化范围限制在原生 VAE 的 `decoder`。

```text
一次性缓存：RGB + DA2 prior → 冻结 VGC U-Net 完整 50 步 DDIM → 最终 latent z
                                                        → 原解码结果 d_ref

后训练：固定 z → 可训练的 VAE decoder → 原始像素深度 d
                                      ├─ 已确认天空内部：监督 d = +1
                                      └─ 非天空：保留 d_ref 的数值和局部梯度

部署：原 RGB/DA2 编码器 + 原 VGC U-Net 多步 DDIM + 校准后的 decoder
```

U-Net、VAE encoder、quant_conv、post_quant_conv、DA2 全部冻结。只更新 decoder，不增加网络输入或推理分割器。
训练缓存来自**从纯噪声走完完整采样链的最终状态**，不使用 GT 加噪得到的中间预测。
由于编码器与 U-Net 固定，后训练不会使缓存对应的 latent 分布过时。

目标函数为：

```text
L = mean_sky((d - 1)^2)
  + 5 * mean_non_sky(SmoothL1(d, d_ref; beta=0.05))
  + 0.5 * mean_non_sky_pairs(|gradient(d) - gradient(d_ref)|)
```

`d` 是 `decode_depth` 裁剪之前的输出；+1 是归一化最远端，最终显示为 `(d+1)/2 = 1`。
它代表本模型的相对深度远端约定，并非米制无限远。
不对预测做可改变符号的仿射对齐，不在 loss 前 clamp，不重新做 min-max 归一化。
平方误差同时惩罚天空均值偏离和内部起伏，不额外叠加天空平滑 loss。

天空标签先腐蚀 2 个输出像素，减少边界误标；非天空标签保留细线、树枝等结构的监督。
标签不确定的位置跳过，不把 invalid GT、>=80m、低纹理、上半图或 DA2 的远端当成天空标签。

与已失败方法的区别：

| 旧方案 | 此次变化 |
| --- | --- |
| 语义关系主训练 | 有符号的最终像素深度监督；冻结 U-Net 表示 |
| >=80m 补标签 | 使用天空语义标签，不扩大 GT 主扩散监督域；不把目标再编码为 latent |
| SIPR | 不扰动 DA2 prior、不训练 U-Net、不取 GT 分位数锚点；训练原生解码器直接拟合输出 |
| 短轨迹后训练 | 不展开 GT 噪声轨迹；缓存完整 VGC 采样结果，后训练不执行扩散 |
| PFR 后处理 | 优化模型 decoder 权重；推理没有阈值纠正、mask 融合或输出涂色 |

## 研究依据与证据边界

[Marigold V2（2026-09）](https://arxiv.org/html/2609.08084v1) 在第二阶段解冻 VAE decoder，
并结合 SinkLoss 改善输出边界，说明最终解码映射也是可优化对象。其单步 DiT 和完整训练方法没有迁移到此处。
本实现不采用 SinkLoss：天空内部目标为常量，先用直接、有方向的像素约束做最小实验。

也有反面证据：[Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think](https://arxiv.org/html/2409.11355v2)
的附录 C / Table A-2 中，普通 decoder 微调没有提高其深度基准结果。
所以“解冻 decoder 就会更好”并不成立。此次是有天空语义标签、非天空保留和独立验证的定向校准假设。

本方案不是已证明有效或已证明新颖的论文方法。如果冻结 latent 丢失了辨认天空所需的信息，
decoder 可能无法在修复天空的同时保留物体。阴天、雾、白墙、反光和动漫域都必须实测。
真实图像天空标签也可能误把远楼、雪山或天花板标为天空，这类错误会直接污染监督。

## 准备数据与标签

使用独立的训练 RGB 和验证 RGB 文件夹，按场景划分，不把同一视频相邻帧分到两边。
可以先做数百张以内的小实验，覆盖晴天、云、雾、树枝/楼房边界和动漫；同时加入无天空的室内图做保留样本。
数量建议只是实验起点，没有验证过最优数据量。

本次提供的 `output/out` 七张图保留作最终人工检查，不应全放入训练集再宣称泛化提升。
独立验证集也应同时包含天空和室内/无天空场景。

如果已有可靠的天空标注，直接准备 JSONL，每行：

```json
{"image":"rgb/scene01.jpg","mask":"masks/scene01.png","split":"train","group":"scene01"}
{"image":"rgb/scene02.jpg","mask":"masks/scene02.png","split":"val","group":"scene02"}
```

路径相对 manifest 所在目录；mask 与原图同尺寸，单通道 PNG：255=天空、0=非天空、128=不确定。
`group` 可省略；存在时程序会禁止同一 group 跨训练/验证集。重复图片字节内容也会被拒绝。
不同编码或同场景不同帧的泄漏仍需由数据划分保证。

没有标注时，用 [NVIDIA SegFormer ADE20K 权重](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640)
离线生成候选 mask。这是新增的**可选数据准备依赖**，只在下面第一条命令中加载；本地权重目录也可传给 `--model`。
脚本按模型的 `sky` 类名找类别，不硬编码可能错位的类别编号。
天空概率 >0.9 标天空；天空概率 <0.05 且最高类别置信度 >0.5 标非天空；其余为不确定。
概率阈值不是准确率保证。缓存前检查并修正 mask，尤其是薄云、浓雾、树枝、线缆和室内白墙。

标签生成脚本明确使用 `use_safetensors=True`，兼容项目的 PyTorch 2.4，避免新版 Transformers
拒绝加载 `.bin` 权重的错误。Hub 模型可能通过 safetensors 转换 revision 提供安全格式，首次解析仍需联网；
使用本地模型目录时需包含 `model.safetensors`（或对应分片索引）。不应通过关闭安全检查绕过报错。
模型加载失败后遗留的空 `labels/` 或空 `labels/masks/` 可以直接重跑；有已生成内容时应改用新输出目录。

## 运行指令

以下均在项目根目录执行。辅助脚本用 `python -m script...`，避免脚本目录导致项目包导入失败。
`/path/to/SD2` 是原 VGC 使用的完整 pipeline（例如 `pretrained_checkpoint/stable-diffusion-2`），
`/path/to/VGC/checkpoint/iter_023000` 是包含 `unet/` 的原 VGC checkpoint。

1. 可选：生成候选天空标注。

如果沿用项目配置中的已解压 VKITTI/Hypersim，先从文件清单自动提取 RGB 子集：

```bash
python -m script.prepare_decoder_rgb --base_data_dir /root/Dataset --output_dir ./output/decoder_rgb
```

默认训练集为 160 张 VKITTI + 40 张 Hypersim，验证集为 40 + 10 张，数量只是小实验起点。
VKITTI 沿用仓库 train/val 场景划分；Hypersim 按场景留出约 20% 作为此次后训练验证候选。
这是后训练层面的留出，不表示这些图像从未参与过原 VGC 的训练。脚本只读取清单第一列 RGB，
不会把深度 PNG 当 RGB。数据路径来自 `config/dataset/dataset_train.yaml`，文件缺失会明确报错。
可据此将下一条命令中的目录分别替换为 `./output/decoder_rgb/train` 和 `./output/decoder_rgb/val`。

```bash
python -m script.prepare_sky_masks --train_rgb_dir /path/to/train_rgb --val_rgb_dir /path/to/val_rgb --output_dir ./output/sky_labels
```

2. 从原 VGC 完整推理缓存最终 latent。它会保存一份精确匹配的 `source_vae/`，不是从失败模型重编码深度图。

```bash
python -m script.cache_decoder_latents --base_checkpoint /path/to/SD2 --unet_checkpoint /path/to/VGC/checkpoint/iter_023000 --manifest ./output/sky_labels/manifest.jsonl --output_dir ./output/decoder_cache --steps 50 --processing_res 512 --seeds 2024 2025
```

有自备标注时替换 `--manifest` 即可。一次缓存开销是图像数 × seed 数 × 50 次 U-Net 前向，
**这一步并不免费**；后面重复调 loss 或学习率可复用缓存。只在缓存完整后生成 `index.json`，不支持直接训练未完成缓存。
缓存脚本仍依赖项目原有外部 DA2-Giant 源码/权重与 GPU 环境；没有 DA2 时无法完成真实缓存。

3. 后训练 decoder。

```bash
python train_decoder_calibration.py --config config/train_decoder_calibration.yaml --cache_dir ./output/decoder_cache --output_dir ./output/decoder_calibration
```

默认 1000 个优化步，LR 1e-5，50 步学习率预热，单图前向、梯度累积 4。
这是实验上限，不需要等到最后才查看：每 50 步验证、每 250 步保存可恢复状态。
如果加入了无天空回放图，抽样中约 25% 用它们；其余从含天空图中采样，两类都保护非天空区域。
本入口只加载 `source_vae`，训练中不会加载或运行 DA2、CLIP 或 U-Net。

4. 使用通过验证筛选的 decoder 推理，仍然 50 步。

```bash
python run.py --checkpoint /path/to/SD2 --unet_checkpoint /path/to/VGC/checkpoint/iter_023000 --decoder_checkpoint ./output/decoder_calibration/best --input_rgb_dir ./output/out --output_dir ./output/decoder_check --denoise_steps 50 --ensemble_size 1 --processing_res 512 --seed 2024
```

保持相同参数，去掉 `--decoder_checkpoint` 跑原 VGC 对照。不要拿当前失败语义模型作为唯一对照。
`run.py` / `infer.py` 都支持这两个覆盖参数；decoder 加载器会核对 encoder/quant_conv/post_quant_conv
与缓存来源相同，只装载 decoder 权重。推理不需要 manifest 或 mask。

初次比较固定 ensemble=1、seed、步数和分辨率。更换采样步数、分辨率或开启 ensemble，
会改变 latent / 聚合结果的分布，需要另做验证；不承诺一次校准适用于所有设置。

中断后恢复到新的输出目录，配置和缓存保持完全相同：

```bash
python train_decoder_calibration.py --config config/train_decoder_calibration.yaml --cache_dir ./output/decoder_cache --resume ./output/decoder_calibration/step_000250 --output_dir ./output/decoder_calibration_resumed
```

## 如何判断这次是否值得保留

`baseline.json` 记录原 VGC 在验证缓存上的指标，`metrics.jsonl` 记录每次验证，所有深度指标用 `[0,1]` 单位且不裁掉越界误差：

- `sky_mae`：天空与 1 的绝对误差。
- `sky_std`：每张图天空内部标准差，再取平均。
- `sky_below_09`：天空预测 <0.9 的比例。
- `non_sky_mae`：非天空相对原 VGC 的平均漂移。
- `non_sky_p95`：逐图漂移第 95 百分位中的最大值，防止平均数掩盖单幅图损伤。

`best/` 仅在天空误差优于已有候选、低于 0.9 的比例不差于原 VGC，
并且非天空 MAE ≤0.02、最差图 P95 ≤0.05 时保存。阈值是显式实验容差，不是通用质量标准。
这些条件只表示相对基线有改善，**不代表天空已经足够远或完全解决**；还应查看天空误差绝对值、边界及未见场景。
如果始终不满足条件，程序不产生 `best/`，不要直接把最后一步当作成功模型。

若前 100～250 步出现天空指标不降、非天空漂移持续超限或新斑点，应停止本轮，检查 mask 和原 VGC 缓存。
如果数据与标签无误但仍有这一冲突，说明 decoder 的表达/可分性可能不足；不建议再次盲目加大权重或训练时间。
非天空保留只约束相对原模型的变化，不是 GT 精度指标；它会保留原模型本已有的错误。

## 本地实现验证

```bash
python -B -m unittest discover -s tests -p "test_decoder*.py" -v
```

10 项检查通过：天空纠正方向与饱和区梯度、未知标签和非天空保护、边界腐蚀、空区域、
验证筛选、完整多步采样捕获及异常清理、标注对齐/数据泄漏、仅 decoder 更新、推理装载保护编码器，
以及真实小型 AutoencoderKL 的训练/保存/断点精确恢复。
没有真实 SD2/DA2 权重和训练集，所以没有执行本方案的实际天空训练；测试结果不是效果证明。
