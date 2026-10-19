# 监督覆盖率：服务器记录，本地统一画图

两个服务器分别扫描自己拥有的数据集，每个数据集生成一个 `disp_name.jsonl`。
收集这些文件后，在本地运行另一个脚本，一次生成完整图表。
数据集目录、split 和解码规则直接沿用项目现有配置。

## 训练服务器：Hypersim + VKITTI

```bash
cd /root/ApDepth-G
python script/supervision_coverage.py \
  --dataset_config config/dataset/dataset_train.yaml \
  --base_data_dir /root/Dataset \
  --device cpu --num_workers 8 \
  --output_dir output/coverage_records_train
```

得到 `hypersim_train.jsonl`、`vkitti_train.jsonl`。
也可使用完整训练配置 `--dataset_config config/train_marigold.yaml --split train`，
只选择对应 split，不要求其他 split 的数据集存在。

## 推理服务器：五个主实验数据集

```bash
python script/supervision_coverage.py \
  --dataset_config \
    config/dataset/data_nyu_test.yaml \
    config/dataset/data_kitti_eigen_test.yaml \
    config/dataset/data_eth3d.yaml \
    config/dataset/data_scannet_val.yaml \
    config/dataset/data_diode_all.yaml \
  --base_data_dir /root/Dataset \
  --device cpu --num_workers 8 \
  --output_dir output/coverage_records_eval
```

按上述顺序生成 `nyu_test_full.jsonl`、`kitti_eigen_test_full.jsonl`、
`eth3d_full.jsonl`、`scannet_val_800_1.jsonl`、`diode_val_all.jsonl`。
文件名来自配置的 disp_name，保留 split 信息。
混合配置按 dataset_list 顺序展开；多个单独配置按命令行顺序处理。

## 本地：收集七个文件，一次画图

把七个 `.jsonl` 复制到本地项目的 `output/coverage_records/`，运行：

```text
python script/plot_supervision_coverage.py --records output/coverage_records --output_dir output/coverage_figure
```

文件夹输入默认按 Hypersim、VKITTI、NYUv2、KITTI、ETH3D、ScanNet、DIODE 排序；
通配符输入按文件名排序。指定其它柱子顺序时，按顺序传入文件即可。
下面的多行写法适用于 Bash，PowerShell 可合并为一行：

```bash
python script/plot_supervision_coverage.py --records \
  output/coverage_records/hypersim_train.jsonl \
  output/coverage_records/vkitti_train.jsonl \
  output/coverage_records/nyu_test_full.jsonl \
  output/coverage_records/kitti_eigen_test_full.jsonl \
  output/coverage_records/eth3d_full.jsonl \
  output/coverage_records/scannet_val_800_1.jsonl \
  output/coverage_records/diode_val_all.jsonl \
  --output_dir output/coverage_figure
```

本地仅需 numpy、matplotlib，不需要数据集、split、配置文件、PyTorch 或模型。
文件中记录的服务器路径只用于追溯，不会被打开。输出：

- `coverage_all.png/.pdf`：像素覆盖率竖向 100% 堆叠柱状图。
- `coverage_latent.png/.pdf`：全有效／全无效／混合块占比图。
- `summary.csv`、`latent_summary.csv`：两种聚合方式的汇总表。
- `summary.json`：汇总结果及原始统计口径。
- `image_counts.csv`：各数据集的图片数和 Training/Test 用途；JSON 同时记录分类及总图片数。
- `coverage_combined_single_column.png/.pdf`：额外的 3.5 英寸单栏图，三类块占比使用分组柱，像素有效覆盖率使用折线。

图中明确分开训练集（Hypersim、VKITTI）与五个测试集，柱子下标注每集图片数，
标题下显示总图片数、训练图片数、测试图片数。数量均来自实际完成的逐图记录。

可随时修改绘图样式或加 `--average image` 重画，无需重扫数据。
重画更新输出图表和汇总表，不改动输入记录文件。
如果只生成单栏组合图并保留已有两张独立图，追加 `--combined_only`：

```text
python script/plot_supervision_coverage.py --records output/vgc_coverge --output_dir output/vgc_coverge/figures --combined_only
```

单栏组合图仅保留图例、坐标轴、训练／测试分组、每集图片数及覆盖率数值；
总图片量和详细说明仍可从汇总文件中查看。

## 单个记录文件

JSONL 每行一个 JSON 对象，每个数据集只有一个需要拷回本地的文件：

1. `metadata`：数据集、样本数、解析后的配置、深度范围、裁剪／缩放、块尺寸和抽样设置。
2. 每图一条 `image`：split 索引、RGB 路径、尺寸、各类像素数、逐图有效占比和块计数。
3. `complete`：完成标记和实际图像数。

逐图记录按 split 顺序保存；随机抽样后仍按原索引排列。
运行中写 `名称.jsonl.partial`，每图刷新，完成该数据集后改名为 `名称.jsonl`。
后续数据集出错不影响前面已经完成的文件。
本地只自动收集 `.jsonl`，校验完成标记、样本数和计数一致性，并拒绝重复数据集。
同名记录或 partial 不会被服务器覆盖；重跑请选新目录。

## 参数与统计口径

服务器依赖项目原环境的 torch、torchvision、numpy、Pillow、omegaconf、tqdm，
不需要 matplotlib。只加载深度／掩码，不读 RGB 或模型；兼容现有 tar／解压目录。
只统计 split 列出的图像，不扫描 split 外的文件，也不按混合训练抽样概率加权。

- `--device cpu` 默认；`cuda:0` 或 `auto` 也可。GPU 用于计数和池化，解码仍在 CPU。
- `--num_workers 0` 关闭多进程。
- `--max_samples 20` 每数据集固定种子无放回抽样；正式运行去掉它。默认 seed=2024。
- `--dry_run` 只核对配置、数据根目录和清单数量。
- `--dataset_dir disp_name=/absolute/path` 可覆盖单个目录；现有路径正确时无需使用。
- `--depth_type raw` 默认；`filled` 可使用 NYUv2 填充深度，无填充数据的类仍用 raw。
- `--native_resolution` 忽略 resize_to_hw，仍保留 benchmark 空间裁剪。

像素分母是 benchmark 空间裁剪、配置的最近邻缩放之后的图像范围。
每个像素恰好属于以下一类：

- `valid_pixels`：通过项目配置的有效性规则。
- `data_invalid_pixels`：未通过深度有效性规则。
- `crop_excluded_pixels`：深度有效，但被 Eigen/Garg 等评价掩码排除。

benchmark 空间裁剪外像素不进入分母；评价掩码排除像素留在分母中并单列。
Hypersim PNG 按毫米解码，严格 `1e-5 < d < 65 m`；VKITTI 按厘米解码，
严格 `1e-5 < d < 80 m`。DIODE 沿用现有类直接使用提供的有效掩码，
不额外叠加名义上的 0.6–350 m 限制。非有限／非正深度被标为有效时会报错。
测试集统计的是评价支持覆盖率，不表示这些数据参与训练。
这里的 DIODE 覆盖率基于数据集原始有效掩码，不包含 `eval.py` 按 PPD 协议
额外执行的深度范围和 Sobel 边缘过滤，因此与最终指标使用的掩码覆盖率不同。

默认图表是总有效像素数／总像素数；本地 `--average image` 改为逐图占比等权平均。
CSV 始终保留两种结果；空区域不参与该区域的图像平均。

块统计遵循 VGC 的全块有效性规则，默认 `--latent_scale 8`。
不包含样本门控和 DA2 远场选择，也不代表 VAE 的完整感受野。
不能整除块尺寸时，边缘按原池化规则舍去并记录 latent_border_pixels。
本地合并要求块尺寸相同。

## 可选：独立天空标注

无效深度不等于天空。没有语义标注时，只统计整体覆盖率。
已有独立天空标注时，可给单个数据集配置附加：

```yaml
coverage:
  sky_mask_list: /root/annotations/vkitti_sky_files.txt
  sky_mask_dir: /root/annotations/vkitti_sky
```

标注清单每行 `RGB相对路径 二值掩码相对路径`，第一列与原 split 一致；
第二列相对 sky_mask_dir，未指定时相对标注清单目录。
掩码必须与裁剪前的深度图同尺寸、已配准；支持 PNG/NPY，0=非天空、1/255=天空。
选中样本都须有标注。文件会额外保存 sky/non_sky 计数，本地自动画对应图表，
无需另带天空掩码。
