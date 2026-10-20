# AGENT.md — Celeste Sky-Depth Consistency & Ordering Evaluation (SCOE)

> **项目执行说明（交给 Astra 6 / Coding Agent）**  
> 状态：设计规范 v1.0 / 待在用户本地代码仓库中实现  
> 基础仓库：https://github.com/Haruko386/ApDepth-G.git  
> 目标工作目录：用户本地已检出的 `ApDepth-G` 仓库根目录  
> 适用范围：**仅天空专项评估**，不修改 Celeste 训练、VGC 损失或主实验指标  
> 语言：中文文档、英文代码标识符、完整类型注解、可复现结果

---

## 0. Agent 角色、权限与工作纪律

你是一名资深计算机视觉研究工程师 / Python 评价系统开发者。任务是**在用户现有的 ApDepth-G 本地仓库上实现一套独立、严谨、可复现的天空深度评价工具**，而不是重新实现深度模型或修改论文方法。

必须遵守：

1. **先阅读本地源码再改动。** 本文件中的路径是基于公开仓库 `main` 页面观察到的线索；**本地文件、当前分支和用户已修改代码是最终事实来源**。先 `git status --short`、记录 commit、阅读 `README.md`、`run.py`、`infer.py`、`eval.py`、`src/dataset/`、`src/util/`、`script/`、`config/` 与相关测试；不假设本地与 GitHub 完全一致。
2. **不覆盖用户工作。** 不执行 `git reset --hard`、`git clean -fd`、强制 checkout、覆盖原文件、删除训练输出、提交/推送或大规模下载，除非用户明确授权。存在未提交改动时，只触碰新模块及获准文件。
3. **默认可新增专用评估代码、配置、文档与测试**；若需要修改训练、推理主路径、修改现有 `eval.py` 行为或新增重量级依赖，必须先说明理由并请用户确认。尽量通过适配器复用现有 `.npy` 预测结果。
4. **不编造实验结果。** 没有实际数据时使用 `N/A` 或 `pending`，不得用模拟数字充当论文结果。
5. **不要把“天空有效深度”等同于真值无穷远。** 不给天空像素赋予人工 GT `inf`、`1`、655.35m，也不计算无效天空 GT 的 AbsRel。
6. **不以模型输出选择评估图像。** 数据集筛选和区域定义由 RGB/官方 sky mask 固定，所有模型使用相同样本集与像素区域。
7. **每完成一阶段，更新进度表和简短工作日志**，报告完成百分比、已运行命令、测试结果、阻塞项；禁止宣称未执行的检查通过。
8. 当本文件与本地源码事实冲突时，记录差异、说明最小替代方案，必要时向用户询问；不得静默猜测。

### 0.1 已确认的公开仓库线索（仍须本地核验）

- 根目录有 `run.py`、`infer.py`、`eval.py`、`src/`、`script/`、`config/`、`data_split/`、`tests/`。
- 原有 `eval.py` 读取模型推理产生的 `.npy` 文件，支持 `--prediction_dir`、`--dataset_config`、`--base_data_dir`，通过数据集相对路径寻找预测值，并包含 GT 对齐逻辑。新任务**不应直接挪用 GT 对齐后的、裁剪过的预测**来计算天空专项指标。
- `run.py` 支持 `--checkpoint`、`--unet_checkpoint`、`--input_rgb_dir`、`--output_dir`、`--denoise_steps`、`--ensemble_size`、`--processing_res`、`--seed`。具体选项、文件名及输出格式必须对照本地实现。
- README 描述某些新 checkpoint 具有与训练匹配的 `scheduler/`、`inference_settings.json`，评估 8ch/12ch 模型时**必须加载正确的架构开关和匹配的 scheduler**；不可任意互换权重或配置。
- 公开 README 中主模型评估使用 50 DDIM steps、`ensemble_size=1` 等设置；真正实验参数以用户确定的 checkpoint 和实际 paper setting 为准。

参考：
- 仓库：https://github.com/Haruko386/ApDepth-G
- 现有评估：https://github.com/Haruko386/ApDepth-G/blob/main/eval.py
- VKITTI2 官方说明：https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/
- HRWSI 论文附录（天空掩码生成来源）：https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Xian_Structure-Guided_Ranking_Loss_CVPR_2020_supplemental.pdf

---

## 1. 研究背景、目标与不做的事情

**论文：** *Mind the Sky: Resolving Sky-Depth Inconsistency with Validity-Guided Diffusion*（Celeste）。

研究关注：天空像素可能被预测得异常近、天空内部存在无物理依据的深度起伏，同时模型需要保留天空边界附近前景的几何结构。

当前常规 NYUv2/KITTI/DIODE 的 AbsRel、δ1 多在**有效深度 GT**处计算；真实天空经常不在有效 GT 中，因而不能充分量化目标问题。

**交付目标：** 给出面向天空的独立、可重复的两数据集评价协议 **SCOE — Sky-Depth Consistency and Ordering Evaluation**：

- 数据集：**HRWSI** + **Virtual KITTI 2（严格独立测试集）**。
- 使用数据集天空语义标注 `S`，**绝不把 VGC 的 fully invalid `U` 或 DA2 远景支持集当成天空真值**。
- 测试图像天空占比至少 **5%**。
- 天空一致性主指标：全局非天空范围归一化 **IQR**（越低越好）。
- 辅助指标：全局非天空范围归一化 **P95–P5**（越低越好）。
- 天空/前景深度次序：**Sky–Foreground Ordering Error（SFOE）**（越低越好），局部边界候选带为图像短边的**每侧 2%**。
- 每张图先求分数，然后**逐图等权平均**；HRWSI 和 VKITTI2 分别出表。
- 全部使用原始浮点预测，不从 PNG 热力图/8bit 灰度图计算指标；不把天空值强制赋为无穷远。

### 1.1 非目标

- 不训练/微调新分割模型；若官方标注缺失，不擅自用 Mask2Former 等替代后直接称为官方 GT。
- 不修改 Celeste 的训练过程、VGC、噪声调度、模型架构或权重。
- 不把 SCOE 当成度量距离真值评价；它评估一致性与相对排序，而非天空的米级真值。
- 不因为特定模型得分不理想而修改阈值或筛选规则。
- 不直接将本评价的数值与 NYUv2/KITTI AbsRel 合并成单一总分。

---

## 2. 必须先完成的数据审计（第一阻断关卡）

### 2.1 HRWSI

1. 读取**用户本地实际下载包**，列出文件树、RGB、sky mask、validity mask、depth 文件格式及命名规则；不能根据论文“提供天空标注”就假设包里有某个具体文件名。
2. 优先使用作者提供的**原始验证/测试划分**；如无法确认官方测试划分，记录实际数据来源与 split，明确标记为 *curated evaluation subset*，不宣称其是官方 test。
3. 若 HRWSI 天空掩码是论文方法生成的伪语义标注，需要在文档中写明此限制；提供 RGB + mask 叠加质量检查（至少抽样 50 张，覆盖低云、树冠、建筑、窗户和极薄天空）。
4. 检查天空掩码与原图的空间对齐、掩码是否含未知值、图像/标注重复；报告总数、通过 5% 筛选后的数量。
5. 原 HRWSI RGB-D 深度可由立体匹配估计，天空深度可能经 sky-mask 后处理。**此处不以它的天空深度作为独立 GT。**

### 2.2 VKITTI2

官方路径模式（实际目录大小写以解包后为准）：

```text
SceneXX/<variation>/frames/rgb/Camera_0/rgb_00000.jpg
SceneXX/<variation>/frames/classSegmentation/Camera_0/classgt_00000.png
SceneXX/<variation>/colors.txt
```

官方 `colors.txt` 中 `Sky` 为 **RGB=(90,200,255)**。图像请优先用 PIL 的 RGB 模式读取，避免 OpenCV BGR 误判。**从本地 `colors.txt` 解析并校验类别颜色，而不是依赖硬编码。** sky = 全通道精确匹配；只有确认掩码经压缩/变换导致颜色漂移时才讨论容差方案（默认禁止）。

**训练/测试泄漏审计是 P0：**

- VKITTI2 存在 `Scene01/02/06/18/20` 与 `clone`、`fog`、`rain`、`morning` 等变化版本，同一 Scene 的不同变化、不同摄像头和连续帧不能简单认为相互独立。
- **首先核对本地 `data_split/`、`config/`、真实训练样本列表与当前所谓 test split**。输出以 `scene / variation / camera / frame` 为键的交集报告。
- 主要报告只可使用可证明**与训练几何场景独立**的测试部分。优先按 Scene 级别划分；如果训练覆盖了所有五个 Scene，则**不能把剩余天气/帧称为跨场景独立测试**。必须将此列为 BLOCKED，与用户商量：沿用非独立诊断集（明确标注 seen-scene）或重新划分/训练。不得静默标为 zero-shot 或 held-out scene。
- 还要检查与 HRWSI 的近重复图（如有明显数据交叉），至少依据路径/哈希/可选感知哈希给出审计报告。

**阶段产物：** `dataset_audit.json` + `dataset_audit.md` + 固定测试 manifest（包括数据来源、scene 和划分证据）。未完成审计，禁止生成“可投稿主表”。

---

## 3. 基础数据契约与预测表示

### 3.1 样本 Manifest（独立于模型）

建议每条记录：

```json
{
  "sample_id": "vkitti2/SceneXX/clone/Camera_0/00000",
  "dataset": "vkitti2",
  "split": "user_verified_test",
  "scene_id": "SceneXX",
  "variation": "clone",
  "camera": "Camera_0",
  "frame_id": "00000",
  "rgb_path": "...",
  "sky_mask_path": "...",
  "width": 1242,
  "height": 375,
  "sky_ratio": 0.32,
  "mask_source": "official_class_segmentation",
  "split_provenance": "..."
}
```

以上尺寸、比例**只是 schema 示例，不是真实样本数据**。路径使用相对数据根目录的可移植表示，另存必要的元数据/哈希。HRWSI 不存在的 scene/camera 字段允许为 `null`。

### 3.2 模型 Prediction Manifest

每个模型独立：`model_id`、checkpoint 路径或标识、预测根目录、输出格式、推理命令、git commit、seed、resolution、denoise steps、ensemble size、scheduler、对应 inference config、输出语义信息。

- 主流程读取 `float32`/`float64` 单通道二维 `.npy`（若输入是 `[1,H,W]`，只允许明确记录的 squeeze），拒绝任意彩色 PNG 充当原始数值图。
- 输入 `H×W` 必须与 RGB/sky mask 严格对齐，模型推理尺寸不同则由**明确的统一重采样协议**对齐；pred 使用连续插值，mask 使用最近邻，不允许把 mask 做 bilinear。
- 模型输出可能是 **depth、relative depth、disparity、inverse depth、归一化 latent-depth 表示**。逐模型阅读官方接口，明确**何者代表“更远”**，不能仅根据显示颜色判断。
- 本协议的 IQR/Wide 在**正仿射变换**下不变，**不在深度与倒数深度等非线性变换下不变**。如果两个模型定义在不同的表示空间，需要提供明示、可审计的共同评价表示及转换策略，并在报告中注明；无法可靠确定时，**不得宣称跨模型的幅值一致性分数完全可比**，SFOE 可在统一远近方向后独立报告。
- 默认不做基于 GT 的 least-square alignment，不做每张图的 sky-only min-max normalization，不对天空区域置远景常数，不对评价前预测 clip 到 `[0,1]`。
- 可允许 `distance_direction` 配置 `larger_is_farther` / `smaller_is_farther`，仅在原始模型的数学约定支持时执行保序、正仿射方向统一（如负号）。**负号翻转不等于 depth↔disparity 的物理转换**。
- 缺失模型预测、NaN/Inf、非预期维度必须报错并写 `failure_reason`；不能静默丢图，让某模型仅在简单样本上得分。

### 3.3 共享评价索引

先筛选并冻结数据集样本列表；**每一个模型必须提交同一组 sample_id 的预测**。若有失败样本，报告失败数与完整率，并且数值主比较使用一致的样本交集，交集的构造规则在模型打分前固定且公开；如大面积失败则阻断主表。不能给失败预测随意赋 0 分或偷换样本集合。

---

## 4. 正式指标定义（数学协议）

记图像 n 高宽为 `(H_n,W_n)`，`L_n=min(H_n,W_n)`；`S_n` 为独立 sky binary mask；`\bar S_n = 1-S_n`；模型 k 在原始图像坐标系下输出 `D_n^(k)`，统一为**数值越大越远**。图像中像素索引为 `i`。所有 percentile 需采用固定的 NumPy 定义（记录 method；推荐 `linear`），不随模型而异。

### 4.1 固定样本筛选：天空面积至少 5%

`sky_ratio(n) = sum_i S_n(i) / (H_n * W_n)`，保留 `sky_ratio >= 0.05` 的样本。

该阈值适用于原始语义 mask，**不使用预测深度、GT validity 或预测质量做数据筛选**。

### 4.2 天空内部 mask（建议默认）

为减少边界语义误差，定义内缩距离 `e_n = ceil(0.01*L_n)`，`S_int = erosion(S_n, e_n)`，度量单位为像素。**1% 是协议实现默认值，不是已经确定的 2% 候选带。**

- 实现时将其等价理解为：天空像素距非天空至少 `e_n`，采用欧氏距离变换；固定边界包含规则。
- 天空触碰图像边缘时不应把图像外全部视为非天空，避免无理由丢掉图像上边缘的天空；明确形态学 border 处理。
- 如果腐蚀后无有效天空，记录 `insufficient_sky_interior`；不得把空数组送入 percentile。

### 4.3 归一化参考：全图非天空深度动态范围

`R_n^(k) = Q95(D_n^(k)[~S_n]) - Q5(D_n^(k)[~S_n])`。

注意这是**模型自身预测的全图非天空区域**，而不是天空深度的范围。统计仅用 finite 数据，并且默认每个模型全图值必须 finite，避免模型相关的采样偏置。

分母退化：

- 不使用固定 `+1e-6` 导致仿射不变性被破坏。
- 可将 `R <= tolerance * robust_reference_scale` 标记为 `degenerate_foreground_range`，其中 `tolerance` 是无量纲预注册阈值，尺度量必须具有正仿射不变的差值形式；最保守且最清晰的 v1 实现是**只判断 R 是否数学上为 0 / 非有限，并报错或标记退化**，其余使用 double 精度计算。
- 单独输出 `R`、异常计数及前景范围分布，避免模型通过放大全局前景范围获得较好的归一化分数却未被发现。

### 4.4 一致性主指标：Sky-IQR ↓

`C_IQR(n,k) = [Q75(D[S_int])-Q25(D[S_int])] / R_n^(k)`。

### 4.5 一致性辅助指标：Sky-P95–P5 ↓

`C_WIDE(n,k) = [Q95(D[S_int])-Q5(D[S_int])] / R_n^(k)`。

这两个指标对 `D' = aD+b, a>0` 严格不变（分母非零）；**不**对任意倒数深度变换保持不变，也**不能**单独判断天空是否比建筑远。IQR 对小面积塌陷可能不敏感，Wide 较敏感但仍可能遗漏少于约 5% 的异常像素；需与 SFOE/可视化一起解释。

### 4.6 天空–前景局部次序错误：SFOE ↓

**用户已决定：** 每侧候选带宽 `r_n=ceil(0.02*L_n)`。推荐的可执行 v1 配对方案如下：

1. 从 mask 计算 `sky-to-foreground` 的有符号欧氏距离场，获取可定义稳定法线的 sky-foreground boundary；优先使用二值掩码距离场的中心差分求方向，法线朝向**非天空**。跳过法线无效、边界尖角严重歧义、无法取两侧样本的位置。
2. 在固定边界点 `p` 沿法线向两侧采样；为了避开最邻近的混合像素，使用 `t∈[ceil(0.01L_n), ceil(0.02L_n)]` 的整数距离。取样时仅保留严格属于正确类别的像素，不越界，前景候选可选限制为非天空且非 undefined/ignore 语义（如果数据集提供）。
3. 对每个有效边界点求局部两个样本序列的中位数 `d_s(p)` 与 `d_f(p)`。使用**预先固定**的边界点采样规则（例如以像素为单位每隔 2 像素采点、边界排序稳定），各模型必须使用**相同边界点、法线和采样索引**。生成边界配对缓存，计算模型前不再改变。
4. 定义 `E_ord(n,k) = (1/|P_n|) * sum_{p∈P_n} 1[d_s(p) <= d_f(p)]`，其中 `P_n` 为共享有效边界点集合。数值越低越好，范围 `[0,1]`；相等视为失败（常数深度不能通过排序指标）。
5. 对无法定义足够边界点的图像仅标记 `insufficient_boundary_pairs`；保留它在 IQR/Wide 中的合法成绩，**报告各指标自己的样本 N**。
6. 最少有效配对点数推荐 `min_pairs=50`，属于建议默认值，不是用户已定参数；先统计样本覆盖率，若大量剔除应与用户确认后再锁定。缓存并可视化 10–30 个边界采样案例，防止法线方向搞反。

**重要补充：** 这套指标衡量“天空比局部前景更远”的**相对排序**，不是 sky/foreground 的真实米级距离。仅以相邻区域排序好坏作为几何评价的一部分，不宣称完全保持了前景边缘的真实深度。若进一步主张“前景精度没有退化”，应使用有效 GT 另行评估。

### 4.7 汇总方式

同一数据集内，对于每个指标 `m`：

`m_dataset(k) = mean_{n∈N_m} m(n,k)`。

- **逐图等权**，不按 sky pixel 数量加权。
- 每个指标分别列出 `N_m` 与失败数；表中的各模型应在同一 `N_m` 上比较。
- HRWSI 和 VKITTI2 **各自一份结果**；不用混合成单一数字。
- 额外输出中位数、标准差、P25/P75、bootstrap 95% CI 作为可选辅助统计（按图像 bootstrap；VKITTI 连续帧高度相关时，应优先按 scene/sequence 分组 bootstrap，不可假设逐帧独立）。
- 主表统计的有效性依赖样本划分审计通过和输出协议锁定。

---

## 5. 与现有 ApDepth-G 的集成方案

### 5.1 首选“离线预测后评价”的解耦架构

不要在计算指标时重新加载 diffusion U-Net 或依赖 GPU；先统一保存预测 `.npy`，再在 CPU 上读取 sky mask 和深度进行统计。这样能兼容 Celeste、8ch/12ch ablation 以及 Marigold/DepthMaster 等外部模型。

**建议文件组织（按本地结构微调）：**

```text
ApDepth-G/
├── agent.md                         # 本文档，复制到本地仓库根目录
├── config/
│   └── sky_eval.yaml                # 仅 SCOE 的参数、数据/模型索引配置
├── src/
│   └── sky_eval/
│       ├── __init__.py
│       ├── schemas.py               # manifest / prediction schema
│       ├── datasets.py              # HRWSI, VKITTI2 indexing & mask readers
│       ├── masks.py                 # sky area, erosion, signed distance, boundary pairs
│       ├── metrics.py               # IQR / P95-P5 / SFOE, pure functions
│       ├── predictions.py           # reading existing .npy, output semantics validation
│       ├── audit.py                 # split leakage & mask QA audit
│       ├── aggregate.py             # equal-image averaging / failures / CI
│       └── visuals.py               # RGB/mask/prediction diagnostic layouts
├── script/
│   └── sky_eval.py                  # CLI: audit, prepare, evaluate, report
├── tests/
│   └── sky_eval/
│       ├── test_masks.py
│       ├── test_metrics.py
│       ├── test_affine_invariance.py
│       ├── test_dataset_index.py
│       └── test_end_to_end.py
└── doc/
    └── sky_eval.md                 # 安装、路径、数据集限制、例子和协议
```

如果 `script/sky_eval.py` 与现有脚本规范冲突，可改为 `script/sky_eval/` 目录，但需要解释。现有 `eval.py` 不改动，避免破坏旧 benchmark 的数值复现。

### 5.2 CLI 需求

至少提供如下能力（具体参数名可以调整，需文档同步）：

```bash
# 只读取数据与标注，检查掩码分布和 VKITTI 训练交叉；不加载模型
python script/sky_eval.py audit --config config/sky_eval.yaml

# 按 5% 等条件生成冻结的测试 manifest + 采样对缓存
python script/sky_eval.py prepare --config config/sky_eval.yaml

# 读取已存在的 float32 .npy 预测计算指标，不依赖 GPU
python script/sky_eval.py evaluate --config config/sky_eval.yaml --model celeste_full

# 合并多个模型结果，输出论文表格和质量报告
python script/sky_eval.py report --config config/sky_eval.yaml
```

必须提供 `--help`、准确退出码、清晰日志、dry-run/limit-samples 烟雾模式；命令要支持从仓库根目录运行，也能处理绝对数据路径。严禁输出覆盖旧结果而不告知；推荐按 `run_id`/`model_id` 隔离输出。

### 5.3 配置文件默认参数

```yaml
protocol:
  name: SCOE
  version: '1.0'
  min_sky_fraction: 0.05
  sky_interior_erosion_fraction: 0.01  # 建议默认，可将来锁定
  boundary_band_fraction: 0.02        # 用户已确认
  boundary_inner_offset_fraction: 0.01  # 建议默认，避免边界像素
  min_boundary_pairs: 50             # 建议默认，使用前报告保留率
  percentiles: [5, 25, 75, 95]
  percentile_method: linear
  aggregation: per_image_macro_mean
  exclude_sky_from_reference: true
  use_depth_gt_for_sky: false
  use_model_specific_sky_mask: false
  allow_scene_leakage: false

datasets:
  hrwsi:
    enabled: true
    root: null   # 用户配置，禁止硬编码用户机器路径
    split: null  # 必须从实际文件中审计
  vkitti2:
    enabled: true
    rgb_root: null
    class_seg_root: null
    train_list: null
    test_list: null
    verify_scene_disjoint: true

models: {}  # model_id -> prediction root, checkpoint metadata, output semantics
output:
  root: output/sky_eval
```

`train_list`/`test_list` 必须对应**实际数据集索引和用户训练记录**。不允许猜测哪一个 scene 就是 test。

---

## 6. 输出规范与论文表格

输出目录建议：

```text
output/sky_eval/<run_id>/
  config_resolved.yaml
  environment.json              # Python, NumPy, OpenCV/PIL, torch versions
  repository_state.json         # git commit, dirty status, code version
  dataset_audit.json
  dataset_audit.md
  manifest.jsonl
  mask_stats.csv
  sample_pairs/                 # 可重用的边界配对索引
  models/<model_id>/
    per_image.csv               # 三指标、R、计数、失败原因、mask info
    metrics.json                # 平均值、std、中位数、N、失败统计
    diagnostics/                # 人工质量检查图（可限量）
  comparison.csv
  comparison.md
  paper_table.tex
  protocol_lock.json            # hash: config + manifest + code + pair cache
```

`per_image.csv` 建议列：

- `dataset,sample_id,model_id,sky_ratio,width,height`
- `n_sky_pixels,n_sky_interior_pixels,n_boundary_pairs`
- `reference_range,sky_iqr,sky_p95_p5,sfoe`
- `status,failure_reason,prediction_path,prediction_dtype`

**生成主表格式：**

| Method | HRWSI IQR↓ | HRWSI Wide↓ | HRWSI SFOE↓ | VKITTI2 IQR↓ | VKITTI2 Wide↓ | VKITTI2 SFOE↓ |
|---|---:|---:|---:|---:|---:|---:|
| Marigold | TBD | TBD | TBD | TBD | TBD | TBD |
| Marigold V2 | TBD | TBD | TBD | TBD | TBD | TBD |
| DepthMaster | TBD | TBD | TBD | TBD | TBD | TBD |
| DA2-Giant | TBD | TBD | TBD | TBD | TBD | TBD |
| Celeste w/o VGC | TBD | TBD | TBD | TBD | TBD | TBD |
| Celeste Full | TBD | TBD | TBD | TBD | TBD | TBD |

仅在实际存在某模型预测时才填写真实数值。除了表格，提供数据集图像数量、掩码覆盖率、预测失败率、VKITTI 是否独立、每个指标有效 N 和分数范围说明。

### 6.1 可视化 QA

最低要求：

- 同一张图的 RGB、原始 sky mask、腐蚀后 sky interior、两侧 boundary band、配对法线示意。
- 两模型同图的预测对比：共享明确的显示范围和方向标注，不能为了视觉效果给每幅 sky 单独 min-max。
- 高分错误案例与低分正确案例各若干张，人工验证指标与肉眼判断一致。
- 错误掩码、方向反转、天空面积过小、仅有极细天空、天空有孔洞、树枝遮挡、法线穿过其他物体的诊断示例。
- 图片仅为 QA；定量只使用 `.npy` 浮点预测，不依赖渲染图。

---

## 7. 单元测试、性质测试与验收标准

### 7.1 必须通过的数值性质测试

1. **正仿射不变性**：对任意 `a>0,b`，`Sky-IQR(aD+b)==Sky-IQR(D)`、`Wide` 同样一致；SFOE 保持不变（忽略浮点边界误差）。
2. **完美稳定天空**：天空恒定且明显远于前景、非天空范围非退化 ⇒ IQR=Wide=0，SFOE=0。
3. **错误天空常数近景**：天空恒定但更近 ⇒ IQR=Wide=0，SFOE 接近/等于 1。证明不能仅依赖一致性评分。
4. **全图常数深度**：归一化分母为 0，必须标记 `degenerate_foreground_range`；SFOE 中相等视为排序失败；**不可获得完美总评**。
5. **局部塌陷**：约 10% 天空故意变近，Wide 应体现更大变化，IQR 可能维持较小；将该局限写入文档。
6. **sky mask 一致性**：BGR/RGB 错误必须能被至少一个官方 VKITTI2 样例测试发现。
7. **边界方向**：天空上/建筑下的人工图，取样法线一定指向前景，互换方向会导致测试失败。
8. **全景 sky / 零 sky / 太少 sky**：分别按协议排除或报状态；不崩溃、不输出 NaN 数值当作有效结果。
9. **预测尺寸/有限性**：不匹配、缺少文件、NaN/Inf 必须给出可读错误和计数。
10. **模型比较公平性**：相同 `sample_id`、mask 与边界配对索引被所有模型共用，不因为模型数值而变化。
11. **跨图汇总**：两张天空大小不同的图像权重相同；不是全体像素 pooled percentile 后再计算。
12. **确定性**：相同 manifest/config/prediction 与运行环境重复运行，应得到一致结果（如使用 bootstrap，固定随机种子）。
13. **泄漏阻断**：若 VKITTI 测试与训练 scene 存在交集、且 `allow_scene_leakage=false`，生成正式主表必须失败并显示交集明细。
14. **分母投机测试**：刻意放大非天空动态范围时指标可能改善，文档必须说明这不是天空真实质量提升，并且输出 reference range 诊断。

### 7.2 最小端到端烟雾测试

构造 3–5 组人工 RGB/mask/float32 `.npy`，无 GPU、无外部数据依赖地完整运行 `audit -> prepare -> evaluate -> report`，检查 CSV/JSON/TeX 数据一致性。

只在拥有真实数据和本地模型预测的条件下再跑真数据 smoke test；不得将人工数据打成论文实测结果。

### 7.3 完成交付定义（DoD）

- [ ] 源码与训练/推理现有接口核验记录已完成。
- [ ] HRWSI/VKITTI2 数据与 split 审计完成；对潜在场景泄漏有明确结论。
- [ ] 固定数据集 manifest、mask QA 和边界配对缓存生成成功。
- [ ] CPU 离线指标 evaluator 实现完毕。
- [ ] Sky-IQR / Sky-P95-P5 / SFOE 与数学定义相符。
- [ ] 用户确定的 5% 和 2% 参数不可在结果生成后自动调优。
- [ ] 模型表示与方向经过显式检查，未经核实的跨模型比较被阻断。
- [ ] 处理所有失败状态，无静默缺失与虚构结果。
- [ ] 单元测试、性质测试与烟雾测试均通过。
- [ ] 提供 README 使用示例、正式指标 LaTeX 公式和可生成的结果表。
- [ ] 未修改原有训练或常规 `eval.py` 数值逻辑。
- [ ] 最终报告含实际执行命令、测试记录、未完成限制和下一步计划。

---

## 8. 开发阶段与进度记录（每阶段必须更新）

> 预计工作占比仅用于可视化进度，不等于真实时间；未完成项保持 `[ ]`。

| Phase | 工作 | 权重 | 状态 | 完成标准 |
|---|---|---:|---|---|
| P0 | 仓库阅读与本地差异审计 | 10% | [ ] | 已记录 git 状态、现有 `.npy` 约定、checkpoint/inference 规则 |
| P1 | 数据集解包结构与 split/泄漏审计 | 20% | [ ] | 数据树、官方天空标签、测试名单、split 证据 |
| P2 | 掩码构造与边界配对 | 20% | [ ] | 固定 5% / 1% / 2% 规则，缓存与 QA 可视化 |
| P3 | SCOE 指标与等权汇总 | 20% | [ ] | 指标实现及正仿射不变性测试 |
| P4 | CLI、预测适配与结果导出 | 15% | [ ] | 可复现 CSV/JSON/TeX 与安全失败 |
| P5 | 测试、文档、最终审查 | 15% | [ ] | 单元/端到端测试、复现指南、限制说明 |
| **合计** | | **100%** | **0%（开始前）** | |

建议每次汇报格式：

```text
[Celeste SCOE] 阶段 P? / 总进度 ?%
本轮完成：...
修改文件：...
执行命令：...
测试结果：...（未运行需写“未运行”）
关键发现 / 不确定性：...
阻塞项（需用户决策）：...
下一步：...
```

在 `doc/sky_eval_devlog.md` 追加日志，逐阶段勾选，不制造虚假的进度或测试记录。

---

## 9. Agent 首轮必须执行的工作（现在开始）

1. 在本地 `ApDepth-G` 根目录检查 `git status` 和关键源码；确认当前分支与 `main` 的差异。
2. 输出一份**不超过一页的实现路径映射**，说明如何复用 `run.py`/`infer.py`/现有 `.npy` 命名，哪些模块必须新建。
3. 检查 HRWSI 和 VKITTI2 实际数据是否在本地；如无，**先做人工 fixture 的纯指标系统，不阻塞开发**，并清楚列出用户需要提供的目录/文件清单。
4. 优先进行 VKITTI2 train/test 重叠检查。如果当前训练已覆盖全部 Scene，立刻告知用户实验不能声称场景独立，不要擅自自行划分后当作零样本测试。
5. 开始 P0–P3 的**最小可运行版本**：mask 读取、人工数据下 IQR/Wide/SFOE 和单元测试。除涉及用户决定的研究设定外，不必反复要求许可即可推进新增模块。
6. 在能够证明结果正确之前，不改论文、不填真实实验表、不推送 GitHub。

### 需要用户决策才允许改变的设置

- `min_sky_fraction = 0.05`
- `boundary_band_fraction = 0.02`（每侧）
- `IQR` 为主，`P95–P5` 为辅助
- Global non-sky normalization + local sky-foreground ordering
- HRWSI + VKITTI2 两个数据集，分别出表
- 是否接受已见场景的 VKITTI2 **diagnostic**（不可冒称 held-out）
- 当模型输出是 disparity / inverse depth、又无合法统一表示策略时，如何限定跨模型一致性比较

其它如 **1% sky erosion、1% 边界内侧偏移、min_pairs=50、percentile linear** 都是设计默认建议；实现时记录到固定版本配置，若第一轮审计证明不合理，**在正式测分前**向用户提交一次修改提案并锁定。不能看过模型结果再调参。

---

## 10. 论文表述边界

可以说：`SCOE measures the relative consistency of predicted sky depths and their ordering with respect to nearby foreground regions.`

不得说：`SCOE directly measures true infinite sky depth`、`a lower IQR proves correct geometry`、`VGC eliminates all teacher errors`。

当结果完成后，第四章可单独增加 `Sky-Depth Consistency Evaluation`，包含数据集及预注册协议、IQR/Wide/SFOE 数学定义、样本数量、失效案例及消融对照；正式措辞必须依据实测结果再写。

**最终成功标准不是“Celeste 排名第一”，而是评价过程具有数学解释力、独立性、公平性与可复现性。**
