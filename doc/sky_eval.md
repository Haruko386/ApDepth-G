# SCOE：天空深度一致性与局部排序评价

实现范围：独立 CPU 离线工具，不加载 diffusion、DA2 或 GPU，不访问网络。
原训练、推理及 `eval.py` 保持不变。**本轮按用户要求不执行测试、不下载模型/数据集。**
测试用例已提供，下面的运行命令是后续使用说明，不代表已经执行。

2026-10-20 新增用户指定的 **Mask2Former 伪标签诊断流程**：`script/segment_sky.py`
从 `BASE_DATA_DIR` 的目录或 tar 读取 VKITTI2、KITTI、DIODE RGB，生成 mask 和可用的 SCOE 配置。
运行方法见 [README](../README.md#mask2former-天空分割vkitti2--kitti--diode)。
以下 HRWSI+官方 VKITTI2 主协议保持原定义；伪标签流程单独标记，支持三数据集诊断表。
分割阶段可使用 GPU；数值评价阶段仍为 CPU。
分割入口必填 `--data_split`，可指定 `--dataset kitti` 等数据集；多个 dataset 与 split 按顺序一一对应。
原始 split 路径保留到生成的推理/SCOE 配置，eval 核验相同名单与哈希。
掩码位于数据集自己的 `sky_masks/<运行名>/` 下；tar 数据位于归档所在目录。
项目 `output/sky_segmentation/<运行名>/` 只保存元数据、预览和配置。

## 当前交付与边界

- 新增 `src/sky_eval/`、`script/sky_eval.py`、`config/sky_eval.yaml`、`tests/sky_eval/test_scoe.py`。
- 支持 `audit → prepare → evaluate → report`，记录数据/配置/源码/缓存/预测 SHA256。
- HRWSI 和 VKITTI2 分别统计，每个指标的跨模型比较使用预先规定的共同有效样本交集。
- 缺少数据、语义证据或独立性证据时阻断正式主表。默认仅生成明确标注的诊断报告。
- 尚未执行单元、性质、CLI 或真实数据测试；未生成真实模型成绩。
- v1 严格要求完整 RGB 坐标、原始分辨率浮点预测；**不实现自动重采样、裁剪反变换或 depth/disparity 非线性转换**。
- 不生成 bootstrap CI。VKITTI 连续帧不能当作独立样本；后续若增加 CI，应按 scene/sequence 分组。
- 未完成真实 mask 的人工 QA、至少 50 张 HRWSI 覆盖检查、真实边界配对保留率审查。

## 本地核验结论

初始 commit 为 `02e2571257fbf2bfcc4655e183793255023a3746`，分支 main。
仓库 train 名单为 20,148 帧（Scene02/06/18/20），val 为 5,364 帧（Scene01）；
精确 scene/variation/camera/frame 和 scene 交集都为零。
这只是名单审计，不能证明一个尚未指定的 checkpoint 没训练过 Scene01。

本地 `F:/vkitti/vkitti_rgb/rgb` 下有 RGB，但 `F:/vkitti` 没有发现 `colors.txt` 和 `classgt_*.png`。
用户已确认目前没有所需语义标注、HRWSI、目标预测和训练记录，先交付工具。
单独 depth 或深度有效性 mask 不可替代 sky semantic mask。

`F:/train_marigold/config.yaml` 引用仓库相对训练名单；未证实该文件与本次待评 checkpoint 对应，
也没有当时的名单哈希，因此仅记录为线索。

`run.py` 保存 `depth_npy/<rgb stem>_pred.npy`。
`infer.py` 保留 RGB 相对父目录；VKITTI `id` 模式文件名为 `pred_rgb_00000.npy`。
原 VKITTI 配置启用 benchmark crop，不能将这种预测直接拉伸为完整天空图。
可另建仅供推理的 dataset config，关闭 crop，再沿用原 `infer.py`；本实现不改原配置。
多 Scene 同名帧不能直接用扁平 `run` 命名，检测到碰撞会报错，建议显式 sample_id 映射。

本地 `model_overrides.py` 仅接受 12ch U-Net，不能假定通用 8ch 支持。
checkpoint 的 `scheduler/`、`inference_settings.json` 或训练 `config.yaml` 必须匹配。
pipeline 原生输出在保存前已有 `[0,1]` clip，故这里的“原始预测”指保存的 float `.npy`，
不等于未截断 latent。本工具不再 clip、不做 GT 对齐；模型原生后处理必须记入元数据。

## 环境与命令

使用已有 NumPy、SciPy、Pillow、PyYAML；无新增重量级依赖。
本机已检查 DL 环境包含这些包；默认 Python 缺 NumPy。未安装任何包。

在仓库根目录的 PowerShell 中，后续可以使用：

```powershell
$scoePython = 'D:\Anaconda\envs\DL\python.exe'
& $scoePython -B script/sky_eval.py --help
& $scoePython -B script/sky_eval.py audit --config config/sky_eval.yaml --dry-run
& $scoePython -B script/sky_eval.py audit --config config/sky_eval.yaml --run-id experiment_001
& $scoePython -B script/sky_eval.py prepare --config config/sky_eval.yaml --run-id experiment_001
& $scoePython -B script/sky_eval.py evaluate --config config/sky_eval.yaml --run-id experiment_001 --model celeste_full
& $scoePython -B script/sky_eval.py report --config config/sky_eval.yaml --run-id experiment_001
# 只有数据、划分、QA、模型表示和训练证据全部通过才可执行正式导出：
& $scoePython -B script/sky_eval.py report --config config/sky_eval.yaml --run-id experiment_001 --formal
```

全部配置相对路径均相对**仓库根目录**，与调用者当前目录无关；支持绝对数据路径。
从别处运行时给入口脚本、配置文件传绝对路径。
`--dry-run` 仅验证配置并输出计划路径，不读取数据、不创建输出、不等于审计或测试通过。
只有 audit 支持 `--limit-samples N`：每个数据集最多检查 N 张，后续阶段沿用该冻结子集；
其报告始终是 diagnostic，不能混入正式全量结果。

退出码：0 表示该阶段完成；2 表示配置/输入错误、审计阻断、模型部分失败或正式报告阻断。
audit 即使阻断仍保存审计报告；evaluate 即使部分失败仍保存所有样本行和计数。
每个阶段/每个模型只允许写一次；失败后保留产物供定位，修改输入后使用新的 run_id。
不要删除错误行或补写旧结果。模型列表、元数据也必须在 audit 前配置好。

## 数据契约

### VKITTI2

把 `rgb_root` 指到直接含 SceneXX 的 RGB 根目录，例如本机已存在的
`F:/vkitti/vkitti_rgb/rgb`。把 `class_seg_root` 指向直接含 SceneXX 的语义标注根目录。
以下文件须对应同一 frame、camera、variation、scene：

```text
<rgb_root>/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
<class_seg_root>/Scene01/clone/frames/classSegmentation/Camera_0/classgt_00000.png
<class_seg_root>/Scene01/clone/colors.txt
```

从实际 `colors.txt` 读取类别，验证 Sky=(90,200,255)，Pillow 按 RGB 读取并逐通道精确匹配。
所有未知颜色直接报错，不做容差匹配，也不会把未知类别默认为非天空。
`train_list`/`test_list` 使用原仓库 RGB 路径格式（允许后面有 depth 列），
按 Scene/variation/camera/frame 归一化；重复键和空名单拒绝。
Scene 有交集且 `allow_scene_leakage=false` 时 prepare 被阻断。
如用户明确接受 seen-scene 诊断，需在新实验配置中显式设 true；正式报告仍被阻断。

### HRWSI

不猜测未取得数据包的文件名或 mask 编码。请根据实际下载包制作输入 JSONL，
一行一个 Sample，RGB/mask 相对 `root`：

```json
{"sample_id":"hrwsi/example","dataset":"hrwsi","split":"curated_evaluation_subset","rgb_path":"rgb/example.jpg","sky_mask_path":"sky/example.png","mask_source":"author_pseudo_semantic_sky","split_provenance":"实际来源及划分说明"}
```

输入 manifest 不需要尺寸和天空占比；audit 从实际文件计算并冻结。
必须显式配置 `sky_values` 和 `non_sky_values`，例如只有确认实际编码后才可写 `[255]` 与 `[0]`。
不支持含未知/ignore 值的 HRWSI mask：先确定该标注协议，再扩展读取器，不静默把它当背景。
mask 必须是单通道，与 RGB 尺寸一致。拒绝把 validity、VGC、DA2 支持集作为语义来源。
HRWSI 作者方法产生的天空标签可能是伪语义标注，应在 `mask_provenance` 写明来源与限制。
立体估计及天空后处理得到的 depth 不作为这里的独立天空 GT。
正式报告前人工审查至少 50 张，覆盖低云、树冠、建筑、窗户、极薄天空；填写
`qa_reviewed: true` 和 `qa_review_count`。自动生成 50 张图不能代替人工审查。

audit 输出 RGB 和 mask 文件 SHA256 重复组；重复 RGB 会阻断 prepare，先在模型打分前
修改原始候选名单并重审。相同 mask 可能合法，仅报告。
字节哈希无法证明没有重编码或近重复图；跨数据集人工审查后才可设置
`cross_dataset_duplicate_reviewed: true`。

## 模型预测元数据

配置在 `models` 下，以下为格式模板，字段值需根据实际推理记录填写；它不是可用模型记录：

```yaml
models:
  celeste_full:
    root: /path/to/predictions
    layout: explicit
    files:
      vkitti2/Scene01/clone/Camera_0/00000: scene01/clone/camera0/rgb_00000_pred.npy
    checkpoint: /path/to/checkpoint
    inference_command: '实际完整推理命令'
    git_commit: '生成预测时的 commit'
    seed: 2024
    resolution: original_full_frame
    denoise_steps: 50
    ensemble_size: 1
    scheduler: '实际 scheduler 路径/类型/哈希'
    inference_config: '实际配置路径/哈希；含 conditioning、DA2 等开关'
    native_postprocessing: '例如 pipeline 自带 clip [0,1]，与实际实现一致'
    representation: relative_depth
    distance_direction: larger_is_farther
    semantics_source: '具体源码/接口约定及版本'
    amplitude_comparability_verified: false
    geometry: full_frame
    resampling: reject
    squeeze_leading_channel: false
    training_evidence:
      vkitti_train_list: /path/to/actual_training_list.txt
      train_list_sha256: '训练时名单快照的 SHA256'
      training_record: /path/to/actual_training_record.txt
      training_record_sha256: '训练记录的 SHA256'
      confirms_list_snapshot: false
```

`explicit` 的 `files` 必须覆盖冻结集合；缺少映射会记录失败。
`run` 自动解析 `<root>/depth_npy/<stem>_pred.npy`。
`infer_vkitti` 自动解析 `<root>/<rgb relative parent>/pred_<rgb stem>.npy`；
如果旧输出父目录多了 `rgb/` 前缀，调整 root 或用显式映射。
新增 `infer_split` 用于 VKITTI2/KITTI/DIODE，解析
`<root>/<dataset>/<rgb relative parent>/pred_<rgb stem>.npy`，保持不同数据集的预测隔离。
只接受 float32/float64 二维 `.npy`；只有配置 `squeeze_leading_channel: true` 才允许 `[1,H,W]`。
NaN/Inf 在整张图上检查，不丢掉困难像素。输入 PNG、整数深度、任意其他维度一律拒绝。

`smaller_is_farther` 只执行负号翻转以统一排序方向，不等于物理倒数变换。
不同 depth/disparity/latent-depth 表示的 IQR/Wide 不能直接比较。
只有所有模型 representation 相同且均显式核实 `amplitude_comparability_verified: true`，
报告才输出跨模型 IQR/Wide；否则保留各模型独立成绩，比较表幅值为 N/A，只比较 SFOE。
需要非线性转换时请在有依据的独立导出步骤完成，并记录原/目标表示、公式、无效值策略；
本工具不会自行猜测转换或给天空置常数。

正式报告会逐模型检查实际训练名单与测试名单的 scene 交集、训练记录和哈希。
对确实完全未使用 VKITTI 的外部模型，可提供 `training_record`、`training_record_sha256`、
`no_vkitti_training: true`、`reviewed: true` 作为人工核实的训练来源证据；
这些开关是明确声明，不代表工具能自动证明模型预训练数据独立。

## 冻结的数学协议

设 `S` 为语义天空，`L=min(H,W)`，`D` 已统一为越大越远。
固定天空原始占比 `sum(S)/(HW) >= 0.05`；全天空因没有参考前景单独排除。
不使用 depth GT、模型输出或 validity 来筛图。

$$S_{int}=\{i\in S:\operatorname{dist}(i,\bar S)\ge\lceil0.01L\rceil\}.$$

距离是图像内像素中心之间的 Euclidean EDT。图像外不当作前景，故保留图像顶边的天空。
采用包含等号的边界规则；无非天空/无天空时单独处理。

$$R=Q_{95}(D[\bar S])-Q_5(D[\bar S]),\qquad
C_{IQR}=\frac{Q_{75}(D[S_{int}])-Q_{25}(D[S_{int}])}{R},\qquad
C_{Wide}=\frac{Q_{95}(D[S_{int}])-Q_5(D[S_{int}])}{R}.$$

全部 percentile 使用 NumPy `method='linear'`，内部 float64。
`R=0` 或非有限时标记退化，不添加 epsilon。没有 interior 则一致性分数为空。
分别输出 R 和其分布，不能掩盖通过扩大前景动态范围改善归一化指标的问题。

SFOE 每侧半径 `r=ceil(0.02L)`，内侧偏移 `e=ceil(0.01L)`。
固定实现规则：

1. 提取所有四邻域 sky/foreground 类别跳变边的**中点**，按 `(y,x)` 排序，每隔两个候选取一个。
   这是确定的栅格候选序列抽样，不声称等弧长轮廓采样。
2. 有符号场 `EDT(~S)-EDT(S)` 在前景为正。中心差分梯度在线性插值后的中点处归一化，法线指向前景。
   零梯度以及朝向对应跳变边的投影小于 0.5 的歧义点跳过。
3. 沿正/负法线在整数 `t=e,...,r` 采样；像素坐标取最近中心，恰好半像素的平局朝边界中点舍入，
   避免统一向上舍入使前景侧偏移一像素。取样后的 EDT 距离须处于 `[e,r]`，否则保守丢弃该点。
   额外用每半像素一处的半射线检查（包括未计分的内侧 gap），要求两侧始终属于正确类别且不越界，
   避免穿过孔洞/细枝后又回到目标类别。v1 对整条半射线有污染的候选采取保守丢弃。
4. 每点两侧分别取预测中位数。相等视为排序失败。

$$\operatorname{SFOE}=\frac{1}{|P|}\sum_{p\in P}\mathbf1[\operatorname{median}D_s(p)\le\operatorname{median}D_f(p)].$$

最少 50 对，覆盖率写入 `pair_coverage.csv`。少于 50 不抹去合法 IQR/Wide。
1% erosion、1% 内侧偏移、50 对是设计默认值，真实保留率审查尚未执行。
正式打分前若需要调整，提交研究设定变更并新建 run；不能看过成绩后优化阈值。

各指标最后逐图等权平均，不按天空面积加权。
跨模型每个数据集/每个指标分别取所有配置模型的有效 sample_id 交集，导出交集 ID、N、缺失数。
`max_failure_fraction=0` 是保守默认：正式表不容许任何指标缺失；提高阈值必须在 audit 前预注册。
不计算跨数据集混合总分。

IQR/Wide 对正仿射变换不变，对倒数变换并不保持不变；SFOE 不能证明前景米级精度。
常数近天空可以获得 IQR=0，但会在排序中失败。
约 10% 局部塌陷可能只影响 Wide；低于约 5% 的异常可能连 Wide 都漏掉。
扩大非天空动态范围可能压低归一化值，不代表天空质量改善。

## 输出与防覆盖

```text
output/sky_eval/<run_id>/
  config_resolved.{json,yaml}, environment.json, repository_state.json
  dataset_audit.{json,md}, mask_stats.csv, audit_lock.json
  candidate_manifest.jsonl
  manifest.jsonl, pair_coverage.csv, protocol_lock.json
  sample_pairs/<sample_id hash>.npz
  mask_diagnostics/<sample_id hash>.png
  models/<model_id>/
    prediction_manifest.json, per_image.{csv,json}, metrics.json, result_lock.json
  diagnostic_report/
    report_status.json, comparison.{csv,json,md}, comparison_wide.csv
    diagnostic_table.tex, diagnostics/*.png
  formal_report/                    # 只有显式 --formal 才创建
    BLOCKED.md 或 comparison.* + paper_table.tex
```

相对原提案把比较文件放入 `diagnostic_report` / `formal_report`，目的是明确区分诊断和正式输出，避免误用。
正式 TeX 为 Method + HRWSI/VKITTI2 三项指标；长格式 CSV 同时给 N、缺失数和具体交集。
mask QA 显示 RGB、语义区域、内部、两侧带、法线；黄色端点表示前景。
报告挑选首个模型的低/高 SFOE 案例，同图各模型共享完整图的显示范围，白色代表远。
不同表示时共享数值显示仅供检查，不能当成跨模型幅值证据。
图片只供 QA，不用于定量指标。

配置、源码、名单、审计结果、manifest、pair cache 改变会触发拒绝；报告还核验评估结果和原预测哈希。
不允许一个模型悄悄少评样本，也不接受修改 CSV 后导出论文表。

## 后续验证（本轮未运行）

```powershell
& $scoePython -B -m unittest discover -s tests/sky_eval -v
```

已编写的测试覆盖正仿射不变性、稳定远天空/常数近天空/全图常数、局部塌陷、分母投机、
天空占比、空 interior、边界不足、方向/孔洞/顶边处理、缓存确定性、未知语义色、
官方 palette 的 RGB/BGR 合成回归、场景泄漏、输入 dtype/shape/有限性、逐图等权、共享交集、
四张人工数据的两模型 CLI 流程、CSV/JSON/TeX 一致性、防覆盖、缓存篡改和 dry-run 无写入。
官方真实 VKITTI 样例回归仍待语义数据到位；palette 合成样例不能代替它。

交付之后仍需：运行这些测试并修复可能问题；取得真实语义数据及预测；核实训练独立性；
完成 mask QA 和配对保留率审查；然后才运行正式评估与表格导出。
