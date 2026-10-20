# SCOE 开发日志

## 2026-10-19：P0 完成（10%）

| 阶段 | 权重 | 状态 |
|---|---:|---|
| P0 源码审计 | 10% | 完成 |
| P1 真实数据与划分 | 20% | 名单静态审计已完成；真实数据审计待补充（计 5%） |
| P2 掩码与配对 | 20% | 代码、缓存、QA 图实现；运行验收待执行（计 15%） |
| P3 指标与汇总 | 20% | 代码与性质测试用例已实现；未运行（计 15%） |
| P4 CLI 与导出 | 15% | 四阶段入口及导出已实现；端到端验收待执行（计 10%） |
| P5 测试与文档 | 15% | 文档及测试文件已交付；未执行测试（计 5%） |

按原任务含真实数据与验证的完整交付口径，保守总进度约 **60%**；
这不是测试通过率，也不是已完成实验的比例。本轮用户缩小为先交付实现，暂不测试。

初始 commit：`02e2571257fbf2bfcc4655e183793255023a3746`，分支 main。
初始非论文工作区变更：用户已有 `src/__pycache__/__init__.cpython-312.pyc` 修改和未跟踪 `agent.md`，保留。
全部搜索、读取、输出排除 `ApDepth-G-paper`，不读取其中内容。

已运行：限定路径的 `git status --short`、`git diff --stat main`、`git rev-parse HEAD`；
阅读 README、run/infer/eval、dataset、util、配置和已有推理/评估测试。
Python 标准库审计仓库 VKITTI 名单：train=20148（Scene02/06/18/20），val=5364（Scene01）；
scene、精确 scene/variation/camera/frame 交集均为空。训练/验证均含两个 camera、六个 variation。
`F:/train_marigold/config.yaml` 引用相同相对训练名单，但未提供当时名单快照或目标 checkpoint，故不能证明目标模型独立。
扫描 `F:/vkitti` 未找到 `colors.txt`、`classgt_*.png`；有 RGB/depth 和单独权重文件。
已检查 F 盘常见数据目录，未定位 HRWSI，不能推断整机不存在该数据集。
默认 Python 缺 numpy；`D:/Anaconda/envs/DL/python.exe` 已有 numpy/scipy/Pillow/PyYAML，可用 unittest 测试，无需安装。
本阶段未运行数值测试。

### 实现路径映射（一页内）

| 原接口 | 复用方式 / 限制 | 新模块 |
|---|---|---|
| run.py | `depth_npy/<RGB stem>_pred.npy`；多场景同名帧必须分目录运行或显式映射，防碰撞 | predictions.py |
| infer.py | 保留 RGB 相对父目录；VKITTI id 模式为 `pred_rgb_00000.npy` | predictions.py |
| VKITTI dataset | 原配置启用 benchmark crop，SCOE 必须用完整 RGB 坐标；拒绝把裁剪预测拉伸成全图 | datasets.py、schemas.py |
| eval.py | 含 GT 对齐和裁剪，不调用其处理后预测 | metrics.py、aggregate.py |
| model_overrides.py | 本地仅接受 12ch U-Net，必须匹配 scheduler 和 inference_settings/config；不假定已支持 8ch | 预测元数据记录 |
| marigold_pipeline.py | 输出前 clip 到 [0,1]；记录 native postprocessing，不再额外 clip，不宣称未截断 latent | 预测元数据记录 |
| data_split/config | 审计文件哈希、scene 和精确帧交集；目标训练证据另行提供 | audit.py |
| 新入口 | audit → prepare → evaluate → report，CPU 离线、冻结配置/数据/配对、失败计数 | script/sky_eval.py、src/sky_eval/ |

不更改旧训练、推理、eval、README 或用户已有文件；使用独立中文指南。

## 2026-10-19：P1–P5 实现交付，运行验证暂缓

用户补充：所需语义标注、HRWSI、目标预测及训练记录均未准备，要求“先不用测试，先完成这些东西”。
按此要求完成实现与测试文件，**没有运行单元测试、性质测试、CLI 烟雾测试、真实数据 evaluator 或模型推理**。
没有下载数据集、模型或依赖，没有提交、推送。

新增实现：

- `schemas.py`：协议和 manifest 类型、相对路径约束、哈希与拒绝覆盖写入。
- `datasets.py` / `audit.py`：显式 HRWSI manifest、VKITTI 官方 RGB palette 校验、未知标签拒绝、RGB/mask 对齐、
  5% 筛选、全天空排除、精确帧/场景交集、字节重复检查、逐样本失败记录。
- `masks.py`：内部 EDT、逐侧 2% 候选带、法线朝前景、模型无关栅格采样与缓存、半射线类别检查。
  静态审查发现统一向上舍入会让前景侧偏移一像素，改为半像素平局朝边界中点舍入，并检查实际 EDT 取样带。
  这是一项未跑模型前的离散实现修正，未改 5%/2% 参数。
- `metrics.py` / `aggregate.py`：IQR/Wide/SFOE、分母退化、指标独立缺失、逐图等权、固定共同交集；
  对偶数中位数采用先除二再相加，避免有限大数求和溢出。
- `predictions.py`：原 run/infer 命名与显式映射；拒绝命名碰撞、非法 dtype/shape/NaN/Inf；
  明确方向、原生后处理、完整图像坐标；不执行重采样或非线性表示转换。
- `workflow.py` / `script/sky_eval.py`：audit/prepare/evaluate/report、dry-run、limit、退出码、run/model 隔离；
  冻结配置、源码、审计、名单、数据、pair cache 与结果，阻断过期结果或数据漂移。
- `visuals.py`：掩码/内部/候选带/法线 QA，低/高排序错误案例及共享显示范围模型对比。
- `config/sky_eval.yaml`：默认 5%/1%/2%/1%、50 对、linear、禁止场景泄漏；真实路径留空。
- `doc/sky_eval.md`：中文指南、公式、模型元数据、复现命令、数据依赖及当前限制。
- `tests/sky_eval/test_scoe.py`：17 个 unittest 方法，包括四张人工 RGB/mask/float32 预测的两模型端到端流程。
  官方 palette 回归是合成样例，真实官方标注回归仍待数据到位。

实际执行检查：

1. 使用 Python 标准库 `ast.parse` 对 12 个新增 Python 文件做语法解析；通过。
2. AST 检查新增函数的参数和返回类型注解；未发现缺失（self/cls 除外）。
3. 新增 Python 文件尾部空白静态扫描；通过。
4. 限定目录的 `git diff --check`；无输出。它不覆盖 untracked 新文件，所以另做上述空白扫描。
5. 限定目录的 `git status --short`；原 tracked 变更仍仅为用户已有 pyc，本轮其余均为新增文件。
6. 用标准库重新统计仓库 train/val 名单并写 `doc/sky_eval_repository_audit.json`，包含名单哈希和静态检查状态。

未使用测试结果支撑任何数值正确性声明。语法解析通过不等于运行或数值验证通过。

仍待完成：用户允许后运行测试并修复问题；补齐真实语义标注与 HRWSI；目标 checkpoint 训练历史核验；
真实 mask 至少 50 张 QA、配对保留率审查、模型输出表示审查及全量评估。
正式表格目前无数据；默认配置中 models 为空，不编造论文结果。

## 2026-10-20：Mask2Former 伪标签分割入口与三数据集接入

用户授权新增研究流程：使用 Mask2Former 分割 VKITTI、KITTI、DIODE 评估集天空，
从 BASE_DATA_DIR 读取文件，提供项目代码及 README 运行说明。
本次代码/文档编写进度 **100%**；运行验收 **未开始**，沿用用户“先不用测试”的要求。
这不更新此前原始 HRWSI+官方 VKITTI 协议的真实实验完成率。

| 子阶段 | 状态 |
|---|---|
| 本地数据配置、目录/tar 与 RGB 相对路径核验 | 完成源码阅读 |
| 离线 Mask2Former 分割、mask/manifest/元数据/QA 输出 | 完成实现，未运行 |
| SCOE binary_manifest、tar RGB、三数据集诊断导出 | 完成实现，未运行 |
| README、全图深度推理配置生成、模型记录模板 | 完成 |
| 静态检查 | 通过语法/注解/空白检查 |
| 模型推理与测试 | 未运行；未下载模型/数据集/依赖 |

起始 commit：`5b98fda78bcd846981b631cd501a919f66434383`；限定非论文目录的初始 git status 干净。
读取了 `config/dataset/data_vkitti_val.yaml`、KITTI Eigen test、DIODE all 配置、对应名单和 dataset/infer 实现。
确认这些数据集均以名单第一列读 RGB，现有 KITTI/VKITTI 默认使用 benchmark crop。
因此分割固定使用完整 RGB，另导出 `depth_inference.yaml` 关闭 crop，复用现有 `infer.py`，不改常规 benchmark 配置。

新增：`script/segment_sky.py`、`config/sky_segmentation.yaml`、
`config/sky_eval_mask2former_models.example.yaml`、`src/sky_eval/rgb_io.py`、
`src/sky_eval/segmentation.py`、`tests/sky_eval/test_sky_segmentation.py`。
更新 README、SCOE 指南与 sky_eval 的数据/审计/预测/可视化/报告适配。

实现约定：

- BASE_DATA_DIR 下按原配置与明确回退目录寻找 RGB，直接读 tar，不解压、不读 depth GT。
- VKITTI2 使用仓库 Scene01 val，KITTI 用 Eigen test，DIODE 默认全部发布 val（含室内）；保留各自实际划分名称。
- 本地完整 Transformers checkpoint，强制 `local_files_only=True` 和离线环境；阻止嵌套 pretrained-backbone 下载。
- 从 `id2label` 查 sky；使用 processor 的 semantic postprocess 还原到原始 H/W；0/255 二值 PNG。
- 无天空也输出，分割失败记录每张图；存在失败时不导出 SCOE 配置。
- 每张图原始 RGB/mask 哈希、名单/权重/processor/源码版本留档；同一套伪标签供所有深度模型使用。
- 显式标记 Mask2Former pseudo labels，诊断表支持 VKITTI2/KITTI/DIODE，原官方标签主表的 formal 检查保留。

已阅读核对的接口参考（仅网页说明，未下载权重/数据）：
[Transformers Mask2Former](https://huggingface.co/docs/transformers/en/model_doc/mask2former)、
[ADE20K 模型说明](https://huggingface.co/facebook/mask2former-swin-large-ade-semantic)。

实际检查：标准库 AST 解析、函数参数/返回注解扫描、尾随空白扫描、限定文件 `git diff --check`。
新增 6 个离线 fixture 测试方法，覆盖目录/tar 一致性、重复归档键、BASE_DATA_DIR 选路、sky 类别编号查找、
零天空输出、SCOE 读入、全图推理配置、预测命名与失败不导出；**均未执行**。
未调用真实 Mask2Former、未运行 CLI dry-run 或模型推理，也未修改 `ApDepth-G-paper` 或 `eval.py`。

### 2026-10-20 补充：显式 data_split 与数据集目录内掩码

用户明确要求使用其现有 split，通过 `--data_split` 传入，并与后续 eval 共用；掩码直接生成在数据集文件夹内。
本次实现/文档调整进度 **100%**，运行测试仍未开始。

- `segment_sky.py --data_split` 为必填，可接收多个名单，与 `--datasets` 按顺序一一对应。
  同时支持单数 `--dataset`，标准 `data_split/<dataset>/...` 路径可自动识别数据集；任意自定义路径需明确指定 dataset。
- 传入名单覆盖 dataset YAML 的 filenames；不修改原始文件。split 的额外列在快照中保留。
- mask 改为 `<RGB目录>/sky_masks/<output_dir最后一级>/...sky.png`；tar 数据改为其所在目录下的 `sky_masks`。
  项目 `output/sky_segmentation/<运行名>/` 仅保存索引、日志、预览和生成配置。
  输出目录和数据集内掩码运行目录均拒绝覆盖，支持换运行名重做。
- `depth_inference.yaml` 的 filenames 直接引用用户原始 split；若限量或原名单含空行/注释，
  则为兼容旧 loader 使用相同有效行/顺序的快照。SCOE 中的 data_split 始终保存原始名单路径。
- eval 核对原始 split 哈希、总数、图像集合与顺序，并将名单加入 audit 冻结检查，阻止换名单后复用旧掩码。
- README 的单数据集、三数据集、PowerShell 和后续 eval 命令均同步更新。

已执行：标准库 AST、类型注解和空白静态扫描（6 个变动 Python 文件），分割 YAML 解析，限定路径 git diff --check。
检查通过；Git 仅提示部分文件下次操作会将 LF 转为 CRLF，未执行行尾转换。
新增 split 覆盖/自动识别/参数数量、数据集内落盘、原名单引用和 eval 拒绝变更名单的测试用例；未运行。
初始工作区存在上轮已暂存修改，保留原暂存状态；本轮未提交或重新暂存。
没有下载模型/数据集，没有实际生成掩码，没有访问论文目录。

### 2026-10-20 补充：VKITTI 540 张稀疏名单

用户最新明确要求：**仅从原 val 抽样**。已撤销此前多场景诊断候选，删除其名单及 YAML，
并从生成脚本、审计 JSON 和使用说明中移除该选项。
当前 split 与文档调整进度 **100%**；没有执行模型/指标测试。

仅保留 `vkitti_val_sampled_540.txt`：6 variations × 2 cameras × 45，共 540 行，
全部逐行来自原 `vkitti_val.txt`，与仓库 train 的帧和 scene 交集均为零。
保留原始 train/val 以及该 540 张子集的内容，未启用场景泄漏、未改变默认 val 配置。
抽样脚本只以 val 为候选来源；读取 train 仅用于交集审计。

已更新 `data_split/vkitti/README.md`、README 入口和 val-only 审计记录；
提供 `config/dataset/data_vkitti_val_sampled_540.yaml` 及 Linux 分割/复现命令。
输出 SHA256 保持 `f3410f16f7597ddfd8f022079719d56038741727a66c4cf08e342739845ee947`。

检查：540 行、无重复、全部属于原 val、无训练帧/scene 交集、来源 train/val 哈希不变，
脚本静态语法和 YAML 引用检查。没有读取图像、依据预测筛图、下载、提交、模型推理或访问论文目录。
