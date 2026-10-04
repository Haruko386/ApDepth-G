# ApDepth-G：VGC 之后的天空塌陷修复尝试记录

> 2026-10-04 更新：第 14 节记录 `residual_snr` 主训练失败。用户实测天空比原方案更差，
> 城市场景中天空被预测为近处，并与远处山体/树林形成明显反向层次。该补全项已从主训练移除。
> 新实验转向扩散端点分布对齐，不再增加天空或 invalid region 监督，见 `doc/terminal_snr_alignment.md`。

> 2026-10-01 后续决定：保留 decoder calibration 后训练，尚未收到本方案失败反馈。
> 当时新增的噪声感知先验残差补全后来已失败，见第 14 节。
> 用户反馈目前 VGC U-Net 的天空塌陷更严重，因此原 VGC 不再视为已验证解决方案；
> 以下“相对有效”等表述保留为历史记录，新主训练效果同样待验证。

> 2026-10-01 更新：第 13 节记录语义关系主训练的失败及天空远近反转。
> 用户现已明确重新选择后训练；这覆盖第 12 节当时“不再后训练”的阶段性决定。
> 新候选为冻结原 VGC 去噪器、缓存完整多步最终 latent 后校准深度解码器，见 `doc/decoder_calibration.md`。

> 2026-09-24 更新：第 12 节记录去噪轨迹后训练在 1000 步的失败反馈。
> 用户决定不再继续后训练方向；后续使用 SD2 初始化的完整主训练，并保留多步推理。

> 2026-09-23 更新：新增第 11 节，记录“已知远端主扩散监督恢复”的实际失败结果。
> 本文件是实验记录，不代表执行指令；第 7 节 SIPR 的状态是历史记录。

> 本文档用于记录 ApDepth-G 在 **VGC（Validity-Guided Completion）之后**，围绕户外场景中 **sky / far-field depth collapse** 问题尝试过的创新点、当时的设计动机、实现思路和最终状态。
> 这些内容主要作为研发日志和后续论文整理参考，**不代表当前最终方法**。其中已经明确失败或被移除的方法，不应继续作为论文贡献点使用。

---

## 0. 背景：VGC 是目前最后一个相对有效的阶段

在后续一系列天空修复方案之前，VGC 是目前比较明确取得过正向结果的一版。

### VGC：Validity-Guided Completion

核心问题是：

- 户外训练数据中，天空、超量程区域、无有效 LiDAR / depth 标注区域经常被 `valid_mask` 排除；
- 主 diffusion reconstruction loss 只对有效深度区域监督；
- 因此天空和远景区域实际上缺少直接训练信号；
- 网络容易在这些区域出现：
  - 天空被预测得过近；
  - 大片远景塌陷；
  - 低纹理区域产生伪结构；
  - DA2 prior 的错误纹理被扩散模型进一步放大。

VGC 的做法比较保守：

1. 找到 `invalid depth` 区域；
2. 只在 invalid ratio 足够大的样本上启用；
3. 使用 DA2 latent 作为弱几何参考；
4. 不逐像素强制复制 DA2，而是只对 invalid region 的整体 latent response 做弱 anchor；
5. 对 invalid 区域增加平滑约束；
6. 不改变原有 12-channel 输入结构。

典型形式：

```text
Predicted clean latent
        │
        ├── Region-level weak DA2 anchor
        │
        └── Invalid-region smoothness
```

曾经有一版在 DIODE 上出现过大致：

```text
AbsRel: ~8.5 → ~7.8
```

的改善，因此 VGC 后来一直被保留为基础模块之一。

---

# 1. AFG — Ambiguity-aware Far-field Geometry

## 当时的动机

VGC 虽然能给 invalid region 一些训练信号，但它并不知道：

> 哪些 invalid region 真的是远景 / 天空，哪些只是普通缺失区域。

所以当时想进一步引入一种 **“歧义感知的远景几何约束”**：

- 对模型不确定、缺乏 GT、同时可能属于 far-field 的区域增加额外约束；
- 希望比单纯 invalid mask 更有针对性；
- 尽量避免把 indoor missing pixels 错当作天空。

## 当时的设计思路

大致想法是：

```text
Invalid region
      │
      ↓
Ambiguity / confidence estimation
      │
      ↓
Far-field candidate selection
      │
      ↓
Geometry regularization
```

希望利用：

- GT valid / invalid 分布；
- DA2 prior；
- prediction 与 prior 的差异；
- 局部几何一致性；

来判断哪些区域属于“高歧义远景”。

## 实际结果

这一方向结果非常差。

曾出现类似：

```text
NYU error:
~4.5 → ~20
```

的灾难性退化。

这说明该约束明显干扰了模型原有的 indoor depth learning。

## 当时推测的失败原因

### 1. far-field 判定过强

模型并不能可靠地区分：

```text
invalid sky
```

与：

```text
普通 missing depth / object boundary / indoor invalid region
```

导致辅助 loss 被错误地施加到大量正常区域。

### 2. 新 loss 梯度过强

AFG 不只是“修补无监督区域”，而是开始明显影响主干表示。

最终出现：

```text
天空可能没完全解决
+
正常 indoor geometry 被破坏
```

### 3. 使用 prior 的方式过于激进

如果 DA2 本身在局部有错误，则该错误可能被 AFG 当作 far-field geometry signal 放大。

## 最终状态

**完全放弃。**

```text
Status: FAILED / REMOVED
```

后续不再作为候选创新点。

---

# 2. HFP — Horizon-Far Prior

## 当时的动机

观察户外图像时，一个很自然的经验先验是：

> 图像上半部分、更接近地平线和天空的区域，通常对应更远的场景。

因此尝试引入一个非常轻量的 **Horizon-Far Prior**。

希望做到：

- 不额外引入 segmentation model；
- 不改变 U-Net 输入；
- 只通过图像位置先验帮助远景稳定；
- 对上方区域提供一个弱的 far-field bias。

## 当时的基本想法

类似：

```text
Image vertical position
        │
        ↓
upper / horizon region
        │
        ↓
weak far-depth prior
```

即：

```text
越靠上
→ 越倾向于远
```

但并不是简单把上半幅图全部设为天空，而是希望通过 soft weighting 做弱约束。

## 实际结果

**基本没有效果。**

既没有明显改善天空塌陷，也没有带来稳定的 outdoor metric 增益。

## 当时推测的失败原因

### 1. “上方 = 远景”本身太弱

真实图像里：

- 高楼；
- 树；
- 山体；
- 室内墙面；
- 仰视物体；

都可能占据图像上半部分。

因此纯 vertical prior 的语义信息太弱。

### 2. 没有解决根本监督缺失问题

天空塌陷最关键的问题还是：

```text
sky invalid
→ no GT supervision
```

位置先验只是给了一个弱 bias，并没有真正构造稳定的训练目标。

## 最终状态

```text
Status: NO EFFECT / REMOVED
```

不再保留。

---

# 3. SFO — Sky-Foreground Ordinal Ranking

## 当时的动机

固定要求天空等于某个 depth 值可能太强。

因此尝试把问题改成一个更宽松的 **ordinal relation**：

> 不要求 sky depth 精确是多少，只要求 sky 比 foreground 更远。

这比：

```text
sky = maximum depth
```

理论上更柔和。

## 当时的核心想法

构造：

```text
Sky region
    +
Foreground reference region
        │
        ↓
ordinal ranking constraint
```

例如希望：

```text
D_sky > D_foreground + margin
```

在当前归一化定义下，可以写成类似：

```text
relu(D_foreground + margin - D_sky)
```

目标只是防止：

```text
sky 被预测到 foreground 前面
```

而不是强制天空一定等于 +1。

## 当时认为的优点

- 比 absolute sky target 更柔和；
- 理论上更适合 affine-invariant depth；
- 可以直接针对“天空变近”这一错误；
- 不一定需要真实 sky metric depth。

## 实际结果

最终没有观察到足够稳定的收益。

天空问题仍然存在，同时整体指标没有形成明确改善。

## 当时推测的失败原因

### 1. ordinal relation 太稀疏

只告诉模型：

```text
A 要比 B 远
```

但并没有告诉模型：

```text
到底应该远多少
```

训练信号可能不足。

### 2. foreground reference 不稳定

不同图片中用于比较的 foreground depth 分布变化很大。

如果 reference 选得不好，ranking loss 的意义就会变弱。

### 3. diffusion latent 与 pixel ordinal supervision 之间存在间接性

单步训练时的 `pred_x0` 本身具有 timestep-dependent uncertainty。

简单 ordinal loss 很难稳定地作用到所有 timestep。

## 最终状态

```text
Status: FAILED / REMOVED
```

SFO 已从当前训练方案中删除。

---

# 4. Sky-Aware / Sky Mask 额外输入通道

## 当时的动机

既然问题集中在天空，那么最直接的办法就是：

> 把 sky mask 本身作为 condition 输入 U-Net。

原本 12-channel：

```text
RGB latent          4
DA2 depth latent    4
Noisy depth latent  4
----------------------
                   12 ch
```

曾尝试增加：

```text
Sky mask            1
```

形成类似：

```text
13-channel conditioning
```

## 当时的设想

网络显式知道：

```text
这里是天空
```

以后，可以在去噪时主动采用不同的几何策略。

理论路径：

```text
RGB
DA2 prior
Noisy depth
Sky semantic mask
       ↓
     U-Net
```

## 后来为什么放弃

这种方法的问题在于它改变了模型本身的输入空间。

### 1. 训练 / 推理依赖额外 sky segmentation

这使系统从：

```text
RGB → Depth
```

变成：

```text
RGB
+
Sky segmentation
→ Depth
```

推理链条更复杂。

### 2. 容易形成 shortcut

网络可能直接学习：

```text
sky mask = far
```

而不是从 RGB / geometry prior 中真正学习稳健的远景结构。

### 3. 与原本 12-channel pipeline 不一致

当前 ApDepth-G 的一个核心结构就是：

```text
RGB latent
+
DA2 latent
+
Depth latent
```

固定 12 channel。

增加第 13 channel 会让训练和已有 pipeline、checkpoint、架构图都复杂很多。

## 最终状态

```text
Status: ABANDONED / REMOVED
```

当前模型重新保持固定 12-channel。

---

# 5. PFR — Prior-guided Far-field Rectification

## 定位

PFR 和前面几个不完全一样。

它并不是主要训练创新，而是后来考虑过的 **inference-time post-processing**。

## 当时的动机

如果训练模型偶尔还是会出现：

```text
DA2 认为天空很远
但 diffusion prediction 把天空拉近
```

那么可以在最终输出后利用 DA2 prior 做一次轻量纠正。

## 当时的基本思路

```text
Predicted Depth
      +
DA2 Prior
      ↓
Far-field candidate detection
      ↓
if predicted region is abnormally near
      ↓
soft rectification
```

特点：

- 不改 U-Net；
- 不需要重新训练；
- 可以作为 pipeline patch；
- 默认关闭，只在 outdoor preset 中开启。

## 为什么没有作为核心创新继续推进

它的问题非常明确：

> 它修的是输出，不是模型本身。

因此即使视觉上能缓解部分天空错误，也很难证明：

```text
model learned better geometry
```

更像一种 heuristic post-processing。

另外，如果 prior 错了，post-process 也可能引入新错误。

## 最终状态

```text
Status: OPTIONAL / NOT CORE
```

不能和真正 trainable innovation 放在同一个级别。

---

# 6. CGFD — Confidence-Gated Far-field Distillation

## 当时的动机

这是 VGC 之后投入设计比较完整的一次尝试。

核心出发点仍然是：

```text
main diffusion loss
只监督 valid GT
```

所以：

```text
sky / far-field invalid region
缺少直接 supervision
```

VGC 已经证明：

> 给 invalid region 一点弱约束可能是有意义的。

于是 CGFD 想进一步解决 VGC 的一个问题：

> DA2 prior 不一定可靠，所以不能直接拿 DA2 去监督 invalid sky。

因此提出：

**Confidence-Gated Far-field Distillation**

---

## 设计流程

### Step 1：在 valid GT 上校准 DA2

先拟合：

```text
aligned_prior = a × DA2 + b
```

通过有 GT 的区域判断 DA2 在当前图片是否可信。

---

### Step 2：计算 teacher confidence

例如：

```text
fit MAE 小
→ confidence 高

fit MAE 大
→ confidence 低
```

如果 DA2 在有 GT 的区域都不准：

```text
CGFD = 0
```

---

### Step 3：寻找 invalid far-field candidate

要求：

```text
invalid
AND
aligned DA2 says far
```

才认为该区域可能是：

```text
sky / far-field
```

---

### Step 4：Low-pass distillation

为了避免复制 DA2 pseudo-texture：

```text
teacher
↓
large AvgPool

student
↓
large AvgPool
```

只蒸馏低频几何趋势。

---

### Step 5：Collapse-only loss

不是要求 student 完全复制 teacher。

而只在：

```text
teacher says FAR
student says NEAR
```

的时候惩罚。

形式类似：

```text
relu(teacher_far - student - margin)
```

如果 student 已经足够远：

```text
loss = 0
```

---

### Step 6：各种 gate

当时还加入：

- minimum invalid ratio；
- teacher confidence；
- minimum candidate ratio；
- low-pass kernel；
- auxiliary loss cap；
- warm-up；
- late-stage learning-rate adaptation。

原计划大致是：

```text
0 ~ 23k
Base Training

23k ~ 25k
CGFD warm-up

25k ~ 28k
Full CGFD
```

---

## 为什么当时觉得它可能有效

因为它试图同时解决：

```text
VGC 太粗
DA2 又不能完全相信
```

这两个问题。

而且整个方案比较保守：

```text
可靠才教
只教 invalid
只教 far-field
只教 low-frequency
只纠正 near-collapse
```

理论上应该尽量不影响 indoor。

---

## 实际结果

最终用户实际训练后确认：

> **CGFD 没有什么用。**

没有取得比 VGC 更好的稳定结果。

## 推测的失败原因

### 1. CGFD 的 gate 太多

虽然保守，但也可能导致真正参与训练的有效 pixel 太少：

```text
invalid
∩ far
∩ reliable teacher
∩ enough candidate
```

最终梯度过弱。

### 2. DA2 在天空本身仍缺乏可靠绝对监督

即使在 valid region 拟合得很好，也不能证明：

```text
invalid sky region 的 DA2 prediction 就一定正确
```

这是一个根本问题。

### 3. low-pass + collapse-only 进一步削弱梯度

为了安全，引入了大量抑制机制。

结果可能变成：

```text
理论上很安全
实际上几乎学不到东西
```

### 4. late-stage adaptation 空间有限

23k 后模型已经基本形成稳定分布。

后面少量迭代可能不足以真正改变 sky behavior。

## 最终状态

```text
Status: FAILED / REMOVED
```

CGFD 已不再作为后续核心方向。

---

# 7. SIPR — Sky-Infinity Prior Regularization

> 当前状态：**正在实验 / 尚未验证**

SIPR 是 CGFD 失败以后重新设计的一条更直接的路线。

与 CGFD 最大的区别是：

> 不再绕着 invalid region + DA2 confidence 去“猜”哪里是天空，而是直接使用 sky semantic mask。

---

## 设计动机

之前几次失败暴露出一个问题：

```text
invalid region ≠ sky
upper region ≠ sky
DA2 far region ≠ sky
```

如果真正的问题就是：

```text
sky collapse
```

那么不如直接明确知道：

```text
where is sky
```

但又不想把 sky mask 作为第 13 个 U-Net input channel。

因此 SIPR 中：

```text
sky mask
```

只用于：

```text
training supervision
```

而不是 network conditioning。

---

## SIPR Part 1：Sky-Prior Dropout

当前 ApDepth-G 输入有：

```text
DA2 depth latent
```

问题是：

> 如果 DA2 天空本身存在错误纹理，U-Net 可能过度依赖这个 prior。

因此训练时随机弱化 DA2 latent 的 sky region：

```text
DA2 latent
      │
      └── sky region
             ↓
      stochastic attenuation
```

例如：

```text
prior_dropout_prob = 0.25
prior_keep_scale = 0.20
```

目标是让网络不要形成：

```text
DA2 sky says X
→ directly copy X
```

的 shortcut。

---

## SIPR Part 2：Scene-Adaptive Infinity Anchor

不直接规定：

```text
sky = +1
```

而是从当前图片有效 GT 中找：

```text
far tail
```

例如：

```text
95% depth quantile
+
small margin
```

形成：

```text
far_anchor
```

然后：

```text
sky depth < far_anchor
→ penalty
```

如果天空已经足够远：

```text
loss = 0
```

即：

```text
one-sided far constraint
```

---

## SIPR Part 3：Sky Interior Regularization

天空内部理论上应当比较平滑。

但是不能直接平滑整个 sky mask，否则容易影响：

- building boundary；
- tree boundary；
- wire；
- horizon。

因此先对 sky mask 做 erosion：

```text
Sky Mask
   ↓
Erosion
   ↓
Sky Interior
```

只在 sky interior 上计算：

```text
Gradient smoothness
+
Variance regularization
```

目标是减少：

```text
pseudo-texture
局部错误深度块
sky surface fragmentation
```

---

## 为什么这次和前几个方法不同

### 与 HFP 不同

HFP：

```text
根据图像位置猜远景
```

SIPR：

```text
明确 semantic sky
```

---

### 与 SFO 不同

SFO：

```text
sky vs foreground ranking
```

SIPR：

```text
scene-adaptive far lower bound
```

---

### 与 CGFD 不同

CGFD：

```text
invalid
+
DA2 reliability
+
far candidate
+
low-pass distillation
```

SIPR：

```text
explicit sky region
+
simple far prior
```

整体路径更直接。

---

### 与 Sky-Aware 13-channel 不同

Sky-Aware：

```text
sky mask → U-Net input
```

SIPR：

```text
sky mask → training loss only
```

不会改变 12-channel inference architecture。

---

## 当前计划

目前计划将 SIPR 放在训练后期：

```text
0 ~ 19k
Base Training

19k ~ 21k
SIPR warm-up

21k ~ 24k
Full SIPR
```

当前建议仍然保持较低辅助权重，例如：

```text
weight = 0.02
max_loss_ratio = 0.10
```

避免天空专项监督破坏整体 depth quality。

## 当前状态

```text
Status: RUNNING / UNVERIFIED
```

在正式实验结果出来以前：

- 不应称为有效创新；
- 不应写入论文最终 contribution；
- 可以作为当前重点验证方向。

---

# 8. VGC 之后各方案状态汇总

| 方法 | 核心想法 | 结果 / 当前状态 |
|---|---|---|
| **VGC** | invalid region weak DA2 anchor + smoothness | **保留；目前最后一个相对有效的方案** |
| **residual_snr** | invalid region 的 DA2 残差均值与带符号梯度、按 SNR 加权 | **用户实测天空更差且远近反向；已移除，见第 14 节** |
| **AFG** | ambiguity-aware far-field geometry | **失败，NYU 严重退化，移除** |
| **HFP** | horizon / upper-image far prior | **基本无效果，移除** |
| **SFO** | sky–foreground ordinal ranking | **无明显收益，移除** |
| **Sky-Aware 13ch** | sky mask 作为额外 U-Net condition | **放弃，恢复固定 12ch** |
| **PFR** | inference-time DA2 far-field rectification | **仅可选 post-processing，不作为核心训练创新** |
| **CGFD** | confidence-gated DA2 far-field distillation | **训练后无明显作用，移除** |
| **SIPR** | explicit sky infinity regularization | **当前实验中，尚未验证** |
| **已知远端主监督恢复** | VKITTI >=80m 映射到 +1 并恢复主扩散监督 | **用户实测失败，天空出现异常斑点；已移除，见第 11 节** |
| **去噪轨迹后训练** | 先展开两步 DDIM，再纠正中间状态 | **1000 步效果很差；停止并移除，见第 12 节** |
| **DA2 空间语义关系主训练** | 对齐教师与 U-Net 的空间相似矩阵 | **实测不可用，部分天空远近反转；已移除，见第 13 节** |

---

# 9. 从这些失败方案中得到的经验

## 9.1 不要再用“图像位置”代替天空语义

已经证明：

```text
upper region
horizon region
invalid region
```

都不能稳定等价于：

```text
sky
```

后续如果确实针对天空，应优先使用明确的 sky semantic region。

---

## 9.2 不要过度相信 DA2

DA2 对整体 geometry 有很强价值，因此 12-channel prior conditioning 应继续保留。

但是：

```text
DA2 prior ≠ GT
```

尤其：

```text
sky
low texture
far-field
```

区域不能无条件复制。

后续若继续使用 DA2 做额外 supervision，应尽量采用：

- weak constraint；
- region-level statistics；
- dropout / attenuation；
- one-sided constraint；

而不是强 pixel-wise distillation。

---

## 9.3 辅助 loss 越复杂，不代表越有效

CGFD 是一个典型例子。

它加入了：

```text
alignment
confidence
far selection
low-pass
margin
gate
loss cap
warm-up
```

但最终没有形成更好的结果。

说明当前问题可能不需要继续增加更多复杂 gating，而更需要：

> 更准确地定义监督区域和目标。

---

## 9.4 Indoor non-regression 必须作为硬约束

任何 outdoor / sky 优化都必须同时检查：

```text
NYUv2
```

不能出现：

```text
outdoor 稍有改善
但 indoor 明显退化
```

AFG 已经证明这种情况完全不可接受。

---

## 9.5 当前更值得坚持的主线

目前相对清晰的 ApDepth-G 主线仍然是：

```text
1. DA2-guided 12-channel conditioning

2. Multi-resolution Noise

3. Offset Noise

4. Min-SNR Reweighting

5. Latent Gradient Consistency

6. VGC

7. Multi-step Diffusion Refinement
```

SIPR 暂时作为：

```text
Candidate #8
```

单独验证。

如果 SIPR 最终仍然没有超过 VGC，那么后续更合理的选择可能不是继续堆新的 sky-specific loss，而是重新考虑：

- training data composition；
- GT invalid handling；
- DA2 prior representation；
- diffusion timestep curriculum；
- decoder / latent supervision；
- outdoor-specific data augmentation；
- broader scene-level geometry supervision。

---

# 10. 一句话研究记录

> **VGC 之后，围绕 sky collapse 先后尝试了 AFG、HFP、SFO、Sky-Aware conditioning、PFR、CGFD 等方案，但均未取得比 VGC 更稳定的提升；当前新的 SIPR 正在验证，核心思路从“猜测 far-field”转向“显式 sky supervision + scene-adaptive infinity prior”。**

---

# 11. 已知远端主扩散监督恢复（2026-09-23：失败 / 移除）

## 当时的假设与实现

发现 VKITTI 的 `max_depth=80` 会让天空及其他 >=80 米区域退出有效 GT 主损失。
据此假设恢复这些位置的主监督能缓解天空塌陷：

- 从原始深度提取有限且在 [80,655.35] 米内的 `known_far_mask`。
- 在 VAE 编码之前，将这些位置的归一化深度置为 +1。
- 用 `valid | known_far` 扩展 latent 监督，包括近景/远景混合 cell。
- 新增位置使用原扩散目标，单独求均值后乘 0.5。
- 已知远端位置退出 VGC，原量程评测掩码保持不变。
- 建议从原 VGC 的 23000 优化步 checkpoint，以 LR 5e-6 另开 6000 步微调。

## 用户实际反馈

用户训练后明确反馈：方案存在明显问题，天空甚至出现不正常的奇怪斑点。
反馈图为石桥、树林、水面和天空场景，深度图天空未保持稳定的统一远端，
并可见局部不规则色块；用户要求撤销此方案。

未提供该结果对应的准确训练步数、seed、原始浮点深度或量化指标。
因此不虚构“6000 步训练完成”、斑点率、NYU 退化幅度等结果。

**结论：FAILED / REMOVED，不再推荐此方案或其微调 checkpoint。**

## 复盘：已确认事实与待证原因

确认存在量程掩码导致的监督缺口，不等于确认它就是天空塌陷的唯一根因。
此前把它作为优先修复点并没有得到用户实验支持，不能继续称为已解决的问题。

以下仅为可能原因，没有消融实验验证：

1. >=80 米不是天空语义。远建筑、树木和天空共享强制端点监督，可能改变远景结构。
2. 常量像素目标经图像 VAE 编码后，不一定产生常量 latent；局部 mask 也无法隔离
   VAE 感受野的影响。因此 latent 拟合不足以保证解码后天空无斑点。
3. 扩大监督域并改变 VGC 覆盖，会改变训练梯度分布，可能破坏原有有效解。
4. 只修改 GT/掩码，没有处理多步推理时模型不断接收自身误差的训练—推理差异。
   是否确实存在误差累积，需要固定 seed 的中间去噪结果验证。

## 撤销范围与后续约束

删除该实验的配置、工具、测试和说明：`train_sky_finetune.yaml`、
`far_supervision.py`、`audit_far_supervision.py`、`test_far_supervision.py`、`doc/sky_fix.md`。
恢复数据集、训练目标、主监督掩码和 VGC 区域；撤回该轮附带的 loss/入口改动。
保留用户原有 batch、VGC 配置及独立代码整理。

后续实验必须保留 **SD2 + DA2 的 12 通道、多步去噪推理**；用户已明确否决单步路线。
不能再次以另一名称恢复 >=80 米强制主监督，也不能把新实验或论文结果写成已验证的天空修复。

---

# 12. 去噪轨迹后训练（2026-09-24：1000 步效果很差 / 停止）

## 方案记录

从原 VGC checkpoint 另开后训练：LR 5e-6，计划 6000 步，500 步内将展开批次比例
增加到 25%。这些批次先以 no_grad 实际执行两步 DDIM，再在下一步反传。
中间状态的 effective noise 根据原始 GT 重新计算；沿用 12 通道、GT mask 和 VGC。
依据 ADDP 对训练—推理分布差异的分析，以及 DepthGen 的展开去噪训练先例。

## 实测与结论

用户在 **1000 个优化步**时反馈“效果很差”，并明确决定不继续后训练方向。
没有提供本轮准确指标、图像或中间轨迹，不能补写具体退化幅度、斑点形态，
也不能说完整 6000 步已验证失败。

**状态：当前配置下 1000 步实测不佳，停止 / 移除；不再建议延长此后训练。**
这不等于证明所有展开训练理论上无效，但已不足以继续推荐当前路线。

## 教训与后续限制

- 数值正确、单元测试和小网络反传通过，不等于真实 SD2 的天空质量有改善。
- 没有轨迹证据，不能把“多步误差累积”当成已确认的根因。
- 短轨迹仍由 GT 加噪初始化，不能代表从纯噪声进行的完整推理分布。
- 后训练可能改变已收敛模型的目标分布；这只是可能原因，未有消融证据。
- 用户要求新方案进入完整主训练，不能换名称继续做 23000 步之后的追加微调。

删除 `train_rollout.yaml`、`denoising_rollout.py`、对应测试、后训练初始化入口及旧说明。
保留通用的中间去噪记录脚本 `trace_depth_denoising.py`，它不是训练方法。
该后续语义关系实验也已失败，见第 13 节；多步 DDIM 和 12 通道约束继续有效。

---

# 13. DA2 空间语义关系主训练（2026-10-01：失败 / 移除）

方案：从 SD2 开始主训练，复用 DA2 最后层 patch tokens，将空间中心化后的余弦关系矩阵
与 U-Net mid_block 的关系矩阵对齐，最大网格 8×12，权重 0.02、1000 步预热，保留 VGC。
用户实测明确判定不可用、效果很差，并指出天空远近甚至反转。未提供本轮确切训练步数或浮点预测指标。

已逐一查看 `output/out` 与 `output/depth_colored` 的 7 对同名图片。按项目固定的 Spectral
色图（归一化值越大越远），观察到：

- `17943623232_5f974cad30_k`、`26439902152_640a2d754e_k`：大片天空为黄/橙色，远处建筑为蓝/绿，天空相对建筑的深度次序错误。
- `25617303635_4e6320d859_b`：石桥场景顶部天空仍出现黄/橙区域，远处树冠反而偏蓝。
- `22390096157_4dceea42ae_k`：雾中街道天空有明显不规则色块和梯度。
- `52845333940_6eb1a7a5ff_k`：城市远景天空从上方绿/青向下方蓝色渐变，未成为统一远端。
- 向日葵 `20304092008_ede0b1304e_k` 和钢桥 `27295431082_5e7d25ba64_k` 的大片天空呈蓝色，
  因此不能说每张图都反转，也不能据此把所有输出全局取反。

**状态：FAILED / REMOVED。** 删除 `semantic_relations.py`、其主训练配置/测试/说明和训练器接入。
保留原 VGC、用户 batch 设置、诊断脚本及独立代码整理。

复盘中可由数学直接确认的不足：`A = normalize(F) normalize(F)^T` 对整体特征符号反转不变，
空间中心化还消除了共同偏移。这一目标没有直接确定深度远近方向，也不保证解码像素的天空平坦性。
这是约束不足的证据，不是“该符号不变性必然导致本次天空反转”的因果证明。
DA2 的语义相似也不等价于同一深度；全图约束可能与原有几何表示冲突，其影响仍无消融验证。

新的后训练不能从这轮失败的 U-Net 接着训。回到原 VGC checkpoint，冻结完整多步采样链，
只让深度解码器学习明确语义天空标签的像素远端约束，并保留非天空输出。
与 §11 的 >=80m latent 主监督、§12 的 GT 加噪短轨迹后训练不同；也不把天空 mask 加进推理输入。
这是一项新的待验证实验，不预先认定有效。若冻结 latent 无法区分天空和物体，解码器校准也可能失败。

---

# 14. `residual_snr` 先验残差补全（2026-10-04：天空更差 / 移除）

## 方案

该方案从 SD2 重新训练主 U-Net，在原 VGC 位置改为噪声感知的先验残差补全：

- 令 `r = pred_x0 - stop_grad(z_DA2)`；
- 在完全无效的 latent cell 内约束 `r` 的区域均值；
- 约束 `r` 的带符号水平/垂直梯度，试图阻止 DA2 远近方向被翻转；
- 用 `min(SNR, 5) / 5` 衰减高噪声时间步的补全项；
- 保留 12 通道、DA2、原主扩散损失、Min-SNR、latent gradient 和 50 步 DDIM。

实现时同时修正了不同 prediction type 的 Min-SNR 换算以及有效边界梯度掩码。
这些通用修正本身没有独立真实训练消融，不能把本次失败归因于其中任意一项。

## 用户反馈与可观察现象

用户明确反馈：`residual_snr` 的天空效果反而更差。提供的城市道路结果中，按项目 Spectral
色图方向观察，顶部大面积天空呈红/洋红近端颜色，远处山体和树林却呈蓝/绿色，形成明显的
天空—地平线远近反向；天空边界还有一圈强烈的黄/绿色过渡。这与“天空保持稳定远端”的目标相反。

当前没有该结果对应的精确 checkpoint 步数、相同 seed 的 VGC 基线、浮点深度或数据集指标，
因此这里只记录定性失败，不虚构退化幅度，也不把单张图推断成所有场景都会反转。

**状态：FAILED / REMOVED。** 删除残差补全 loss、配置开关、专项测试和原实验说明；
主训练恢复原 VGC 的弱区域均值锚点与 invalid-region smoothness。

## 与此前失败方案合并后的判断

已知远端主监督、显式天空 infinity、语义关系和本次 residual 补全虽然目标不同，但共同现象是：
越直接地把额外目标压到天空、far-field 或 invalid region，越容易出现斑点、错误坡度或远近反向。
目前证据更支持停止追加天空监督，而不是继续更换 mask、teacher 或辅助 loss 的形式。

可能原因仍只是待验证解释：invalid 不等于天空；DA2 在天空/低纹理区不是可靠 GT；VAE latent
的局部均值与梯度不直接等于解码像素的绝对远近；辅助目标还会通过共享 U-Net 改变整个场景。

下一轮因此只改扩散训练与多步采样的端点一致性，不新增任何 sky mask、远端伪标签、GT 有效域
或 invalid-region loss。具体候选见 `doc/terminal_snr_alignment.md`，仍属于未验证实验。
