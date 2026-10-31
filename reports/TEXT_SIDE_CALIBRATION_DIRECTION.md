# 从 C2/ScanNet 出发：文本侧校准能否成为有力创新？

日期：2026-11-01

> **撤回（2026-11-01）**：下面第 2–6 节的“vocabulary-aware / 场景类别 bank”方案**不成立**。
> LERF-OVS 逐 query 处理正是开放词汇的定义；使用场景内其他类别（14–21 个）做文本中心化、
> hard-negative 或 bank softmax，会把任务变成闭集/直推，违背任务设定。
> C2 在 ScanNet 上的收益不能通过“把 class bank 搬进 LERF”来移植。
>
> 仍可合法使用的文本侧信号必须**只依赖当前 query**：
> 1. 同一 query 的模板/复述一致性（paraphrase consistency），用作可靠性/不确定性；
> 2. 当前 query 的组合分解（属性/部件，如 "green toy chair"），不引入外部类别；
> 3. 固定的通用背景提示（"background", "wall", "floor" 等，且不得包含测试类别），需谨慎；
> 4. 候选间的 per-query 分数归一化（本来就在用）。
>
> 这些是**小模块**，很难单独成为 CVPR 主创新；文本侧不能作为解决 3D-OVS 的核心路线。
> 若要主创新，仍应回到 class-agnostic 实例 proposal / query-conditioned 子图（QCMG）。

---

## 1. 先把 C2 的 ScanNet 结论读准

DeCo 的 ScanNet 数字（pooled，18 类）：

| 配置 | mIoU |
|---|---:|
| baseline（4 templates 平均，无算子） | 40.79 |
| text-only（C2） | **38.19**（比 baseline 低） |
| image raw + text bare | 39.02 |
| image raw + lexicon/templates | 46.82 |
| image centred + text bare | 53.67 |
| image centred + lexicon/templates | **57.85** |
| C1+C2 unified | **58.55** |

2×2 交互 **−3.62**（次可加）；lexicon ladder（image centred）：bare 42.53 → 4 templates 48.55 →
LSE 50.76 → hand lexicon 53.17 → templates+lexicon 53.62。

**结论**：
1. **C2 单独不涨点**；文本侧只有在与 C1（共享模态去除）和**闭集 class bank** 结合时才有效。
2. 它的机制不是“更多文本”，而是**在 class bank 上去共享模态、锐化类间边界**。
3. 这也解释了 LERF 上 C2 失败：LERF 是 per-query 二分类检索，没有 class bank；
   lexicon 扩展只会引入 false positives（C2 4 templates 0.6003、lexicon 0.4929，均低于 0.6401）。

## 2. 创新机会：Vocabulary-Aware Text Calibration（VATC）——已撤回，违反开放词汇设定

LERF 每场景的类别数只有 **14–21**（figurines 21、ramen 14、teatime 14、waldo 18）。
这是一个天然的小 class bank。当前 3D-OVS 方法**把每个 query 独立处理，完全忽略场景词表**。
我们可以把 C2 的机制以 query-conditioned 的方式引入：

1. **场景词表文本中心化**（C2a，但用场景类别 bank，而不是 synonyms）：
   `t_c' = l2n(t_c − mean_{c' in scene} t_c')`；图像特征不动。
2. **类间硬负 margin**（核心新点）：
   `s_c = z_c − λ · max_{c'≠c} z_c'` 或 `− λ · logsumexp_{c'≠c} z_c'`，
   其中 `z_c = <f, t_c>`；直接压掉“不同类也高”的共享语义。
3. **Uniqueness / margin 作为可靠性**：
   `u = z_c − max_{c'≠c} z_c'`；用 u 决定是否信任 mask 证据/gating，
   或在候选间做 margin-aware rerank。
4. **词表感知的 readout**：对每个候选，在场景 bank 上做 softmax/argmax（类似 ScanNet），
   但**只用于校准当前 query 的分数**，不把其他类别的文本当作查询扩展。

## 3. 为什么这可能是一个有力创新（而不是 C2 重做）

| 维度 | C2（DeCo） | VATC（提议） |
|---|---|---|
| 场景 | 闭集 ScanNet 19 类 | 开放词汇 LERF，但利用场景词表 |
| 文本用法 | class bank + lexicon/templates | **只用场景类别 bank 做对比/校准，不扩同义词** |
| 作用 | 去共享模态、锐化类间 | **query-conditioned hard negative + uniqueness** |
| 图像侧 | C1 中心化 | 不动图像特征（正交、可叠加） |
| 失败模式 | LERF 上 lexicon 假阳性 | 词表未知时退回 C2/无校准 |

**潜在贡献**：
1. 新设定：**vocabulary-aware open-vocabulary 3D segmentation**（场景词表已知；
   当前方法忽略这个免费信号）。
2. 机制：文本侧共享模态 + 类间竞争，解释 C2 的 ScanNet/LERF 不对称。
3. 方法：hard-negative margin + uniqueness gating；训练-free、图像侧不动。
4. 协议：给出 vocabulary-aware 与 vocabulary-free 两种评测，明确不偷看。

## 4. 最小实验（先做 E1，便宜）

### E1：文本侧校准（只改读出，不改图像/候选）

- 对 LERF 每个 query，bank = 场景内所有类别（14–21 个）。
- 变体：
  - `c2a_bank`：场景 bank 文本中心化；
  - `hn_max`：`z_c − λ max_{c'≠c} z_c'`；
  - `hn_lse`：`z_c − λ logsumexp_{c'≠c} z_c'`；
  - `bank_softmax`：softmax over bank 的概率/熵。
- 作用位置：candidate score（可先用现有 gateE 的候选池），λ 预注册。
- 指标：macro、found/deep/rank2/recall。
- 成功标准：found 不降 >0.005，macro +≥0.01，或 deep/recall 明显提升。

### E2：Uniqueness gating

- 用 `u = z_c − max_{c'≠c} z_c'` 作为置信度：
  - 只在 u 高时用 mask rerank/gateE；u 低时保守回退。
- 目标：减少 found 翻转，提升 deep。
- 成功标准：gateE 不降，deep 提升。

### E3：ScanNet sanity

- 在 ScanNet 19 类上比较：C1+C2（58.55）vs hard-negative margin；
  预期 margin 与 C2 同向或更好，证明机制一致。
- 若 ScanNet 上 margin 反而更差，说明“类间竞争”不是 C2 的主因，VATC 需要重新表述。

## 5. 风险与 novelty check

1. **使用场景词表是否允许**：LERF 的 GT 提供所有类别；但“开放词汇”通常指任意文本。
   必须定义 **vocabulary-aware setting**，并同时报告 vocabulary-free；不能偷偷用。
2. **novelty**：C2 已做 class-set 文本中心化；negative prompts 在 OV detection 中也有。
   新意必须落在 **query-conditioned hard-negative + uniqueness gating + 协议**，
   而不是“centering 文本”。投稿前做系统 novelty check。
3. **bank 很小（14–21）**：文本中心化可能不稳；需要 bootstrap/逐场景。
4. **LERF 是 per-query 二分类**：bank softmax/argmax 可能引入 false positives；
   hard-negative margin 比 argmax 更安全。
5. **与 gateE 的交互**：必须做同候选池、同读出的消融，避免 A+B 叠加。

## 6. 推荐

先做 **E1 的 `hn_max` / `hn_lse` 两个变体**（只改候选分数，最便宜），
再决定是否做 E2/E3。若 E1 在 found 不降的前提下 macro +≥1，这条线成立；
若 E1 无效，说明文本侧的场景词表在 LERF 上也不提供可用的判别信息，
应回到 QCMG/instance proposal，而不是继续调文本权重。