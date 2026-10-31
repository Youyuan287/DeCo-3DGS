# 多视角聚合的正确协议：固定 token 场 + late-interaction 读出

> 更正（2026-10-31）：早先把 `id_top3` 说成“文本进入 2D→3D 聚合、每个 query 一个场”是不准确的。
> 代码里 `id_top3 = mean(top-3 per-view <f_o, t>)`：场始终是**固定的逐视角特征集合**，
> 文本只在查询时用于打分与选 token。这是 **late interaction（ColBERT 式）**，不是重建字段。

## 1. 两种读出，而不是两种场

对每个候选，固定场里存的是逐视角 mask 特征集合 `{f_o}`（query-agnostic，建一次可复用）。
查询时有两种读出：

| 读出 | 公式 | 是否压成一个向量 | 文本用在哪 |
|---|---|---|---|
| 单向量 | `f_c = Σ a_o f_o`，`s = <f_c, t>` | 是 | 只在最后点积 |
| **late interaction** | `s = mean(top-k_o <f_o, t>)` 或 `max_o` | **否（保留 k 个视角 token）** | 查询时逐 token 打分 |

`id_mean` 是单向量；`id_top3` 是 late interaction。**两者的场完全相同**，差别只在读出。

## 2. 这正好是我们实验里最大的差距

B0、同一 released 读权、同一候选池：

| 读出 | macro | Δ vs main | 95% CI |
|---|---|---:|---|
| main（NAG 单向量） | 0.6393 | — | — |
| id_mean（单向量） | 0.6448 | +0.0056 | [+0.0000,+0.0134] |
| id_mean_c（单向量） | 0.6421 | +0.0028 | [+0.0000,+0.0085] |
| **id_top3（late interaction）** | **0.6619** | **+0.0226** | [+0.0138,+0.0319] |
| **gateE（late interaction + gating）** | **0.6760** | **+0.0366** | [+0.0243,+0.0470] |

**结论**：增益不是“文本进聚合”，而是**不把多视角压成一个向量**。
单向量读出（id_mean）即使有同样的逐视角特征，也只有 +0.56；late interaction 保留 top-3 token
后才能到 +2.26。这与“同类多视角余弦高、但平均会稀释小物体/背景”的诊断一致。

## 3. 协议公平性怎么说

- 场是固定的、query-agnostic 的；文本只在查询时使用——**没有违反“建场不用文本”**。
- 但 baseline 的读出是**单向量**；我们的读出是**多 token late interaction**。
  比较时必须声明**字段的 token 预算与读出方式**，或者给 baseline 同样的逐视角 token（让它也用 late interaction）。
- 因此正确表述是：
  > 在相同的 released 3D 场/候选池上，把单向量读出换成多视角 late interaction，macro +2.26；
  > 加上保守 gating 到 +3.66。不是“同读出下超过 baseline”。

## 4. 现成可参考的方法

（网络检索当前受限，以下为已知工作，投稿前需核对引用信息）

1. **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT** (SIGIR 2020)
   —— late interaction / MaxSim 的原始方法：保留多 token，查询时逐 token 打分，不做单向量压缩。
   我们的 `id_top3` 是其在多视角 mask 特征上的直接类比。
2. **OpenMask3D: Open-Vocabulary 3D Instance Segmentation** (NeurIPS 2023)
   —— 每个 3D instance 用多视角 2D mask 特征描述；与我们“候选级逐视角 mask 特征”最接近。
3. **Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance** (CVPR 2024)
   —— 2D mask 引导的 3D instance 生成；可参考其 mask 关联与多视角融合。
4. **MaskClustering: View Consensus based Mask Graph Analysis for 3D Instance Segmentation** (CVPR 2024)
   —— 用视角共识/几何建 mask 图做 class-agnostic 3D instance；对应我们“2D mask 图 → 3D track”的关联路线。
5. **PartSLIP / PartSLIP++** (CVPR 2023/2024)
   —— 多视角 2D 检测 + CLIP 的多视角融合与多原型思路。
6. **Prototypical Networks** (NeurIPS 2017) / **Deep Sets** (NeurIPS 2017) / **Set Transformer** (ICML 2019)
   —— 多原型/集合表示的经典做法：每个实例保留 k 个原型而不是一个均值。
7. **OVIR-3D / SAM3D / SAI3D / Gaussian Grouping / LangSplat / LEGaussians / Dr.Splat**
   —— 3D instance 生成与语言场建场；用于对照“场怎么建、文本何时用”。

## 5. 从“拉开类间距离”出发的可行设计

在固定 token 场 + late interaction 的框架下，有三条不改字段的读出增强：

1. **Token 级 late interaction 变体**：max、top-1/3/5 均值、logsumexp；
   目标是“只让最像物体的视角说话”，避免背景 token 拉低/抬高分数。
2. **几何/可见性门控 token**：只保留候选支撑可见、覆盖率高的视角 token；
   这是 query-agnostic 的建场/筛选，不违反协议。
3. **文本侧 hard-negative margin**（查询时）：
   `s_c = top-k <f_o,t> - λ · top-k max_{t'≠t} <f_o,t'>`
   用场景内其他类别文本作负例，直接拉开类间相似度；场不变，只在读出端。
   这比“白化/中心化”安全，因为它不动 CLIP 几何。
4. **多原型（k-means）**：把候选的视角 token 聚成 k 个原型，查询时 max over prototypes；
   去掉冗余/背景 token，保留多视角模式。参考 Prototypical Networks / PartSLIP++。

## 6. 下一步实验（按优先级）

1. **late-interaction 消融**：max / top-1 / top-3 / top-5 / logsumexp（固定 token 场）。
2. **token 过滤**：coverage/visibility 门控 vs 全视角；看 teatime 是否修复。
3. **文本 hard-negative margin**：用场景类别集合作负例；λ 预注册。
4. **多原型**：候选内 k-means（k=2/3），max over prototypes。
5. **公平基线**：把同一 token 集与 late interaction 读出也给 baseline 候选，做同读出对比。

结论：**“保留 top-k 有力视角、查询时打分”是我们目前最有效且最符合直觉的方向；
下一步应沿 late interaction 做系统消融，而不是继续把特征平均成一个向量。**