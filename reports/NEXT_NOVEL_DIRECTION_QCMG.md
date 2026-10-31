# 下一步创新方向：Query-Conditioned Mask Graph（QCMG）

日期：2026-11-01

## 1. 为什么现在的点不够新

- `top-k late interaction` ≈ ColBERT/MaxSim；
- 多视角 mask 特征聚合 ≈ OpenMask3D/Open3DIS；
- 视角共识 mask 图 ≈ MaskClustering；
- 我们的 gateE/module C 是在这些之上的工程组合。
审稿人会说：**“这是已有组件的组合，没有新问题。”**

要成为 CVPR 级贡献，必须挑战当前 3D-OVS 的三个隐含假设，而不是再调一个融合权重。

## 2. 当前 3D-OVS 的三个错误假设（我们已有反证）

| 假设 | 我们的证据 | 结论 |
|---|---|---|
| 实例独立于 query 存在（固定候选池） | recall 29/208 池内最佳 0.027；greedy20 0.196 | 候选池是文本驱动的，池外实例永远没有机会 |
| 跨视角一致 = 同一物体 | 一致性 found 13.7–15.3% vs 文本 top-3 65.3%/52.4%；xview 选 mask 0.40/0.30 | 一致性是 query-agnostic 的共享语义，不是身份 |
| 单向量语义场足够 | 单向量读出 +0.56 vs 多 token late interaction +2.26；2D lift 只损失 0.049 | 身份在视角证据集合里，不在一个均值向量里 |

## 3. 新问题表述：身份是 query-conditioned 的实例子图

> **开放词汇 3D 分割不应在固定池里选候选；query 应该在一个 class-agnostic 的 2D mask 图上
> “诱导”出一个连通的实例子图。**

形式化：

- 离线建图 `G=(V,E)`（**不用文本**）：
  - 节点 V：所有视角的 SAM mask（或 3D mask track）；
  - 边 E：几何兼容性——把 mask lift 到 Gaussian 支撑，用支撑重叠/可见性/边界连续性；
  - 特征相似度只作弱先验（不是主关联信号）。
- 查询 t：选连通子图 `S ⊆ V`：

```
max_S  F(S) = Σ_{v∈S} <f_v, t>                  # 文本证据
             − λ Σ_{v∈S} max_{b∈B} <f_v, b>      # 场景背景/竞争对比
             − μ · cut(S, V\S)                   # 边界代价（鼓励闭合实例）
             + ν · coherence(S)                  # 子图内部几何一致性
       s.t.  S 连通, |S| ≤ K
```

- 把 S 的 mask 并集 lift 到 3D，渲染。
- 文本只在查询时出现；图是 query-agnostic 的，可复用。

### 为什么这解决真正的问题

1. **Recall**：图里包含所有 2D mask，不依赖文本候选池；池外实例可以被选中。
2. **Deep/rank2**：子图可以跨 root 组合 mask，不再受 NAG 层级/候选池限制。
3. **Found**：如果单个 mask/track 已经强，子图会保持它；`cut` 与背景项防止乱扩。
4. **Query-relative**：同一个区域，query 不同 → 子图不同（table vs plate）。
5. **避免一致性陷阱**：关联用几何（支撑重叠），评分用文本；一致性不进入评分。

## 4. 与已有工作的差异（需要 novelty check）

| 工作 | 做了什么 | 我们不同 |
|---|---|---|
| MaskClustering | view-consensus mask 图，query-agnostic 聚类 | 我们做 **query-conditioned 子图选择**，不是固定聚类 |
| OpenMask3D | 固定 3D instances + 多视角 mask 特征 | 我们不预设 instance；query 诱导子图 |
| Open3DIS | 2D mask 引导 3D instance | 我们的是图上的连通子图优化，带背景/边界项 |
| ColBERT | 文本 token late interaction | 我们做 3D mask 图上的子图选择 |
| ReLaGS/SAI3D | 层级区域合并 + 文本打分 | 我们不依赖 NAG 层级；class-agnostic mask 图 |
| PartSLIP++ | 多视角融合/多原型 | 我们是图优化 + query-relative 背景 |

**核心新意**：不是“更好的聚合”，而是 **把 open-vocabulary 3D 分割重新定义为
“query-conditioned connected subgraph selection on a class-agnostic mask graph”**。

## 5. 理论可写什么

1. **不可能性**：query-agnostic 的变换若保持文本分数（至多常数平移/正缩放），不改变任何选择；
   所以“更好的场”不能靠中心化/一致性/白化获得。
2. **QCMG 的目标**：文本项模块 + 背景项模块 + 负 cut/coherence → 子模或超模结构；
   在连通约束下是 NP-hard，给出贪心/图割的近似保证（需严格推导）。
3. **稳定性**：子图选择的 margin 与扰动半径；query-relative 身份的定义。
4. **与 late interaction 的关系**：单节点子图 + 无背景 = top-1；top-k = 不连通子图；
   QCMG 是其严格推广。

## 6. 分阶段实验（每步都有明确成功标准）

### Stage 1：图与 proposal recall（最便宜，先做）

- 在 LERF 四场景建 mask 图（节点=逐视角 SAM mask；边=lift 支撑 IoU）。
- 指标：**GT 物体是否在图中**（proposal recall），以及图规模/连通分量数。
- 成功标准：recall 族 29 条中，GT 物体在图中的比例 ≥ 0.6（当前候选池 0）。
- 若图里没有，先解决 mask/lift，而不是子图优化。

### Stage 2：query-conditioned 子图选择

- 实现目标函数（文本 − 背景 − cut + coherence），贪心/图割。
- 与 gateE 比较：recall、found、deep、macro。
- 成功标准：recall 0.02→0.10+，found 不降 >0.005，macro >0.68。

### Stage 3：跨数据集/novel-view

- ScanNet instance-level、3D-OVS；novel-view 渲染评测，证明不是查询视角 2D 捷径。
- 与 MaskClustering/OpenMask3D 类方法做同协议对比。

## 7. 风险

1. 图规模：teatime 26k masks/1M gaussians；需要稀疏化/分块。
2. mask 图噪声：2D mask 不完美；几何边可能连错。
3. 连通约束的优化：NP-hard；贪心/图割的近似与实现。
4. novelty：MaskClustering/Open3DIS 很近；必须把“query-conditioned subgraph”写成新问题，
   并做同协议对比；投稿前做一次系统 novelty check。
5. 如果 Stage 1 proposal recall 上不去，说明 2D mask/lift 不是可靠 proposal，需换 proposal 源。

## 8. 一句话

**不要再做“在固定候选池里更好地选”；要做“query 在 class-agnostic 2D mask 图上诱导实例子图”。**
这是把 3D-OVS 从“文本重排”推进到“文本条件实例发现”的唯一一条能同时解决 recall 和 deep 的路。