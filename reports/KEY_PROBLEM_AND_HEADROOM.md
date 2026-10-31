# 关键问题、GT 头寸与可借鉴的理论/方法

日期：2026-11-01
原则：不按 ReLaGS 的实现细节打补丁；先定位方法的关键假设；用 GT 量化头寸；
只提出有实验证据、有现成方法/数学可借的方向。

---

## 1. 用 GT 把损失拆开：头寸在哪

当前最好：gateE 0.6760（B0）。用 GT 做不同层级的 oracle：

| 层级 | 做法 | macro | 相对当前 |
|---|---|---:|---:|
| 当前 gateE | 固定池 + late interaction + gating | **0.6760** | — |
| 候选池 oracle | 每查询在 released pool 里选最佳候选（GT） | **0.7094** | +3.3 |
| 2D/3D 族级 oracle | 按族在 2D GT-best lift 与 3D 选择间选优（GT） | **0.7145** | +3.9 |
| greedy5 oracle | 池内贪心加候选（GT） | **0.7500** | +7.4 |
| greedy20 oracle | 更大池 + 贪心（GT） | **0.7818** | +10.6 |

分族：

| 族 | n | gateE | 2D GT-best lift | greedy20 |
|---|---:|---:|---:|---:|
| found | 124 | **0.8312** | 0.6654 | 0.8737 |
| rank2 | 36 | 0.6402 | 0.4726 | 0.8218 |
| deep | 19 | 0.6607 | **0.8247** | 0.9108 |
| recall | 29 | **0.0198** | **0.2352** | 0.1959 |

**结论**：
1. 当前已接近“池内 oracle”（0.676 vs 0.709），**选择/读出不是主要头寸**。
2. 真正的大头寸在 **候选/实例生成**：recall 0.02→0.20，deep 0.66→0.82–0.91。
3. found 已经 0.831，继续优化 found 的边界收益很小；必须解决 deep/recall。

---

## 2. 方法的关键问题：feature-similarity hierarchy ≠ objectness

ReLaGS/SAI3D 类方法的核心假设是：

> 物体实例 = 由特征相似度层级合并得到的区域。

我们的实验直接反证：

1. **2D mask 是 object-level，lift 损失很小**：全部 208 查询平均只损失 0.049 IoU；
   deep 的 2D GT-best lift 达 0.825。
2. **3D 区域是混合特征**：逐视角观测是区域 mask 的 beta 混合；区域特征里物体+背景共存。
3. **跨视角一致性不是身份**：一致性 found 13.7–15.3%，文本 top-3 65.3%/52.4%；
   共享/背景语义反而更一致。
4. **层级本身不稳定**：官方与重建的 4% 特征差，导致 ramen/teatime/waldo 只有 61–68% 簇可配对；
   官方 level-0 7852 vs 发布 nag-l1 7936。
5. **候选池是文本驱动的**：recall 29 条池内最佳 0.027；池外实例永远没机会。

所以关键问题不是“怎么更好地聚合/选择”，而是：

> **3D-OVS 缺少 class-agnostic 的 objectness；当前方法用语义聚类的结果当作实例候选。**

这是一个方法层面的假设问题，不是 ReLaGS 某个模块的实现问题。

---

## 3. 不是补丁：Objectness-first 3D-OVS

新的方法主张（不依赖 ReLaGS 的具体分区/合并实现）：

1. **Class-agnostic 3D instance proposal**：
   - 节点：逐视角 SAM mask（object-level，已有且强）；
   - 边：几何兼容性（lift 后 Gaussian 支撑重叠/可见性/边界连续），
     特征相似度只作弱先验（一致性已被证伪）；
   - 结构：mask graph 上的 track / 连通分量 / 谱聚类 → 3D instances；
   - **不用文本**，可离线建一次。
2. **Open-vocabulary recognition**：
   - 每个 instance track 的逐视角 mask token 用 late interaction 打分；
   - 查询时选 top track；可选 query-conditioned 子图精修。
3. **Query-relative identity**：
   - 同一个区域，query 不同 → 选中的 track/子图不同；
   - 实例不是固定池，而是“查询在图上诱导的对象”。

这与“在 ReLaGS 上加一个 gating”有本质区别：**改变的是实例从哪里来**，不是读出。

---

## 4. 可借的现成方法与数学

### 4.1 3D instance proposal / mask graph

- **MaskClustering**（CVPR 2024）：view-consensus mask graph → class-agnostic 3D instances；
  我们改用**几何支撑**而不是特征共识作为主边。
- **Open3DIS**（CVPR 2024）：2D mask guidance for 3D instance proposal。
- **SAM3D / SAI3D**：class-agnostic 3D segment proposal。
- **Gaussian Grouping**：3D instance masks。

### 4.2 图划分/聚类数学

- **Normalized cuts / spectral clustering**（Shi & Malik）：mask graph 的谱划分；
  有明确的 cut 目标和松弛解。
- **Graph cuts / multiway partitioning**：连通实例。
- **Optimal transport / Sinkhorn**：跨视角 mask 匹配，替代启发式关联。
- **Community detection**（Louvain/Leiden）：mask track 的连通社区。

### 4.3 查询条件化的选择

- **Prize-Collecting Steiner Tree / submodular maximization**：
  在 mask graph 上选连通子图，文本作 prize，几何作边代价；有近似保证。
- **Late interaction / ColBERT**：多 token 打分，避免均值稀释。
- **Facility location / set cover**：选择互补的视角/mask。

### 4.4 理论可写

1. **不可能性**：query-agnostic 变换若保持文本分数（常数平移/正缩放），不改变选择；
   所以一致性/中心化/白化不能靠“更好的场”涨点。
2. **Objectness 假设**：feature-similarity hierarchy 不能保证实例边界；给出反例/统计证据。
3. **子图/谱划分**：normalized cut 或 PCST 的目标、松弛、近似保证。
4. **稳定性**：track/子图选择的 margin 与扰动半径。

---

## 5. 第一个必须做的实验：GT proposal recall

在投入方法前，先用 GT 验证 objectness proposal 是否可行：

- 在 LERF 四场景建 mask graph / track（不用文本）；
- 指标：**GT 物体是否在 track 集合中**（proposal recall），以及 track 的 IoU 上界；
- 目标：recall 族 29 条中 ≥60% 的 GT 物体在 track 中（当前文本池 0）；
  deep 族 ≥80%。
- 如果达不到，说明 2D mask/lift 不是可靠 proposal，需要换 proposal 源（例如更强的 2D 分割模型），
  而不是继续做子图优化。

这一步完全用 GT 衡量头寸，符合“先实验、再方法”的原则。

---

## 6. 成功标准与风险

**成功标准（分阶段）**：
1. Proposal：recall 族 GT 在 tracks 中 ≥60%，deep ≥80%。
2. Recognition：在 tracks 上 late interaction，recall 0.02→0.10+，found 不降 >0.005。
3. 端到端：macro >0.70（先接近 2D/3D 族级 oracle 0.7145），再冲 greedy5 0.75。
4. 跨数据集/novel-view：ScanNet instance-level、3D-OVS；novel-view 评测。

**风险**：
- mask graph 规模/噪声；track 关联错误。
- 2D mask 在 recall 族的上界只有 0.235，proposal recall 可能上不去。
- 与 MaskClustering/Open3DIS 的新颖性边界；必须做同协议对比和清晰的问题陈述。
- 子图/谱划分的工程复杂度。

---

## 7. 不建议再做的

- 在固定候选池上调 gating/λ/融合权重；
- 文本侧模板/lexicon/场景词表（已证伪或违规）；
- 一致性/中心化/白化进入评分；
- 把 2D proposal 硬 union 进 3D。

一句话：**关键问题是 objectness，不是读出；用 GT 先验证 proposal recall，再谈图划分和查询条件化选择。**