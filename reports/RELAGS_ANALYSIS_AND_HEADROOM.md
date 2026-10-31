# ReLaGS 分析、不足、改法与指标裕量

日期：2026-11-01
数据：LERF-OVS 四场景 208 查询；官方发布读权；ReLaGS 口径 64.01，DeCo B0 63.9263；
当前最好 gateE 0.6760。

---

## 1. ReLaGS 怎么做

官方 `run_lerf.sh` 的顺序：

1. **MWP 剪枝**：`max_weight_pruning --contribution_threshold 0.0005`。
   对每个 Gaussian 取所有训练视角的最大贡献（α·transmittance 类），低于阈值删除。
2. **超点分区**：`sp_partition`（FRNN kNN）+ `graph_weight`（SAM mask 对比度重加权）
   + 第二次 `sp_partition`（cut-pursuit）→ `nag-l1.pt` 超点标签。
3. **语义投影 merge_proj**：
   - 逐视角渲染 `weight`，取 `seg_map[level]` 与 mask 特征；
   - 计算 superpoint→mask 的 assignment ratio，`RATIO_THRESHOLD=0.3`，加权得到每个 superpoint 的逐视角特征；
   - 累积每个 superpoint 的多视角特征，**ROFA** 过滤：
     `q_i = mean_{j≠i} cos(f_i,f_j)`，保留 `q_i > μ − τσ`（τ=3），再平均；
   - 层级区域生长（`thres_connect 0.9,0.7,0.7`、`thres_merge 20`）得到 4 级 `nag` 与 `nag_feat`。
4. **查询**：文本编码 → 每级余弦 → `search_matched_superpoint_in_mhtree`
   （topk=5 roots、level_until=3、filter_small=50、gap 选前缀）→ 渲染 semantics>0.5。

关键假设：
- **物体实例 = 特征相似度层级合并出来的区域**；
- 多视角特征可用 **一致性（ROFA）** 去噪后平均；
- 候选由 **文本相似度 top-5 roots** 产生。

---

## 2. 不足（都有实验证据）

### 2.1 区域是“混合特征”，不是物体

- 逐视角观测是区域 mask 的 beta 混合，物体+背景共存。
- 2D mask 是 object-level：全部 208 查询 lift 到 3D 平均只损失 **0.049 IoU**；
  deep 的 2D GT-best lift 达 **0.8247**。
- 但 3D 区域特征的一致性聚合把 found 命中打到 **13.7–15.3%**；
  文本 top-3 是 **65.3%/52.4%**。
- **结论**：问题不在“视角不够多”，在区域特征把别的语义混进来了。

### 2.2 层级合并由特征相似度驱动，不是 objectness

- 官方 level-0 **7852** 超点 vs 发布 `nag-l1` **7936**；官方读权不是用发布分区建的。
- 匹配簇特征余弦 0.94–0.97，但 ramen/teatime/waldo 只有 **61–68%** 的簇能在 IoU>0.5 下配对；
  4% 的特征差被区域生长放大成层级差异。
- 层级不同 → 候选池不同 → 少数查询翻转，造成 5–12 分绝对差。
- **结论**：feature-similarity hierarchy ≠ objectness；层级不稳定。

### 2.3 ROFA/一致性是 query-agnostic 的

- 一致性优化“视角之间像不像”，在混合特征上更容易选到共享/背景语义；
- found 命中 13.7–15.3%（文本 top-3 65.3%/52.4%）；
- 在我们的端到端 A/B 中 ROFA 中性/空操作；重建上场景波动 ±14。
- **结论**：一致性只能用于关联/分组，不能用于评分。

### 2.4 候选由文本生成，recall 有结构性上限

- top-5 roots + leaf 提升；池内 oracle **0.7094**；
- recall 族 29/208：池内最佳 **0.027**；greedy20 **0.196**；
- 2D GT-best lift 在 recall 上界 **0.2352**。
- **结论**：池外实例永远没有机会；这是最大的结构性不足。

### 2.5 单向量平均会稀释

- 单向量读出 +0.56 vs 多 token late interaction +2.26（B0）；
- teatime 上 id_top3 命中只有 0.204（主路径 NAG 0.858），说明平均/选择信号在难场景会崩。
- **结论**：不应把多视角压成一个均值向量。

---

## 3. 我会怎么改（方法层面，不是打补丁）

### 3.1 核心：Objectness-first

1. **Class-agnostic 3D instance proposal**（不用文本）：
   - 节点：逐视角 SAM mask；
   - 边：几何兼容性（lift 后 Gaussian 支撑重叠/可见性/边界连续）；
   - 结构：mask graph 上 track / 连通分量 / 谱聚类（normalized cut）→ 3D instances。
2. **Open-vocabulary recognition**：
   - 每个 track 的逐视角 mask token 用 late interaction 打分（top-k，不平均）；
   - 查询时选 track；可选 query-conditioned 子图精修。
3. **Query-relative identity**：同一区域，query 不同 → 选中的 track/子图不同。

这改变的是“实例从哪里来”，不是 ReLaGS 的某个实现细节。

### 3.2 可借的现成方法/数学

- **MaskClustering**（view-consensus mask graph）、**Open3DIS**（2D mask guidance）、
  **SAM3D/SAI3D**、**Gaussian Grouping**：class-agnostic 3D proposal。
- **Normalized cuts / spectral clustering**（Shi & Malik）：mask graph 划分。
- **Optimal transport / Sinkhorn**：跨视角 mask 匹配。
- **Prize-Collecting Steiner Tree / submodular maximization**：查询条件化连通子图；
  文本作 prize，几何作边代价，有近似保证。
- **Late interaction / ColBERT**：多 token 打分。

### 3.3 不做什么

- 不在固定池上调 gating/λ/融合权重；
- 不用一致性/中心化/白化进评分；
- 不用场景词表（违反开放词汇）；
- 不把 2D proposal 硬 union 进 3D。

---

## 4. 指标裕量（GT oracle）

| 层级 | macro | 相对 0.676 |
|---|---:|---:|
| 当前 gateE | **0.6760** | — |
| 池内 oracle（每查询选最佳候选） | **0.7094** | +3.3 |
| 2D/3D 族级 oracle | **0.7145** | +3.9 |
| greedy5 oracle | **0.7500** | +7.4 |
| greedy20 oracle | **0.7818** | +10.6 |

分族与“换 proposal”后的上限：

| 族 | n | 当前 gateE | 2D GT-best lift | greedy20 |
|---|---:|---:|---:|---:|
| found | 124 | 0.8312 | 0.6654 | 0.8737 |
| rank2 | 36 | 0.6402 | 0.4726 | 0.8218 |
| deep | 19 | 0.6607 | **0.8247** | 0.9108 |
| recall | 29 | **0.0198** | **0.2352** | 0.1959 |

按宏平均换算（族内全部达到 2D lift）：
- recall：`(0.2352−0.0198)×29/208 = +0.0300`
- deep：`(0.8247−0.6607)×19/208 = +0.0150`
- 合计 **+0.045 → ≈0.721**（仅靠 proposal 把 recall/deep 拉到 2D 上界）

若再用 GT 在池内贪心加候选：
- greedy20 = **0.7818**（+10.6），但这是 GT 选择，不是可部署方法。

**现实目标**：
- 第一步：objectness proposal 让 recall 从 0.02→0.10+、deep 0.66→0.75+，macro **0.70–0.72**；
- 第二步：query-conditioned 子图/候选联合，冲 **0.73–0.75**；
- 0.78 是 oracle，不作为方法目标。

---

## 5. 最小验证（先做，不投入大改）

**GT proposal recall 审计**：
- 在 LERF 四场景建 mask graph/track（不用文本）；
- 指标：GT 物体是否在 track 集合中；track 的 IoU 上界；
- 目标：recall 族 ≥60% 的 GT 物体在 track 中，deep ≥80%；
- 如果达不到，说明 2D mask/lift 不是可靠 proposal，需要换 proposal 源。

只有这个门控通过，才值得投入 mask graph/谱划分/PCST。

---

## 6. 一句话

**ReLaGS 的关键不足不是聚合或 gating，而是“用特征相似度层级当物体、用文本 top-5 当候选”；
GT 裕量约 +3.3（池内）/ +4.5（proposal 修复 deep+recall）/ +7–10（候选联合 oracle）。
下一步应先做 GT proposal recall，再决定是否投入 objectness-first。**