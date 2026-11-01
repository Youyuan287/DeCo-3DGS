# 超点选择、边界高斯与 DIGS 之后的真实空位

日期：2026-11-01
数据：LERF-OVS 四场景 208 查询；ReLaGS/DeCo 读端（B0，`--center image`）。
GT 为查询视角**可见高斯**的 query-view 代理，**不是官方渲染 mIoU**，只用于同口径头寸比较。
统计已限制在查询视角可见集合（`render_point weight>1e-4`），避免把视外高斯算进分母。

---

## 0. 结论

1. **不能只定义像素边界。** 像素 BIoU 是评测探针；方法空间应是高斯/区域。
   但“边界高斯 + 多峰分布 + late-argmax”已经是 **DIGS 的核心贡献**（LERF-OVS 71.95），
   不能再作为我们的创新点。
2. **ReLaGS 确实只选很少的超点**：中位数 2 个，均值 2.11，87.5% 的查询 ≤3 个；
   可见选中高斯均值 5,389、GT 4,785，precision 0.707、recall 0.674、3D IoU 0.553。
3. **“多选几个超点”不是解法**：扩到整个候选池，3D IoU 0.553→0.432（−0.121），
   精度 0.707→0.493；GT-oracle 最好前缀/单候选也只 +0.079/+0.081。
4. **边界是 found/rank2 的主要误差形态，但不是全局杠杆**：可见假阳性中 64% 在边界带内
   （median 87%）；把边界假阳性全删，3D IoU 只 +0.040。分族：found +0.048、rank2 +0.063、
   deep +0.003、recall +0.004。**deep/recall 的错误是整块/root 选错，边界精修救不了。**
5. **真正的头寸在“单位/本体”**：已选 root 内完美挑高斯，3D IoU 0.553→0.714（+0.161）；
   再选对 root 还有 +0.286。DIGS 已经用 class-agnostic instance ontology 吃掉大部分；
   它自己承认的下一个空位是 **SAM 粒度导致的 part-level 查询**。
6. **建议方向**：多粒度 part-whole 本体 + 查询条件化的粒度/集合选择，
   而不是边界精修、不是扩 top-k、不是重做 class-agnostic 实例发现。

---

## 1. ReLaGS 实际选了什么

代码路径（`nag_data.py`）：

- `search_matched_superpoint_in_mhtree(topk=5, level_until=3, filter_small=50)`：
  先取 root top-5，再在有叶节点相似度更高时把 root 提升为 leaf candidates；
- `_filter_by_similarity_gap`：按相似度最大间隔截断，得到前缀 `g`；
- `_get_related_gaussians(candidates)`：对每个候选，把 `labels[level]==index` 的
  **整个超点**全部置 1。**没有高斯级选择，也没有部分级选择。**

208 条查询（可见集合口径）：

| 指标 | 值 |
|---|---:|
| 选中候选数（超点/叶） mean / median | 2.11 / **2** |
| 选中候选数 ≤3 的比例 | **87.5%** |
| 分布 | 1:90, 2:60, 3:32, 4:20, ≥6:6 |
| 可见选中高斯 mean / median | 5,389 / 2,824 |
| GT 可见高斯 mean / median | 4,785 / 2,514 |
| 3D precision / recall / IoU | **0.707 / 0.674 / 0.553** |
| GT 高斯落在已选 root 内的比例 | 0.714（median 0.872） |
| 边界假阳性 / 全部可见假阳性 | 0.642（median 0.872） |

分场景（3D 代理）：

| 场景 | n | 超点均值 | 可见选中高斯 | GT 高斯 | precision | recall | IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| figurines | 56 | 2.16 | 1,555 | 1,251 | 0.693 | 0.658 | 0.503 |
| ramen | 71 | 2.11 | 3,941 | 4,399 | 0.594 | 0.564 | 0.456 |
| teatime | 59 | 1.92 | 10,887 | 8,326 | 0.803 | 0.802 | 0.686 |
| waldo_kitchen | 22 | 2.50 | 5,082 | 5,529 | 0.855 | 0.721 | 0.633 |

**读法**：可见选中集平均只比 GT 大 13%，不是“一半高斯都不是目标”；
但 precision 在 ramen 只有 0.594，deep/recall 族更低（见下）。
真正的结构问题是 **root 覆盖**：GT 高斯只有 71.4% 落在已选 root 内。

---

## 2. 加超点 / 边界精修 / 叶拆分，各有多少裕量

| 改动 | 3D IoU 均值 | 相对当前 | 说明 |
|---|---:|---:|---|
| 当前 gap 前缀 | **0.553** | — | 实际方法 |
| 扩到整个候选池（所有 top-5/leaf） | 0.432 | **−0.121** | 精度 0.707→0.493，越加越差 |
| GT-oracle 最好前缀 | 0.632 | +0.079 | 前缀长度 oracle |
| GT-oracle 最好单候选 | 0.633 | +0.081 | 单候选 oracle |
| 边界假阳性全删（oracle） | 0.593 | **+0.040** | 只删边界 FP，不能加回漏检 |
| 已选 root 拆叶 + GT 纯度≥0.5（oracle） | 0.585（n=143） | +0.035（143 条配对） | deep +0.096；found/rank2 为负 |
| 已选 root 内完美挑高斯（oracle 上界） | **0.714** | **+0.161** | 精度=1、recall=root recall |
| 全场景完美选高斯（oracle） | 1.000 | +0.286（相对 root 上界） | 需要新的 root 选择 |

**关键结论**：

- **扩池无效**：全池精度 0.493；当前 gap 已接近前缀最优（58.2% 恰好，
  30.8% 选多了，11.1% 选少了）。
- **边界只是 found/rank2 的局部杠杆**：found +0.048、rank2 +0.063；
  deep +0.003、recall +0.004。边界 FP 在 found/rank2 中占 FP 的 82%/59%，
  但这些查询本身 precision 已 0.84/0.71，FP 少，所以绝对增益只有 5–6 分。
- **deep/recall 是 root/单位错误**：deep precision 0.337、recall 0.324；
  recall precision 0.198、recall 0.066；边界精修几乎为 0。
- **叶拆分只对 deep 有 +0.096**，found/rank2 为负，recall 为 0；不能作为通用模块。

分族（可见口径，均值）：

| 族 | n | 超点 | precision | recall | IoU3D | root recall | 全池 | 最好前缀 | 边界增益 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| found | 124 | 1.92 | 0.841 | 0.824 | 0.708 | 0.858 | 0.500 | 0.770 | +0.048 |
| rank2 | 36 | 2.22 | 0.713 | 0.828 | 0.607 | 0.862 | 0.495 | 0.665 | +0.063 |
| deep | 19 | 2.95 | 0.337 | 0.324 | 0.197 | 0.324 | 0.459 | 0.513 | +0.003 |
| recall | 29 | 2.24 | 0.198 | 0.066 | 0.056 | 0.170 | 0.040 | 0.076 | +0.004 |

---

## 3. DIGS 已经占掉了什么

DIGS（arXiv 2610.16177，2026-10-12）在 LERF-OVS 报 **71.95 mean mIoU**，核心是：

- 每个高斯维护 **top-M label distribution**，用 tempered Dirichlet 更新；
- 边界高斯保留多峰分布，**late-argmax renderer** 在像素级融合，得到干净边界；
- 双向状态机维护 **class-agnostic instance ontology**；
- 实例级用 CLIP 视觉 embedding（隔离 3D footprint）+ LLM 属性描述做开放词汇检索。

它自己列出的残余失败：

1. **F1 透明/半透明**：alpha compositing 让前景投票泄漏到背景高斯，透明盒/杯子会把
   footprint 拉大；
2. **F2 SAM 粒度封顶 part-level 查询**：当 query 指代大 SAM mask 的子区域时，
   本体里没有这个 part 的槽位，检索只能返回整个 parent。LERF 上的例子：
   waldo 印刷字符、pirate hat、hooves + dall-e brand。

它的 future work 明确写：**把多粒度 SAM mask 融进同一套 voting/ontology**；
以及联合几何精修（透明/薄结构）。

**因此**：

- 边界高斯：DIGS 核心，不能做；而且我们的 oracle 只有 +0.040，且只对 found/rank2；
- class-agnostic 实例发现：DIGS 核心，不能重做；
- “把 2 个超点扩成更多”：我们的数据是负的，不能做；
- **part-level / 多粒度 / 查询条件化**：DIGS 自己承认的空位，且 LERF 查询里大量存在。

---

## 4. LERF 的查询分布支持“part/属性/集合”方向

四场景 208 条查询里，很多不是“整个物体”：

- part：`porcelain hand`、`tesla door handle`、`pirate hat`、`bear nose`、`hooves`、
  `onion segments`、`waldo`（印刷字符）；
- 属性/组合：`green toy chair`、`red toy chair`、`rubber duck with hat/buoy`、
  `pink ice cream`、`dark cup`、`red cup`、`frog cup`、`yellow desk`、`stainless steel pots`；
- 集合/计数：`three cookies`、`chopsticks`、`pots`、`onion segments`、`hooves`。

DIGS 的检索是 **单个实例 argmax**（`argmax_j cos(e_j, CLIP(q))`），没有 query 解析、
没有 part-whole 层级、没有多实例集合选择。它的 F2 正是 part 查询的失败；
plural/集合查询则是另一个未覆盖的 argmax 限制。

---

## 5. 建议：多粒度 part-whole 本体 + 查询条件化选择

不是“在 ReLaGS 上加边界项”，而是换掉“实例单位”和“查询接口”。

### 5.1 表示

1. **多粒度 SAM 节点**：同一场景用多组 points-per-side / crop layer / 自动 mask 生成，
   得到不同粒度的 2D mask；用 IoU/containment 建 **part-whole DAG**（子节点⊂父节点）。
2. **3D 节点**：每个节点用现有 voting/分布机制落到高斯，得到节点 footprint；
   子节点与父节点共享高斯时，保留 **part-whole 一致性**（子分布是父分布的细化，
   父节点可由子节点组成；冲突时用父节点先验约束子节点）。
3. **粒度标签**：节点带 `level`（whole/part/subpart）和 `containment`，不是扁平实例集。

### 5.2 查询

1. **查询解析**：把 query 分成 `{主体, 属性, 部分, 数量, 关系}`；
   例如 `bear nose` → parent=bear, part=nose；`three cookies` → count=3, sibling set；
   `rubber duck with hat` → 组合/关系。
2. **粒度选择**：同一区域，`bear` 选父节点，`bear nose` 选子节点；
   不是所有 query 都取同一个 SAM 粒度。
3. **集合选择**：plural/count 查询不是 argmax 一个实例，而是在 sibling 集合上做
   **submodular / coverage / PCST** 选择，满足数量/覆盖约束。
4. **part-whole 一致性**：子节点预测不能越出父节点；父节点查询可由子节点投票聚合。

### 5.3 为什么这不是 DIGS 的增量

- DIGS：**扁平、单粒度、单实例 argmax**；F2 明确说 part 查询没有槽位。
- 我们：**层级 part-whole 本体 + 查询条件化粒度/集合选择**；
  贡献在“同一场景按 query 选不同粒度/多个实例”，不是“再做一个更好的边界渲染器”。
- 需要在 related work 里正面对比 DIGS、SAM3D/SAI3D、Open3DIS、MaskClustering、
  Gaussian Grouping，并做同协议实验；novelty check 必须先做。

### 5.4 最小验证（先做，不投入大改）

1. **Part proposal recall**：在四场景建多粒度 mask DAG；对 part 类查询
   （pirate hat、bear nose、hooves、door handle、waldo 字符等），检查 GT 子区域是否
   在节点集合中；对比单粒度 SAM 的 recall。
2. **Granularity oracle**：如果允许每查询在 DAG 中选一个节点（GT 选），
   2D/3D IoU 上界是多少；对比当前单粒度上界。
3. **Set oracle**：对 plural/count 查询，允许选 sibling 集合（GT 选），看 mIoU 上界。
4. 只有 part recall 和 oracle 都显著高于单粒度，才值得投入 DAG + 查询解析；
   否则说明 SAM 粒度本身不够，需要换 2D 源（HQ-SAM/SAM2）或先做透明/薄结构。

---

## 6. 不建议再做

- 像素 BIoU 作为方法目标（只作评测/诊断）；
- 边界高斯/多峰分布/late-argmax（DIGS 核心，且 71.95 在前；我们的 oracle 仅 +0.040）；
- 单纯扩 top-k / 多选超点（全池 IoU 0.432 vs 0.553）；
- 在固定池上继续 rerank/gating（3D oracle 只 +0.08，2D 已近饱和）；
- 重做 class-agnostic instance ontology（DIGS 核心）。

---

## 7. 一句话

**ReLaGS 的“只选 2–3 个超点”是真问题，但解法不是多选、也不是边界精修；
是换单位：多粒度 part-whole 本体 + 查询条件化的粒度/集合选择。
边界高斯已经是 DIGS 的核心；part-level 才是它自己留下的空位。**