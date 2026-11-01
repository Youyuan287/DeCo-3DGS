# Part 多粒度：oracle 很强，选择器是墙

日期：2026-11-01
数据：LERF-OVS 四场景 208 查询；2D GT 为 polygon mask；
oracle 是 GT 选择的上界，不是可部署方法。

---

## 0. 结论

1. **多粒度 mask 的 oracle 非常强**：level-3 单 mask 0.635 → 四层单 mask 0.884
   → 贪心集合 0.936；part 查询 0.372→0.813→0.856；part+set 0.256→0.871→0.898。
   最佳层级在 0/1/2/3 上分布为 57/38/58/55，**没有固定粒度可用**。
2. **但训练-free 的文本选择器选不动**：全量 top1 0.511（低于 level-3 oracle 0.635），
   part top1 0.364、part+set top1 0.000；层级匹配率只有 37.5%。
3. **换更区域级的编码器也只部分改善**：AlphaCLIP（GRIT-1M）在 onion segments 上
   top1 3/7、top3 4/7；bear nose top1 1/3；knife 0/3。仍不足以支撑方法。
4. **多视角 mask track 只救回一部分**：tesla door handle 3→1、porcelain hand 4→1；
   pirate hat 31→68、old camera 仍 top-1 错。
5. **因此瓶颈不是 mask 粒度，而是 part-level language grounding。**
   多粒度本身不是方法；方法必须包含一个能定位 part 的模块，
   否则 oracle 的 +0.25 无法兑现。
6. **但 grounding 不是死路**：GroundingDINO 全图只给大 box；改成
   **parent crop + part box 中心 + 包含中心的最小多粒度 mask** 后，
   在 GT parent 下 true-part top1 达 0.93–0.99（pirate hat 0.93/0.94、
   bear nose 0.99/0.99、hooves 0.87），只有印刷 logo（dall-e brand）失败。
7. **新的瓶颈是 parent 定位**：用检测到的 parent 时，pirate hat 3/4 成功、
   bear nose 1/3 成功；失败都来自 parent top-1 box 选错或没检测到。
   所以方法应做 **parent 假设选择 + part 检测一致性**，而不是 CLIP top-k。
8. **端到端 subset 已有正向信号**：8 条 true-part 中 5 条 lift 到 0.90–0.965
   （gateE 仅 0.00–0.13）；GT parent 上界估算 **+2.1 → 0.697**。
   但检测 parent 只有 4/8，估计约 +1.6；**完整 208 官方评测尚未跑，
   最终指标还没有确认上涨。**

---

## 1. Oracle：2D 多粒度上界

在查询视角的四层 `seg_map`（全局 mask ID 0–244）里，对所有 mask 计算 IoU：

| 组 | n | L3 best | 四层单 mask | 贪心集合 | 平均集合大小 |
|---|---:|---:|---:|---:|---:|
| part | 16 | 0.372 | **0.813** | 0.856 | 2.62 |
| set | 34 | 0.729 | **0.910** | 0.931 | 3.12 |
| part+set | 9 | 0.256 | **0.871** | 0.898 | 3.33 |
| attr | 18 | 0.821 | 0.936 | 0.950 | 3.67 |
| whole | 131 | 0.644 | 0.880 | 0.948 | 3.45 |
| **全部** | **208** | **0.635** | **0.884** | **0.936** | **3.35** |

- 最佳层级分布：L0 57、L1 38、L2 58、L3 55；**固定 level-3 明显不够**。
- `part` 高 IoU 比例：L3 ≥0.5 仅 25%，四层单 mask ≥0.5 87.5%，集合 ≥0.8 81.2%。
- 集合通常 2–4 张 mask；不是“选一张更细的”，而是“按 query 选粒度/组合”。

---

## 2. 文本选择：raw CLIP 失败

用现成的 `language_features/{img}_f.npy`（245×512，per-mask CLIP）做 top-1 / top-k union；
选择不用 GT，只用 query 文本。

| 组 | n | top1 | top3 | top5 | oracle L3 | oracle all |
|---|---:|---:|---:|---:|---:|---:|
| part | 16 | 0.364 | 0.343 | 0.261 | 0.372 | 0.813 |
| set | 34 | 0.629 | 0.689 | 0.575 | 0.729 | 0.910 |
| part+set | 9 | **0.000** | 0.061 | 0.059 | 0.256 | 0.871 |
| attr | 18 | 0.693 | 0.874 | 0.714 | 0.821 | 0.936 |
| whole | 131 | 0.508 | 0.585 | 0.493 | 0.644 | 0.880 |
| **全部** | **208** | **0.511** | **0.586** | **0.489** | **0.635** | **0.884** |

- top1 层级匹配率：37.5%；part+set 33.3%。
- 典型失败：pirate hat（top1/5 全 0，oracle 0.929）、onion segments（0，oracle 0.795）、
  bear nose（0，oracle 0.989）、hooves（0，oracle 0.871）。
- 结论：**CLIP per-mask 分数不能选粒度，也不能选 part**；top5 union 反而常常更差。

### 2.1 模板 / parent 对比 / 3D 特征

- 模板 `a {} in the scene` 对 toy elephant 把正确 mask 从第 17 提到第 6，但 top1 仍错；
- `raw + parent score` 对 toy elephant 排到第 1（IoU 0.973），但 pirate hat/tesla 仍错；
- NAG 3D 特征：pirate hat rank 40、tesla 7、porcelain 3、old camera 9；没有稳定改善。

### 2.2 多视角 mask track

把查询视角 mask 提升到 3D，在 8 个邻近视角投影并找对应 mask，平均特征：

| query | 单视角 rank | 多视角 track rank |
|---|---:|---:|
| tesla door handle | 3 | **1**（IoU 0.839） |
| porcelain hand | 4 | **1**（IoU 0.519） |
| old camera | 102 | 11（top1 IoU 0.005） |
| pirate hat | 31 | 68 |

多视角能救“跨视角稳定的 part”，但救不了“由多块组成/文本歧义”的 part。

### 2.3 AlphaCLIP（区域级 CLIP）

用 `clip_b16_grit1m_fultune_8xe.pth` + mask alpha 输入：

| query | n | top1 | top3 |
|---|---:|---:|---:|
| onion segments | 7 | 3 | 4 |
| bear nose | 3 | 1 | 1 |
| knife | 3 | 0 | 0 |
| spoon | 1 | 1 | 1 |
| cabinet | 1 | 0 | 0 |

比 vanilla CLIP 好，但仍远不足以支撑“多粒度选择”作为一个方法。

---

## 2.4 GroundingDINO + parent crop：第一条可行路径

全图检测 part（`nose`/`hat`/`onion segments`）只给大 box：
bear nose 覆盖 100% 但 box IoU 0.03；pirate hat 的 hat box 全 0。
但改成 parent-conditioned 后：

1. 全图检测 parent（`stuffed bear`/`rubber duck with hat`/`sheep`）；
2. 裁 parent 区域放大；
3. 在裁剪图上检测 part（`nose`/`hat`/`hooves`）；
4. 对每个 part box 取中心，在多粒度 mask 里选**包含该中心的最小 mask**。

### GT parent（隔离 parent 误差）

| query | 帧 | part box best | 选中 mask IoU |
|---|---|---:|---:|
| pirate hat | frame_00041 | 0.597 | **0.929** |
| pirate hat | frame_00195 | 0.582 | **0.941** |
| bear nose | frame_00002 | 0.772 | **0.989** |
| bear nose | frame_00025 | 0.706 | **0.989** |
| hooves | frame_00025 | 0.713 | **0.871** |
| dall-e brand | frame_00107 | 0.520 | 0.083 |

**结论**：只要 parent 是紧的，part 检测 + 中心最小 mask 就能把 true-part 做到 0.87–0.99。
唯一失败的是印刷 logo（太小，SAM/检测器都没有对应 mask）。

### 检测到的 parent（完整链路）

parent 用 GroundingDINO 全图 top-1 box；part 用 parent crop 检测；中心最小 mask：

| query | 成功/总数 | 成功帧 top1 |
|---|---:|---:|
| pirate hat | 3/4 | 0.929 / 0.947 / 0.941 |
| bear nose | 1/3 | 0.989 |
| hooves（多实例） | 0/2 | top3 union 0.41 / 0.19 |
| dall-e brand（印刷） | 0/2 | 0.08 / 0.04 |

失败模式：

- **parent top-1 box 选错**：同一 prompt 的其它 box 其实 cov=1.0；
  这是 hypothesis selection，不是检测不到。
- **多实例**：hooves 有多只，top1 只选一只；需要多 box/多 mask 的集合选择。
- **印刷 logo**：2–3cm 的图案，SAM mask 和检测器都不可靠；应单独处理或排除。

多假设（多 parent prompt × 多 box + support 投票）能救回 3/4 pirate hat，
但大 mask 会拿到高 support；加面积先验又伤小 part。**parent 假设选择仍是开放问题。**

### 2.7 端到端 lift：subset 上确实涨了

用 **GT parent** 裁图、检测 part、中心最小 mask，再把 mask 提升到 Gaussian 并渲染：

| query | 帧 | gateE IoU | part 2D | part lift |
|---|---|---:|---:|---:|
| pirate hat | frame_00041 | 0.130 | 0.929 | **0.903** |
| pirate hat | frame_00105 | 0.013 | 0.054 | 0.041 |
| pirate hat | frame_00152 | 0.115 | 0.947 | **0.940** |
| pirate hat | frame_00195 | 0.066 | 0.941 | **0.906** |
| bear nose | frame_00002 | 0.001 | 0.982 | **0.953** |
| bear nose | frame_00025 | 0.001 | 0.989 | **0.965** |
| bear nose | frame_00107 | 0.001 | 0.000 | 0.000 |
| hooves | frame_00025 | 0.000 | NaN | NaN（GT oracle lift 0.878） |

- 5/8 条 lift 到 **0.90–0.965**；gateE 在这些查询只有 0.00–0.13。
- 按 208 条宏平均估算：这 8 条替换后 **约 +2.1 分 → 0.697**（GT parent 上界口径）。
- 用**检测到的 parent** 时只有 4/8 条能进 part 分支，估计约 +1.6 分；
  完整 208 条官方评测还没跑，所以**最终指标尚未确认上涨**。
- 失败：bear nose frame_00107（parent/part 选择错）、hooves（多实例，top1 只选一只）。

### 2.8 无 GT parent 推断：失败

parent 候选来自 class-agnostic 多粒度 SAM mask（level 2/3，面积 0.3%–45%，IoU-NMS 60 个）；
每个候选裁图检测 part，按 support 选 parent/mask。**不使用 GT parent 类别。**

| query | 成功/总数 | 说明 |
|---|---:|---|
| pirate hat | 0/4 | 大 mask 刷 support，选中面积 4k–9k，GT 1.8k |
| bear nose | 2/3 | 0.982 / 0.989；frame_00107 失败 |
| hooves | 0/2 | 多实例 + parent 选择失败 |
| dall-e brand | 0/2 | 印刷体 |

CLIP 给正确 parent 候选的排名（query / part / q+part）：

- pirate hat：3–54；bear nose：9–27；hooves：15–37；dall-e brand：3。
- 只有 4/11 条能把正确 parent 排进 top-4，**CLIP 不能做 parent 检索**。

**结论**：没有 GT parent 时，当前 CLIP/GroundingDINO + SAM 候选无法可靠推断 parent。
因此 2.7 的 +2.1 仍然是 **GT-parent oracle**，不是可部署方法结果；
最终 208 官方指标没有上涨。

### 2.9 3D 候选分解：part 存在，但只在 leaf 层

对每条 true-part，把 GT part 投到查询视角可见 Gaussian，计算它与 ReLaGS 各层超点的
最佳 IoU（不做文本选择，纯候选覆盖率）：

| query | 帧 | root best | **leaf best** | L1 best | L0 best |
|---|---|---:|---:|---:|---:|
| pirate hat | 00041 | 0.395 | **0.545** | 0.545 | 0.093 |
| pirate hat | 00105 | 0.467 | **0.606** | 0.606 | 0.110 |
| pirate hat | 00152 | 0.301 | **0.641** | 0.641 | 0.119 |
| pirate hat | 00195 | 0.509 | **0.605** | 0.605 | 0.107 |
| bear nose | 00002 | 0.069 | **0.826** | 0.490 | 0.045 |
| bear nose | 00025 | 0.075 | **0.943** | 0.568 | 0.054 |
| bear nose | 00107 | 0.089 | **0.874** | 0.578 | 0.056 |
| hooves | 00025 | 0.221 | **0.695** | 0.668 | 0.100 |
| hooves | 00043 | **0.871** | 0.392 | 0.372 | 0.046 |
| dall-e brand | 00107 | 0.211 | **0.788** | 0.470 | 0.173 |
| dall-e brand | 00129 | 0.199 | **0.779** | 0.494 | 0.183 |

**结论**：

- part **不是没有候选**：leaf 层 best IoU 0.55–0.94（bear 0.83–0.94、pirate 0.55–0.64、
  brand 0.78、hooves 0.70）；hooves 00043 是 root 0.87。
- **root 层经常没有 part**（bear root 0.07–0.09），而 ReLaGS 的候选生成是
  **root top-5 → 只有 leaf 相似度 > root 才提升**。part 的 leaf 相似度通常不高于
  root，所以 leaf part 根本进不了候选池。
- 这解释了为什么 2D/3D 的 part oracle 都很高，但实际查询几乎为 0：
  **不是特征不存在、不是聚合把特征洗没了，而是候选策略把 part 排除在池外。**
- 特征仍是第二瓶颈：即使 leaf 候选在池内，CLIP/AlphaCLIP/NAG 对 part 的文本排序
  只有 rank 3–60；但给定 parent/part 定位后，mask 选择能到 0.90–0.99。

### 2.10 特征、聚合、候选：三层结论

| 层 | 证据 | 结论 |
|---|---|---|
| **初始语义** | 单张 masked-crop CLIP；正确 part 的文本 rank 3–60；AlphaCLIP 也只部分改善 | **一开始就弱**：一个 mask 压成一个向量，丢掉上下文/空间/part-whole |
| **投影/聚合** | leaf→root 后 root 特征混入 parent/背景；root 候选常无 part；leaf 特征仍有 | **会再崩一次**，但不是唯一原因 |
| **候选策略** | leaf part 在 3D 池中 IoU 0.55–0.94；root 常 0.07–0.51；root-first + leaf>root 才提升 | **主因**：leaf part 进不了池 |

因此：

- 只加特征信息、不改候选策略：leaf part 仍然进不了池，指标不会动。
- 只改 leaf-first、不加特征：part 进池了，但文本 rank 仍可能是 3–60，选择不出来。
- 正确组合：**leaf-first 候选池 + part-aware 语义 enrichment + parent/part 假设选择**。

### 2.11 可行的语义 enrichment（不是简单多 crop 平均）

“盲人摸象”的根因是：每个 mask 只看到局部，且被压成单向量。可行信息：

1. **局部 + 上下文多视图**：masked crop、bbox context、1.5× context、full-image overlay；
   不要平均成一个向量，保留为多 token。
2. **part-whole 残差**：`f_leaf`、`f_parent`、`f_leaf - f_parent` 三个 token；
   残差专门编码“这个 part 相对 parent 多了什么”。
3. **多视角 track**：同一 mask 在邻近视角的对应 mask 特征，mean/top-k 作为额外 token；
   只对 track 一致的视角聚合，避免背景语义。
4. **几何 token**：mask 面积比、相对 parent 的位置、紧凑度、3D 法向/曲率；
   part 的“鼻子/帽子/蹄子”有强几何先验。
5. **sibling context**：同一 parent 下的兄弟 mask，做对比/排除，避免把 parent 当成 part。
6. **可选 caption**：VLM/LLM 给每个 mask 生成属性描述（DIGS 路线），但成本和协议风险高。

核心不是“更多特征”，而是 **保留多 token + 关系上下文**，用 late interaction 让 query
匹配最相关的 part token，而不是把 part 压进一个均值。

### 2.12 Enrichment 门控（先做，不直接上方法）

对 GT leaf 候选，比较以下表示的 part 文本 rank / top-k：

- A：当前 `_f`（masked crop 单向量）
- B：A + context crop token
- C：A + parent token + residual token
- D：A + 多视角 track token
- E：A + 几何 token
- F：B–E 全加，late interaction

通过标准：true-part top1 ≥0.5、top3 ≥0.7；且不伤 whole 查询。
只有 enrichment 把 rank 从 3–60 拉到 top-k，才值得接 leaf-first 候选池；
否则说明特征不是可救的，应该换 2D 源或转其他方向。

### 2.13 Enrichment 门控结果：rank 有改善，top-1 仍失败

11 条 true-part，GT-best mask 的文本 rank：

| 表示 | mean rank | median | top1 | top3 | top5 | top10 | top1 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| local (`_f`) | 53.4 | 35 | 2/11 | 2/11 | 3/11 | 3/11 | 0.116 |
| + context | 34.3 | 17 | 1/11 | 4/11 | 5/11 | 5/11 | 0.116 |
| + parent/residual | 39.3 | 16 | 2/11 | 2/11 | 4/11 | 5/11 | 0.100 |
| all（late interaction） | **27.2** | **5** | 1/11 | 4/11 | **6/11** | **6/11** | 0.100 |

- **有改善**：mean rank 53→27，top5 3/11→6/11；bear nose 13/35→3、pirate 78→17、
  dall-e brand 79→4。
- **但 top1 仍只有 1/11，top1 IoU 0.10**；正确 mask 常进 top5/top10，但不是第一名。
- **门控失败**：top1 ≥0.5、top3 ≥0.7 未达到。
- 结论：初始语义确实弱，context/parent/residual 能提高候选 **recall@5**，但
  **不能单独完成选择**；还需要一个 part-whole 选择器（多视角/几何/父子一致性），
  而且必须和 leaf-first 候选池一起用。

### 2.14 Part-whole 选择器门控：失败

对 enrichment top-50 候选补多视角 track（8 近邻视角）、parent/child、几何信号：

| 信号 | best-candidate mean rank | top1 IoU |
|---|---:|---:|
| text_score | 12.1 | 0.100 |
| track_good（cov≥.5 & cos≥.5 的视角数） | 18.2 | **0.175** |
| track_cos_mean | 31.0 | 0.083 |
| track_cov_mean | 15.2 | 0.032 |
| parent_score | 16.7 | 0.100 |
| resid_score | 20.6 | 0.048 |
| area_ratio | 30.5 | 0.084 |
| level | 18.3 | 0.005 |

- 单一信号最好的是 `track_good`，top1 IoU 仍只有 **0.175**；
  文本 0.100；没有任何组合规则把 top1 IoU 抬到 0.45 以上。
- 多视角 track 不具判别力：正确 part 和错误大 mask 都可能在多视角“一致”，
  因为 part 本身在多个视角可见；coverage 高不等于语义对。
- Parent/几何也区分不了具体 part；top50 里正确候选的 text rank 中位是 6，
  但选择器无法把它排到第一。
- **门控 FAIL**：top1 ≥0.5、top3 ≥0.7 未达到。

结论：在当前 **class-agnostic mask 特征 + 多视角几何** 下，part-level
“候选选择”本身不可解；不是再加一个 late-interaction token 能救的。
需要更强的 part 语义源（开放词汇检测器 parent 定位、AlphaCLIP/更强区域模型、
LLM/VLM 属性、或几何 part 分割），或者放弃 Part 作为主方法。

---

## 3. 现有方法在 part 上本来就很差

prior DIGS/CX frozen run（`digs_cx_four_20261030`，不是论文 71.95 的完整配置）：

| 组 | n | b0 | cx | cx≥0.5 |
|---|---:|---:|---:|---:|
| part | 16 | 0.177 | 0.166 | 18.8% |
| set | 34 | 0.767 | 0.767 | 88.2% |
| part+set | 9 | **0.000** | **0.000** | 0% |
| whole | 149 | 0.679 | 0.639 | 71.1% |

大量 part 查询（onion segments、bear nose、tesla door handle、hooves、pirate hat）
在现有方法里接近 0；而 2D oracle 能到 0.8+。
**这是当前方法共同留下的真实空位，不是 ReLaGS 特有的实现问题。**

按宏平均估算：part 16 条从 0.17 到 0.81 = +4.98；part+set 9 条从 0 到 0.87 = +3.77；
合计 **+8.7 mIoU 的上界**（2D 口径；渲染后会打折）。即使只兑现一半，也有 +4 分。

---

## 4. 这意味着什么

- **“多粒度”不是方法，是数据属性**：mask 池里确实有 part，且没有固定 level 可用。
- **方法是“part grounding + 粒度/集合选择”**：
  - 给定 query，先定位 part（box/point/region），再在多粒度 mask 里选；
  - 或者先定位 parent，再在 parent 内选 part；
  - 或者用多视角 track 把 part 的跨视角证据聚合起来。
- 纯 CLIP/AlphaCLIP 的 region-text 匹配不够；**parent-conditioned crop + 中心最小 mask 可行**：
  1. **开放词汇检测器**给 parent/part box（已验证 part box 可到 0.6–0.77）；
  2. **LLM 查询分解**（bear nose → parent=stuffed bear, part=nose；pirate hat →
     parent=rubber duck with hat, part=hat）；
  3. **多粒度 mask 选择**：part box 中心 + 包含中心的最小 mask；
  4. **多实例**（hooves）需要多 box/多 mask 的集合选择；
  5. **印刷 logo**（dall-e brand）当前失败，应单独处理或排除。
- 不建议把“多粒度 mask + CLIP top-k”直接写成方法：top1 0.511 < level-3 oracle 0.635，
  会被审稿人一眼看穿。

---

## 5. 建议的下一步（按风险/收益）

### A. parent 假设选择 + part 检测一致性（推荐，已部分验证）

- **已通过**：GT parent 下 true-part top1 0.87–0.99；检测 parent 下 pirate 3/4、bear 1/3。
- **未通过**：parent top-1 box 选错；hooves 多实例；dall-e brand 太小。
- 下一步：
  1. 用 **object-level proposals**（ReLaGS/DIGS 实例、GroundingDINO 多 box、多 prompt）
     作为 parent 候选，而不是 top-1 box；
  2. 对每个 parent 候选裁图检测 part，得到 part mask 假设；
  3. 用 **跨 parent 假设的一致性 + 多视角一致性 + 尺寸/位置先验**选最终 part mask；
     不要只用 support（大 mask 会刷支持），也不要只用面积（伤小 part）。
- 通过标准：true-part 查询 top1 ≥0.6、top3 ≥0.8；hooves 这类 multi-part 用集合 IoU；
- 然后扩展到全部 part/part+set 25 条，报告 2D→3D lift 后的 mIoU。

### B. 多视角 part track + part-whole DAG（训练-free，工程量中）

- 用 3D 投影在邻近视角建立 mask track；
- 用 track 平均特征 + parent-child 对比 + 集合选择；
- 目前 tesla/porcelain 有效，pirate/old camera 无效；需要更大规模验证。

### C. 转向 set/plural 查询（低风险，收益较小）

- set 组 top1 0.629、oracle 0.910、34 条；比 part 容易；
- 但现有方法 set 已经 0.767，绝对增益约 +1.5–2 分，且 novelty 较弱。

### D. 转向透明/薄结构（DIGS F1，窄）

- 透明盒/杯子；alpha compositing 泄漏；
- 问题清晰但覆盖面小。

---

## 6. 一句话

**多粒度 oracle 证明 part 的 2D 证据存在（0.81–0.87）；CLIP/AlphaCLIP/NAG 都选不动，
但 GroundingDINO + parent crop + box 中心最小 mask 在 parent 正确时能到 0.87–0.99。
Part 方向可行，但方法必须是 parent-conditioned part grounding + 假设选择，
不是“多粒度 mask + CLIP top-k”。**