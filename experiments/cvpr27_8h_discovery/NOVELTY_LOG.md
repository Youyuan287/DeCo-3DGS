# NOVELTY LOG

按 `LITERATURE_PROTOCOL.md` 记录。查询均通过 `tools/litcheck.py`（arXiv API，https）。
判定：OCCUPIED / ADJACENT / OPEN / UNVERIFIED。

---

## 2026-11-03 02:50 — "OV-3D 的失败主要是粒度/范围不对齐（碎片 vs 整体）"

- 查询：`cat:cs.CV AND (all:"open-vocabulary" AND all:"3D" AND all:segmentation)`；`all:"granularity mismatch"`
- 命中：[QAGaussian, arXiv:2608.16103](https://arxiv.org/abs/2608.16103)（2026-08-17）
  摘要原文列出 "granularity mismatch, part-whole leakage, target-reference confusion, relation violations"
- **判定：OCCUPIED（诊断层面）**
- 我们的 delta：仅剩"training-free 版本"。他们的解法是学习式 query-adaptive slots + 粒度路由。
- 剩余风险：若把这条当主贡献，会被直接指为已知问题。

## 2026-11-03 02:50 — "逐类/逐区域分数重标定（zR、a_g、原型校准）"

- 命中：
  - [PTC / Perceptual Anchoring, arXiv:2608.03991](https://arxiv.org/abs/2608.03991)：高置信预测→视觉原型→校准文本嵌入（training-free, 2D）
  - [Training-Free Target-Aware Refinement, arXiv:2610.21619](https://arxiv.org/abs/2610.21619)：文本注入 + 闭式系数；明确指出文本嵌入含公共成分、会使余弦读出变成 query-agnostic
- **判定：OCCUPIED（机制族）**
- 我们的 delta：`a_g` 用固定通用词表做**与查询无关**的离线估计，这一具体形式未必有人做，
  但"去掉共享成分/原型校准"整体上已被占。
- 剩余风险：高。这解释了为什么我们的 zR(+5.62) 与 `a_g`(+0.97) 都显得"不够新"。

## 2026-11-03 02:50 — "training-free 实例发现 + 命名解耦（3DGS）"

- 命中：[DIGS, arXiv:2610.16177](https://arxiv.org/abs/2610.16177)
- 摘要要点：每高斯 top-M 标签分布 + class-agnostic 实例本体，解耦发现与命名，late-argmax 渲染恢复边界
- **判定：OCCUPIED**
- 我们的 delta：无（本轮的"包含关系升格"与其实例发现重叠）。
- 剩余风险：若继续做实例/粒度，必须与 DIGS 做受控对比，否则无法主张。

## 2026-11-03 02:50 — "同一 region abstraction 贯穿各阶段（图传播）"

- 命中：[SCI-CLIP, arXiv:2608.05627](https://arxiv.org/abs/2608.05627)（training-free, 2D）
- **判定：OCCUPIED（2D）**，3D 侧的对应物未检出，但属 ADJACENT 边缘。
- 我们的 delta：若要搬到 3D，需证明跨视角证据带来的额外收益。

## 2026-11-03 02:52 — "对返回的 3D 区域集合给出覆盖保证（conformal risk control）"

- 查询：
  - `cat:cs.CV AND (all:"conformal" AND all:segmentation)`（2025-01-01 起，10 条）
  - `cat:cs.CV AND (all:"conformal" AND all:"3D" AND all:"open-vocabulary")` → 0 条
- 命中：cs.CV 内的 conformal + segmentation 全部是 **2D/医学**（甲状腺结节、体积风险、
  路面可达性等），没有 3D 开放词表区域集合。
- **判定：OPEN（弱证据，2 个查询）**
- 我们的 delta：把审计器的半样本多数证书替换为保形校准，给出"集合以 1−α 覆盖目标"的
  有限样本保证，并让集合大小 g 由校准决定而非分数间隔启发式。
- **剩余风险（写死）**：
  1. 覆盖 ≠ IoU，指标未必涨；
  2. 4 场景 LOSO 的交换性站不住，保证是名义上的；
  3. 必须再查 `"conformal retrieval"`、`"conformal set-valued prediction 3D"` 两组才可升级结论。

## 2026-11-03 02:53 — "OV-3D 的集合级选择（set selection）"

- 查询：`all:"open-vocabulary" AND all:"set selection"` → 0 条（无 cat 限制）
- 邻接命中：DIGS（实例发现）、QAGaussian（粒度路由）、AOV-GS（active mapping）
- **判定：ADJACENT**
- 我们的 delta：必须以"集合级目标函数 + 可证伪的对照"形式提出，不能只说"我们选集合"。

---

## 汇总（截至 2026-11-03）

| 我们曾主张的 | 现状 |
|---|---|
| 粒度不对齐是核心矛盾 | 已被 QAGaussian 命名，**不能当诊断贡献** |
| zR / 行偏置 / 原型校准 | 机制族已被 PTC、Target-Aware 占据 |
| 实例发现 / 包含升格 | 已被 DIGS 占据 |
| 局部对比 / 视角中心化 | 无人明确占据，但**我们自己实测为负**，不成立 |
| **集合级覆盖保证（conformal）** | **唯一尚未被占、且有理论工具可借的方向**（证据弱，需继续查） |