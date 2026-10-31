# 依托 ReLaGS 框架，还有哪些值得挖的环节？

日期：2026-11-01
背景：当前最好 gateE 0.6760（DeCo B0）；文本侧、gating、一致性、读出已接近饱和。
下面按 ReLaGS 的实际流程逐段分析，只列**还没被我们系统碰过**的环节。

---

## 0. ReLaGS/DeCo 流程与已知状态

| 阶段 | 做什么 | 我们的状态 |
|---|---|---|
| 1. 几何剪枝 MWP | `max_weight_pruning` 0.0005 删低贡献高斯 | 未做端到端消融 |
| 2. 超点分区 | FRNN + graph weight + cut-pursuit → `nag-l1.pt` | 未改；官方分区与发布不一致（7852 vs 7936） |
| 3. 语义投影 merge_proj | 逐视角 mask 特征 → 超点特征；RATIO 0.3 / WEIGHT 1e-4；ROFA | ROFA/一致性已证伪；特征余弦 0.94–0.97 |
| 4. 层级区域生长 | `thres_connect 0.9,0.7,0.7`、`thres_merge 20` | 未改；重建层级 61–98% 可配对 |
| 5. 候选生成 | top-5 roots + leaf 提升 + gap | recall 29/208 池内最佳 **0.027** |
| 6. 选择/读出 | module C / mask rerank / gateE | 已充分探索；0.6760 |
| 7. 渲染/掩膜 | semantics>0.5；无 mask 精修 | 未做 2D mask refinement / MWP 消融 |
| 8. 评测 | query mIoU | 未做 instance-level / novel-view |

---

## 1. 最大头寸：候选生成（阶段 5）

**证据**：
- released pool oracle **0.7094**；greedy5 **0.7500**；greedy20 **0.7818**。
- recall 29 条：池内最佳 **0.027**；greedy20 **0.196**。
- 2D GT-best mask 提升上界：recall **0.2352**、deep **0.8247**。

**结论**：我们的所有读出方法都在“池内”工作，池外实例没有机会。候选生成是 ReLaGS 框架里
**最大、最未被触碰**的环节。

**可做的两步（由便宜到贵）**：
1. **top-k 扩池**：把 `pool(sims, 5)` 扩到 10/20，重新导出 `id_top3` 证据，跑 gateE。
   看 recall 是否从 0.027 上升；这是对“候选生成”最直接的验证。
2. **class-agnostic proposal**：不用文本 top-k，而是按区域支撑/可见性/几何一致性提出候选
   （例如所有支撑足够的 root/leaf，或 mask 图 track），再文本打分。
   这对应 QCMG-lite，但可以先只在“proposal recall”层面验证。

**预期**：如果 top-20 能把 recall 从 0.02 提到 0.1+，这是比任何 gating 都大的收益。

---

## 2. 层级区域生长（阶段 4）

**证据**：
- 官方 level-0 7852 vs 发布 `nag-l1` 7936；重建层级 ramen/teatime/waldo 只有 61–68% 簇可配对。
- 4% 的特征差被区域生长放大成层级差异；少数查询翻转导致 5–12 分差距。

**机会**：把区域生长从“特征相似度合并”改为**几何/mask 支撑驱动**的合并，
或用 query-agnostic 的 mask graph track 替代层级合并。
- 这与 QCMG 的核心相同，但可以作为 ReLaGS 的**分区/合并模块**局部替换，而不是重写整个框架。
- 风险/成本高；但它是“实例形成”的真正位置。

---

## 3. 语义投影/特征提取（阶段 3）

**证据**：
- 匹配簇特征余弦 0.94–0.97（不是 1.0）；我们自己的 re-extraction 只有 0.89–0.96。
- 4% 特征差 → 层级差异 → 5–12 分绝对差。

**机会**：改进 **per-view mask 特征提取**（masked pooling vs crop、分辨率、CLIP checkpoint、
fp16/fp32、mask beta 加权）。这不是新问题，但可能带来绝对提升，且与现有读出正交。
- 成本中等；需要重跑 merge_proj。
- 新颖性低（工程/特征质量），更适合作为 appendix 或 backbone 改进。

---

## 4. 渲染/掩膜精修（阶段 7）

**证据**：
- found 0.831、rank2 0.640、deep 0.661；2D lift 在 deep 上界 0.825、recall 0.235，远高于当前。
- 我们所有收益都在“选集合”，没有改渲染掩膜质量。

**机会**：
1. 用 2D mask 对选中候选做**边界/孔洞精修**（intersection/union/conditional random field）；
2. 多视角 mask 共识（lift 后投票）提升掩膜质量；
3. MWP/贡献剪枝对渲染掩膜的影响（ReLaGS 自带，但我们没做端到端消融）。

**预期**：对 found/deep 的 IoU 有直接提升，且不依赖候选生成；成本低–中。
**新颖性**：中等；需要与 2D-3D refinement 工作做 novelty check。

---

## 5. 视角/证据选择（阶段 3/6）

**证据**：
- 逐视角特征是区域混合；top-k 文本会选到背景；一致性失败。
- 几何支撑选择（coverage/visibility）在 2D mask 选择上 +6.6，但还没用于 3D 聚合。

**机会**：用 **coverage/visibility/几何支撑**选择参与聚合的视角/观测，而不是文本或一致性；
query-agnostic，协议安全。成本低；可能对 deep/teatime 有帮助。
- 我们已经验证“几何选 mask”有效但不够；把它用于 3D 聚合/权重可能更直接。

---

## 6. 评测与指标（阶段 8）

**证据**：query mIoU 隐藏 recall；29 条“完全没找到”只贡献 0.02。
**机会**：instance-level recall/precision、novel-view 评测、置信区间。
- 这是 benchmark/analysis 贡献，不是方法；但如果做 QCMG，必须有 instance-level 指标。

---

## 7. 推荐优先级

| 优先级 | 方向 | 成本 | 预期 | 新颖性 |
|---|---|---|---|---|
| **1** | **top-k 扩池（10/20）+ 现有读出** | 中 | recall 0.02→0.1+？ | 中 |
| **2** | **2D mask 精修/多视角共识** | 低–中 | found/deep IoU 直接提升 | 中 |
| 3 | 几何/coverage 选视角用于 3D 聚合 | 低 | deep/teatime | 中低 |
| 4 | class-agnostic proposal / mask graph（QCMG-lite） | 高 | recall + macro | 高 |
| 5 | 特征提取/backbone 改进 | 中 | 绝对提升 | 低 |
| 6 | MWP/剪枝消融 | 低 | 掩膜质量 | 低 |

**建议先做 1**：它最便宜、直接验证“候选生成”这个最大头寸。
如果 top-20 扩池能把 recall 拉起来，再投入 QCMG-lite；如果扩池也救不了，
说明问题在 proposal 质量，需要换 class-agnostic 图。

---

## 8. 不建议再做的

- 继续调 gating / λ / 融合权重；
- 继续在固定池里 rerank；
- 文本侧模板/lexicon/场景词表（已证伪或违规）；
- 一致性/中心化/白化进入评分。

这些都已到天花板。