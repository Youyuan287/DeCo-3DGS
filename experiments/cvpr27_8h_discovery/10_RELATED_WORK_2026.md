# 10 — 相关工作报告（2026-11-03，arXiv API 实检）

> 说明：本环境内置 WebSearch/WebFetch 不可用（返回空 / 域名校验失败），
> 以下均通过 `curl` 直连 `https://export.arxiv.org/api/query` 取得，标题/摘要为原文。

## 与我们诊断重合的（重要：诊断已不新颖）

**QAGaussian — Beyond Similarity Matching: Structured Reasoning for OV Referring Segmentation in 3DGS**
https://arxiv.org/abs/2608.16103
> "global text-region similarity ... is weak for queries involving attributes, reference objects,
> spatial relations, and fine-grained parts. This often causes target-reference confusion,
> **granularity mismatch, part-whole leakage**, and relation violations."

→ 我们本轮独立测到的"场语义单元与查询范围不对齐（碎片 vs 整体）"**已被这篇明确命名**。
他们的解法是**学习式** query-adaptive slots + 关系图 + 粒度自适应 router。
**留给我们的空间：training-free 版本的粒度对齐。** 诊断本身不能当贡献。

## 已经占住的三条路

| 工作 | 占住的东西 | 是否 training-free |
|---|---|---|
| DIGS (2610.16177) | 每高斯 top-M 标签分布 + class-agnostic 实例发现，解耦"发现"与"命名" | 是 |
| PTC / Perceptual Anchoring (2608.03991) | 用高置信预测构造视觉原型，再去校准文本嵌入（2D） | 是 |
| Target-Aware Refinement (2610.21619) | 目标文本嵌入注入 + 闭式系数；并指出"文本嵌入含公共成分，注入会让余弦读出变成 query-agnostic" | 是 |
| SCI-CLIP (2608.05627) | 区域一致交互图 + 参考记忆，用同一 region abstraction 贯穿各阶段（2D） | 是 |
| Regional Semantics + Visual Prototypes (2610.30623) | 改 CLIP 注意力 + 视觉原型分类器 | 否（可训练） |

→ **zR / a_g / 原型校准 / 区域图传播** 这几类在 2025–26 都有人做了。
   我们本轮的 zR(+5.62) 与 PTC 的"原型校准"是同一族；`a_g` 与 Target-Aware Refinement
   指出的"公共成分"问题同源。**不能再声称这些是新颖机制。**

## 尚未被占的（初步判断，需继续核对）

**训练-free 的"输出集合"保证。**
上述工作都在改**打分或表示**；没有一篇对"返回多少个区域、是否包含目标"给出统计保证。
保形风险控制（conformal risk control）在 2026 已有成熟工具（ReCIRC 2609.38112、
Macro Mondrian CRC 2610.35035），但**尚未用于 training-free OV-3D 区域集合**。

我们的审计器已经是一个**未经校准的稳定性检验**（半样本多数，π=0.5 是拍出来的）。
把它替换成保形化版本，可以直接给出"返回集合以 1−α 概率覆盖目标"的有限样本保证，
并把集合大小 g 从"分数间隔启发式"变成校准量。

## 待核实
- QAGaussian 是否开源、其 granularity router 是否可退化为 training-free。
- 是否有工作把 conformal 用到 3D 开放词表检索（继续用同一 API 检索 "conformal 3D retrieval"）。
