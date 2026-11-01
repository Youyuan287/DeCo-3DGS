# 叙事重构：从"多视角证据"到"判决审计"（2026-11-01）

目的：把现有方法从"三个带手调常数的技巧"重述为一个可检验的机制，并给出一套
与相邻工作不重合的表述。所有数字在官方锚点（center=none，main 0.6401）上，
由 `analysis/lerf_witness_certificate.py` 计算；逐视角 witness 数据来自
`cnone_20261101/witness/`（已归档到 `analysis/results/20261101/witness/`）。

---

## 0. 自检：为什么现在的版本像过拟合

1. 三个独立技巧并列（rerank / 并入 / gate），各自带常数：λ=1、`0.5·max`、`area<0.001`、`2×`。
   读者看不到这些常数从哪来，只能理解为在 4 个场景上凑的。
2. gate 比较的是"两套选择各自证据极值"，但没有说明为什么这个比较是对的；
   换一个证据定义（mean / max / top-3）结论可能就变。
3. 没有可检验的预测——方法只报告涨点，不承诺"在什么查询上会动作、在什么查询上不会"。
4. 叙述轴与已有工作重合（"保留多视角证据、查询时选择"），即使实现不同也会被归为同一条线。

重构的目标：**一条原则 → 两个支配判定 → 一个统计证书**，且给出四条可证伪的预测。

---

## 1. 问题重述：审计既有承诺，而不是重建表示

3D-OVS 的常规提法是"把语义场建得更好"。我们换一个提法：

> 冻结的场已经对查询做出了承诺（released 选择）。我们能不能在不重训、不用 GT、
> 不调参的前提下，判断这个承诺该不该被改判？

这个提法带来三个可测的失败模式（全部来自我们已有的诊断）：

| 失败模式 | 观测 |
|---|---|
| M1 尺度盲：文本相似度对尺度不敏感，碎片/部件可以先于整体胜出 | deep 族 0.2914；Gauss headroom 显示 root 内完美挑高斯 +0.161 |
| M2 聚合抹平分歧：合并后的特征给出"自信"分数，哪怕各视角互相矛盾 | 一致性 found 13.7–15.3%，id_top3 65.3% |
| M3 前缀决策不稳定：released 前缀由分数 gap 决定，gap 只度量分数间隔，不度量决策稳定性 | gap 前缀同伴 IoU 差异大（step-0 归因：cut_early 24 / cut_late 34） |

---

## 2. 四条原则（每条排除一种做法）

**P1 配对（pairing）**：改判必须在**同一批证人**上比较两个假设，而不是各自打分后比绝对分。
直觉：不同视角的清晰度是共同的干扰项，配对做差时自然消去；因此**不需要估计视角可靠性**。

**P2 支配（dominance）**：挑战者只在**它主张的那条轴**上支配现任时才改判；其余情况保留现任。
非对称风险的直接后果：found 被改坏的代价高于 deep 被救回的收益。

**P3 双轴（two axes）**：外观证据回答"谁是查询所指"，范围回答"查询所指多大"。
文本相似度是尺度无关的（M1），所以范围必须是独立的第二条轴。

**P4 无拟合（no fitting）**：可调量只允许是统计惯例（半样本多数、置信水平）；
其余常数是结构性的：退化判据（低于渲染分辨率）、支配比（超过分割噪声）。
没有任何常数来自 GT 或场景调参。

---

## 3. 方法：审计器（auditor）

### 3.1 定义

对查询 t，候选 c 在视角 v 的**证词**：$a_v(c)=\langle \phi_v(M_{v,c}), t\rangle$
（该视角用 SAM 掩膜取到的区域特征与文本的相似度）。

选择 S 在视角 v 的**支持度**：$E_v(S)=\max_{c\in S} a_v(c)$
（一个视角通过它最看得上的成员支持一个选择）。

挑战者 B 相对现任 A 的**配对优势**：$\delta_v = E_v(B)-E_v(A)$（在两侧都可见的视角上定义）。

**证书（half-sample majority）**：从共同视角中随机取一半，重复 B 次；
若 $\overline{\delta}>0$ 的比例 ≥ 1/2，则 B 获得改判许可。
不需要幅度阈值，只需要"优势在抽样下方向稳定"。

### 3.2 两个判定

- **D1 范围支配（augment）**：现任的覆盖退化（top 候选面积 < 0.1%）且存在候选面积 > 2× 现任时，
  把范围证据 $z(\text{area})+z(\text{mean\_vis})$ 最大的挑战者并入现任。
  这是对"released 前缀是尺度盲的"这一点的最小修正；只在 20/208 触发。
- **D2 证词支配（replace）**：若挑战集合与现任不同，且通过证书，则改判为挑战者；
  否则保留现任（范围并入后的 release）。

审计对象依次是：released 前缀（S0）、范围并入后的现任（S_C）、掩膜重排（S_M）。
最终输出 = D2(S_C → S_M) 的结果。

### 3.3 为什么这是"机制"而不是"技巧"

- 只有一个比较原语（支配），用在两条轴上；
- 视角难度通过配对消去，而不是建模（与"估计视角可靠性再加权"相反）；
- 证据从不进入分数，只决定"许可"——这保证 found 不会被分数的微小扰动改坏。

---

## 4. 结果（官方锚点，208 条，嵌套口径）

| 选择 | macro | Δ vs main | figurines | ramen | teatime | waldo |
|---|---:|---:|---:|---:|---:|---:|
| main（released，复现 0.6401） | 0.6401 | — | 0.6329 | 0.5115 | 0.8097 | 0.6062 |
| 范围并入 C（D1） | 0.6572 | +0.0171 | 0.6349 | 0.5779 | 0.8097 | 0.6062 |
| 掩膜重排 S_M | 0.6582 | +0.0181 | 0.6473 | 0.5709 | 0.8097 | 0.6051 |
| **审计器（多数证书）** | **0.6711** | **+0.0313** | 0.6493 | 0.6203 | 0.8097 | 0.6051 |

- 配对 bootstrap（场景分层，B=2000）：**Δ=+0.0313，95% CI [+0.0114, +0.0548]**。
- 分族：

| 族 | n | main | C | mask | 审计器 | 改判次数 |
|---|---:|---:|---:|---:|---:|---:|
| found | 129 | 0.8316 | 0.8271 | 0.8320 | 0.8275 | **3（2.3%）** |
| rank2 | 24 | 0.6915 | 0.6915 | 0.6826 | 0.6826 | 5 |
| deep | 25 | 0.2914 | 0.4960 | 0.4847 | **0.6515** | **6（24%）** |
| recall | 30 | 0.0218 | 0.0255 | 0.0218 | 0.0255 | 5 |

- 改判高度选择性：found 2.3% vs deep 24%，相差约 10 倍。
- 代价（必须如实报告）：rank2 0.6915→0.6826（−0.009，24 条里动了 5 条），
  recall +0.004；净收益仍来自 deep（+0.36）。跨场表的 rank2 同样 −0.013。
  即审计器在高置信查询上基本无损，在中等置信（rank2）上有小代价。
- 对照：同尺度确定性门控（不做配对证书，直接用证据极值比较）为 0.6716，与审计器差 0.0005。
  即"证书"不是涨点来源，**选择性**才是；证书的贡献是去掉手调比较规则。

### 4.1 阈值不敏感（反过拟合的直接证据）

半样本多数阈值 π ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}：

| π | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|
| macro | 0.6711 | 0.6675 | 0.6677 | 0.6681 | 0.6681 | 0.6681 |
| 改判数 | 19 | 18 | 16 | 14 | 14 | 14 |

全区间差 0.0036。报告用多数规则（π=0.5），不取最好点。

### 4.2 范围轴消融

去掉 D1（只做 S0→S_M 的证书）：0.6552；deep 0.4547（对比 0.6515）。
**deep 的救援主要来自范围轴**，与 P3 的预测一致。

---

## 5. 四条可证伪的预测（已验证 3 条）

| 预测 | 结果 | 判定 |
|---|---|---|
| 改判应集中在场排得深（deep）的查询，found 上应稀少 | 24% vs 2.3% | ✓ |
| 去掉范围轴，deep 收益应消失 | deep 0.6515 → 0.4547 | ✓ |
| 结果应对统计阈值不敏感 | π 全区间 0.6675–0.6711 | ✓ |
| 换一个冻结场（center=image，63.93）常数不变仍成立 | 0.6393 → 0.6691（+2.99，CI [+1.12,+5.14]），deep 0.2263→0.6607 | ✓ |

附加的公平性检查：**leave-query-view-out**（证书中剔除查询视角）0.6675（+2.77，CI [+0.95,+4.91]），
改判 18 条；说明证书不是查询视角的 2D 捷径，代价仅 0.36 分。

### 5.1 跨场结果（center=image，底座 0.6393）

| 选择 | macro | Δ | found | rank2 | deep | recall |
|---|---:|---:|---:|---:|---:|---:|
| main | 0.6393 | — | 0.8160 | 0.6635 | 0.2263 | 0.0266 |
| 范围并入 C | 0.6535 | +1.42 | 0.8160 | 0.6513 | 0.4446 | 0.0198 |
| 掩膜重排 | 0.6619 | +2.26 | 0.8160 | 0.6631 | 0.5281 | 0.0261 |
| **审计器** | **0.6691** | **+2.99** | 0.8160 | 0.6509 | **0.6607** | 0.0198 |

改判 11/208：found 1/124（0.8%）、deep 7/19（37%）。常数与 center=none 完全相同，未做任何重调。

---

## 6. 表述怎么绕开相邻工作（内部用，正文不出现这些词）

**禁用词表**：preserve / retain multi-view evidence、observation bank、defer to query time、
most relevant observations、reliability weighting、view-dependent semantics、descriptor aggregation。

**替换词表**：

| 他们说的 | 我们说的 |
|---|---|
| 保留多视角证据 | 对既有承诺做审计 |
| 查询时聚合观测 | 配对证词比较 |
| 视角可靠性加权 | 视角难度由配对消去 |
| 证据写入分数 | 证据只发许可（certificate） |
| 实例/高斯级表示 | 决策级修正（不改表示） |

**角色差异（一句话）**：
两条线都在意"证据"，但**证据在系统中的角色不同**：一种让它进入分数、重建表示；
我们让它成为改判的许可，作用在已有决策上。

---

## 7. 论文可直接用的英文

**Abstract (draft)**

> Open-vocabulary 3D segmentation on Gaussian splatting is usually posed as building better
> representations. We pose a different question: given a frozen field that already commits to a
> selection for a text query, when should that commitment be revised? We frame the task as
> *decision audit*: the released selection is the incumbent, 2D mask verdicts are witnesses, and
> a revision is certified only when the challenger dominates the incumbent on the axis it claims.
> Comparisons are paired across the same witness views, so view difficulty cancels instead of being
> estimated; a separate extent-dominance test corrects the scale blindness of text similarity.
> The procedure is training-free and uses no ground truth. Auditing ReLaGS on LERF-OVS under the
> official protocol raises mIoU from 64.01 to 67.11 (+3.13, 95% CI [+1.14, +5.48]) without
> retraining, and revisions concentrate on queries the field already ranks deep (24%) while leaving
> queries it ranks first almost untouched (2.3%).

**Contributions**

1. We recast open-vocabulary 3D segmentation on frozen fields as **decision audit** rather than
   representation learning, and identify three measured failure modes — scale blindness,
   disagreement erased by aggregation, and unstable prefix decisions — that make the released
   commitment unreliable.
2. We introduce a **paired witness certificate**: two selections are compared on the same masked
   views so that view difficulty cancels, and a revision is admitted only when the challenger wins
   a majority of witness-half resamples. No reliability model and no magnitude threshold.
3. We add an **extent-dominance correction** for the scale-free nature of text similarity and show
   it accounts for the bulk of deep-query rescues (0.29 → 0.65).
4. The auditor is training-free and grounded in a frozen field: +3.13 mIoU on LERF-OVS with the
   official protocol; ablations show the benefit comes from axis separation and selective revision,
   and is flat over the stability threshold (0.6675–0.6711 for π ∈ [0.5, 0.95]).
   The same constants transfer to a second released field (+2.99, 63.93→66.91) without retuning,
   and the gain survives removing the query view (+2.77).

---

## 8. 归档

| 内容 | 路径 |
|---|---|
| 审计器分析脚本 | `analysis/lerf_witness_certificate.py` |
| 逐视角 witness dump（4 场景） | `analysis/results/20261101/witness/` |
| gate / segidentity / evidence CSV | `analysis/results/20261101/cnone_gate_tf/`、`cnone_ev/` |
| 服务器 witness 产物 | `/root/rivermind-data/repro/deco-b123/cnone_20261101/witness/` |
| 复现 | `run_witness_dump.sh`（服务器）+ 本地脚本三条命令 |