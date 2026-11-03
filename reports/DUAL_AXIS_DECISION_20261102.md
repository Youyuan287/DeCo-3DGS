# 双轴判决分解：统一 LERF 审计与 ScanNet 校准（2026-11-02）

本文档汇总本轮全部实验、数据与分析，并把它们收拢为一个统一框架。所有数字来自本仓库
`analysis/` 下的脚本与 `analysis/results/20261102/` 下的结果文件；服务器产物在
`fj01:/root/rivermind-data/repro/deco-b123/`。

---

## 0. 一句话结论

冻结的 3D 开放词表场对每个判决单元给出一个得分矩阵 `S(u, a)`（u=单元，a=候选/类别）
和逐视角证据 `W(u, a, v)`。判决误差只有两个正交来源：

1. **沿 alternative 的共享偏置 `b(a)`**：依赖候选/类别、与单元无关。它改变 `argmax`。
2. **沿 view 的排序不稳定**：聚合分数自信，但各视角互相矛盾。它影响"选择是否可靠"。

C1/C2 是 `b(a)` 的估计器；决策审计是第二个来源的检验器；placebo 门决定第二个来源是否可检验。
两个数据集不是两次独立实验，而是同一机制的两个 regime。

---

## 1. 统一框架

### 1.1 算法

```
输入：冻结场 S(u,a)、逐视角证据 W(u,a,v)

1) 估偏置   b(a) ← mean_u S(u,a)              (C1；有类表时用文本库精确化 = C2)
             S' ← S − b
2) 验证据   placebo：在 S 的高置信单元上，证人必须与场一致
             不通过 → 跳过第 3 步（审计弃权）
3) 改判     对每个单元，挑战者通过配对证书才替换 incumbent

输出：S' 的判决 + 是否启用审计的标志
```

### 1.2 三条可证伪预测

| 预测 | LERF-OVS | ScanNet-10 |
|---|---|---|
| P1 单 query 下减 `b` 是恒等变换 | ✓ 173/208 query 逐位相同 | — |
| P2 闭集下减 `b` 改变 argmax | — | ✓ 40.79 → 53.67 pooled |
| P3 证人无效时审计必须弃权 | 门通过，审计 +3.13 | 门拒绝（74–84% 不相容），审计零结果 |

### 1.3 为什么这是定理而不是技巧

单 query 下，候选得分 `⟨f_c − m, t⟩ = ⟨f_c,t⟩ − ⟨m,t⟩`，第二项与候选 `c` 无关，
所以排序不变（P1）。多类下同一项变成 `⟨m, t_a⟩`，依赖类别，所以改变 argmax（P2）。
两个数据集的行为差异由代数决定。

证书是视角可交换下的精确符号检验：零假设下配对优势必须中心化，`π=0.5` 由多数规则推出。
placebo 门检验的正是这个前提（P3）。

### 1.4 与 query-time calibration 的关系

arXiv 2610.30908（Engel et al., 2026-10-22）提出：减去共享场景方向（视觉特征与文本原型
都减）、用通用参考词表白化文本原型、soft clustering + CSLS + relevance margin，
报告在 7 种 lifting 方法与 4 个数据集上同时改善 multiclass 与 single-query。

在本文框架中，这些操作是 `b(a)` 的另一种估计器，属于第 1 步；他们的方法不包含第 2、3 步。
因此正面对比应做成 2×2：

| | 审计 off | 审计 on |
|---|---|---|
| 校准 off（官方 anchor） | 0.6401 | **0.6711（本文审计）** |
| 校准 on（对手式标定，X） | X | **Y（框架叠加）** |

若 `Y > max(0.6711, X)`，主张即为"标定与审计正交，可叠加"。X 尚未复现，是下一步。

---

## 2. ScanNet：零中心审计 pilot

### 2.1 设置

- released 场（ReLaGS，C1/C2 off）：macro **0.4036**；C1+C2 on 臂：macro **0.5018**。
- 现任：released argmax；证人：逐视角 region token。
- 偏差估计：只用场自己最自信的 top-q root（按 released margin），**不用 GT**。
- 变体先验固定；GT 只用于评分。
- 10 场景，K19（18 个有 GT 的类），macro = 场景均值。

### 2.2 released 场结果（Primary）

| 变体 | macro | Δ | 改判数 | 命中率 |
|---|---:|---:|---:|---:|
| baseline | 0.4036 | — | — | — |
| audit95（原版） | 0.0520 | −35.16 | 512 | 3% |
| cal95（原版中心化） | 0.3768 | −2.68 | 224 | 6% |
| plac95 q=0.25 | 0.3805 | −2.31 | 234 | 9% |
| plac95 q=0.5 | 0.3870 | −1.66 | 230 | 8% |
| **plac + 低 margin 门控** | **0.4073** | **+0.37** | 130 | 11% |
| incb95（按现任类条件） | 0.3592 | −4.43 | 236 | 7% |
| resid95（逐类仿射校准） | 0.2286 | −17.50 | 349 | 4% |
| witness_top1 (top3) | 0.0273 | −37.63 | 1471 | 12% |

+0.37 不是证据：

- 5/10 场景为正；
- 全部收益来自 scene0400 的 5 次改判（+10.40），被 scene0000（−4.55）抵消；
- bootstrap 95% CI **[−1.48, +2.86]**，包含 0；
- 去掉 scene0400 后 **−0.74**。

### 2.3 C1+C2 on 臂（Secondary）

| 变体 | macro | Δ |
|---|---:|---:|
| baseline | 0.5018 | — |
| plac + 门控 | 0.4892 | −1.26 |
| plac95 q=0.25 | 0.4649 | −3.69 |
| cal95 | 0.4591 | −4.27 |
| audit95 | 0.0529 | −44.89 |

在 C1+C2 之上再审计只有伤害：C1/C2 已消掉 `b(a)`，第 3 步没有空间。

### 2.4 证人诊断

| 视角聚合 | detection | false-alarm | 证人 top-1 macro |
|---|---:|---:|---:|
| mean | 6.1% | 82.1% | 0.008 |
| max | 14.3% | 74.0% | 0.020 |
| top-3 | 14.1% | 71.4% | 0.027 |
| 加权 mean | 5.4% | 83.6% | 0.007 |

- GT-free placebo 检验（on 臂，场最自信的 top-25% root）：证人 disagreement
  mean **84.1%**、max 75.8%、top-3 73.7%。
- 即：不用 GT 就能预先判断这个证人不可用。

### 2.5 root 纯度

| scene | roots | mixed≥2 类 | 混合质量占比 | 平均最大类占比 |
|---|---:|---:|---:|---:|
| scene0000_00 | 380 | 0.355 | 0.381 | 0.874 |
| scene0062_00 | 92 | 0.272 | 0.140 | 0.908 |
| scene0070_00 | 228 | 0.325 | 0.388 | 0.881 |
| scene0097_00 | 88 | 0.295 | 0.396 | 0.888 |
| scene0140_00 | 366 | 0.325 | 0.465 | 0.891 |
| scene0200_00 | 64 | 0.266 | 0.563 | 0.905 |
| scene0347_00 | 198 | 0.258 | 0.245 | 0.912 |
| scene0400_00 | 68 | 0.382 | 0.495 | 0.858 |
| scene0590_00 | 192 | 0.411 | 0.352 | 0.849 |
| scene0645_00 | 275 | 0.360 | 0.426 | 0.866 |
| **mean** | | **0.325** | **0.385** | **0.883** |

32.5% 的 root 混有两类以上，覆盖 38.5% 的高斯——这是候选/粒度层的真实头寸。

---

## 3. ScanNet：校准扫描（围绕 C1）

### 3.1 官方拆分（RESULTS.md，dataset-pooled，18 类分母）

| 变体 | mIoU | mAcc |
|---|---:|---:|
| ReLaGS published | 41.26 | 57.29 |
| Reproduced | 40.79 | 56.69 |
| **C1 image-side only** | **53.67** | 66.67 |
| C2 templates only | 56.32 | 68.11 |
| C1+C2 renorm=1 | 57.85 | 69.07 |
| **C1+C2 renorm=0** | **58.55** | 69.82 |

C1 单独 +12.88；C2 单独 +15.53；合计 +17.76（次可加）。

### 3.2 分数级校准扫描

在 released 分数矩阵上逐场景施加校准（GT 只评分）。**注意**：本扫描的 pooled 是
"逐场景 present 类计数相加"，与 `aggregate.py` 口径不同，故只比较 Δ，不比较绝对值。

released 场（raw macro 0.4036 / pooled 0.4388）：

| 校准 | macro | pooled | 说明 |
|---|---:|---:|---|
| center（分数级 C1） | 0.4524 (+4.9) | 0.5039 (+6.5) | 均值减法 |
| zscore | 0.4598 (+5.6) | 0.4969 (+5.8) | 逐类标准化 |
| median | 0.4519 (+4.8) | 0.5017 (+6.3) | |
| pc1 / pc2 / pc3 | 更差 | 更差 | 去二阶结构伤人 |
| rankq | 0.4465 (+4.3) | 0.4681 (+2.9) | |
| **Sinkhorn（均匀先验匹配）** | **0.4824 (+7.9)** | **0.5318 (+9.3)** | 8/10 场景为正，LOO 最低 +5.2 |

C1+C2 on 臂（raw macro 0.5018 / pooled 0.5504）：

| 校准 | macro | pooled | 说明 |
|---|---:|---:|---|
| **Sinkhorn** | **0.5101 (+0.83)** | **0.5534 (+0.30)** | 6/10，LOO 最低 +0.08 |
| zscore | 0.5033 (+0.15) | 0.5216 (−2.9) | |
| center | 0.4892 (−1.26) | 0.5406 (−1.0) | |

### 3.3 为什么加不上去

Sinkhorn 在 released 场上的收益集中在背景/先验类：

| class | raw | sinkhorn | Δ |
|---|---:|---:|---:|
| shower curtain | 0.00 | 74.52 | +74.52 |
| wall | 3.01 | 43.92 | +40.92 |
| sofa | 50.17 | 81.35 | +31.18 |
| floor | 60.45 | 82.57 | +22.13 |
| sink | 9.52 | 31.19 | +21.67 |
| desk | 34.57 | 47.20 | +12.63 |
| bathtub | 68.67 | 38.58 | −30.09 |
| counter | 28.55 | 10.70 | −17.85 |

而官方 C1 已把 wall 做到 52.85。在 on 臂上 Sinkhorn 的真实增量是 desk +11.0、door +2.9、
picture +2.5，同时 bathtub −8.0、chair −5.8。**C1 与 Sinkhorn 修的是同一个 `b(a)`，
不是两个独立增益。**

### 3.4 剩余错误在哪

on 臂逐类 IoU：counter 27.2、sink 33.7、desk 35.5、picture 35.7、window 1.6。
这些是小物体/薄结构，叠加 §2.5 的混合 root 统计，指向粒度而非分数刻度。

---

## 4. LERF-OVS：认证式查询扩展 pilot

### 4.1 动机

C2b（同义改写扩展）形式上是 class-agnostic 的。PROTOCOLS.md 已测量：无条件 63 条
lexicon 在单 query 下 −14.7。假设：**用决策证书筛选改写**可以修复。

### 4.2 设置

- 4 场景、208 条 (image, query)；lexicon 63 条，覆盖全部 query。
- 在原候选集上为每个改写额外 dump 逐视角相似度（`*_witness_para.csv`）。
- 三种选择：released prefix（main）、全部改写（uncond）、证书通过才并入（cert）。
- 证书：配对半样本多数（π=0.5），用原 query 的逐视角证据比较 incumbent 与挑战集。
- GT 只评分；oracle = 逐 query 用 GT 选最好的 variant（诊断上界）。

### 4.3 结果

| 场景 | main | uncond | cert | oracle |
|---|---:|---:|---:|---:|
| figurines | 0.6329 | 0.6329 | 0.6329 | 0.6329 |
| ramen | 0.5115 | 0.5115 | 0.5113 | 0.5231 |
| teatime | 0.8097 | 0.8097 | 0.8097 | 0.8097 |
| waldo_kitchen | 0.6062 | 0.5319 | 0.5319 | 0.6062 |
| **mean** | **0.6401** | **0.6215 (−1.86)** | **0.6214 (−1.87)** | **0.6430 (+0.29)** |

- main 精确复现官方 anchor 0.6401。
- cert 只改了 **3 条 query，全部变差**。
- 即便逐 query 用 GT 选最好 variant，也只有 **+0.29**。

### 4.4 结论

- paraphrase 证据与 released 场高度重叠，没有可利用头寸；
- 决策证书比较的是"同一证据源下两个选择的支持度"，**不能验证一个新证据源是否可信**；
- 对 paraphrase 这类新证据源，正确的工具是 placebo 有效性门，而不是支配证书。

C2b 关闭；C2 整体保持闭集补充实验。

---

## 5. 诊断工具：证人有效性门

任何审计/扩展在启用前都应通过：

```
placebo set = 场最自信的一批单元（按 released margin）
指标       = 证人与场在这些单元上的一致率
判定       = 一致率低于阈值（如 80%）则拒绝使用该证人
```

实测：

| 数据集 | placebo disagreement | 结论 |
|---|---:|---|
| LERF-OVS | 高置信 query 上改判率 2.3% | 证人可用 |
| ScanNet（on 臂） | 74–84% | 证人不可用 |

这个门是 GT-free 的，并且**预测了** ScanNet 审计的零结果——不需要跑完才知道。

---

## 6. 结论

1. **两个数据集是同一机制的两个 regime。** LERF：`b(a)=0`（可证），只剩视角不稳定，
   审计有效（0.6401→0.6711）。ScanNet：`b(a)` 主导，校准有效（40.79→58.55）；
   证人无效，审计被门拒绝（最好变体 +0.37，CI 含 0）。
2. **C1 与 Sinkhorn 是同一轴的不同估计器，不叠加。** 在 C1+C2 之上只剩 +0.83 macro。
3. **C2b 没有头寸。** oracle +0.29，认证后 −1.87。
4. **剩余指标头寸在粒度轴。** 32.5% 混合 root、38.5% 质量、小物体类 IoU 27–36。
5. **与标定论文的关系。** 他们的方法与 C1 同属 `b(a)` 轴；本文的第 2、3 步是正交的，
   应通过 X/Y 2×2 表验证可叠加性。

---

## 7. 产物与复现

### 脚本（本仓库）

| 内容 | 路径 |
|---|---|
| ScanNet 零中心审计 pilot | `analysis/scannet_nullcenter_pilot.py` |
| ScanNet 分数级校准扫描 | `analysis/scannet_calibration_sweep.py` |
| LERF 改写 witness dump | `analysis/lerf_expansion_dump.py` |
| LERF 认证式扩展评估 | `analysis/lerf_expansion_eval.py` |
| 既有审计脚本 | `analysis/lerf_witness_certificate.py`、`analysis/scannet_audit_eval.py`、`analysis/scannet_audit_onarm.py` |

### 结果（本仓库）

| 内容 | 路径 |
|---|---|
| 零中心审计（on 臂） | `analysis/results/20261102/scannet_nullcenter/` |
| 零中心审计（released 场） | `analysis/results/20261102/scannet_nullcenter_off/` |
| 校准扫描（两臂） | `analysis/results/20261102/scannet_calibration_sweep/` |
| LERF 查询扩展 | `analysis/results/20261102/lerf_query_expansion/` |
| 既有 LERF 审计结论 | `reports/REFRAME_DECISION_AUDIT_20261101.md` |
| 既有 ScanNet 负结果 | `reports/SCANNET_AUDIT_NEGATIVE_20261101.md` |

### 服务器（fj01）

| 内容 | 路径 |
|---|---|
| ScanNet 数据与分数 | `/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029/` |
| 零中心审计结果（on/off） | `.../scannet_nullcenter_pilot_20261102/`、`.../scannet_nullcenter_pilot_off_20261102/` |
| 校准扫描结果 | `.../cal_sweep_v2_off_off/`、`.../cal_sweep_v2_on_on/` |
| LERF 改写 witness | `.../cnone_20261101/witness_para/` |
| LERF 扩展评估 | `.../cnone_20261101/witness_para/eval/` |
| 运行脚本 | `.../run_scannet_audit_onarm.sh` 等 |

### 复现要点

- ScanNet 必须区分 released（`v2_off_off`）与 C1+C2 on（`v2_on_on`）两臂；
  迁移性结论以 released 场为准，on 臂只用于说明"校准后再审计无空间"。
- LERF 扩展评估中，witness 缺失的 query **不能跳过**，否则 main 会虚高
  （waldo 0.6745 vs 官方 0.6062）。修正后 main 与官方 anchor 逐位一致。
- 校准扫描的 pooled 口径与 `RESULTS.md` 的 `aggregate.py` 不同，只比较 Δ。

---

## 8. 下一步（最小闭环）

1. 实现对手式标定 baseline X（双侧场景方向减法、通用词表白化、CSLS），
   在同一 LERF 协议下得到 X。
2. 在 X 之上跑审计，得到 Y；报告 2×2 表。
3. 用同一个算法入口跑两个数据集，验证 P1/P2/P3 三条预测。
4. 测实例级 oracle 上界，决定粒度轴是否值得投入。