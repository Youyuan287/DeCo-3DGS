# DeCo-3DGS / ReLaGS / LERF-OVS 项目总交接文档

日期：2026-11-01
用途：切换窗口/交接给下一个分析者。本文档把**核心目标、协议、已完成实验、实验设计、结果、想证明的结论、失败原因、当前最好复现、下一步计划、数据与代码位置**集中在一处。
原则：所有数字注明口径；**不同 baseline、不同读出方式、GT oracle 与方法结果不能混用**。

---

## 0. 一页摘要（先读这里）

### 0.1 核心目标

做一个 **CVPR 级别的 training-free 开放词汇 3D 分割方法**，不是给 ReLaGS/DeCo 打补丁。
必须在 **LERF-OVS** 上带来可复现的 mIoU 增长（目标 >0.70，最好 >0.72），并最好在 ScanNet/3D-OVS 或 novel-view 上验证。
硬约束：

- LERF-OVS 必须逐 query 开放词汇；**禁止用场景其他类别/class bank 做推断**。
- 方法必须 training-free；GT 只能用于 evaluation/headroom，不能进入选择。
- 严格协议：LOSO/预注册门控；不允许 GT 监督 reranker 当方法。
- 不做“在固定池上调 gating/λ/融合权重”的增量；要找方法层面的关键问题。

### 0.2 当前最好结果（2026-11-01 晚修正）

**training-free 主结果：官方锚点 0.6401 → gateE-TF 0.6716（+3.15，CI [+0.0120,+0.0539]）。**
口径：路线 A 官方权重 `ref/relags_hf` + center=none + 官方评测；
详见 `reports/CONTRIBUTION_REANCHOR_20261101.md`。

**原 0.6760 仍存在，但含学习组件**：module C 的剪枝用的是冻结 LOSO GBM 的预测 IoU
（`lerf_subset_20261030/preds.csv` 的 `pred` 列），gateE 回退分支是 S_C，
所以 0.6760（底座 0.6393 = DeCo 选簇策略移植）**不能作为 training-free 方法结果**，
只能作"学习选择层"消融/上界。

三个数绝对不能混：

| 口径 | 配置 | 四场景 scene mIoU | macro |
|---|---|---|---:|
| **路线 A（官方锚点）** | `--center none`，官方读权 | 63.29 / 51.15 / 80.97 / 60.62 | **0.6401** |
| 策略移植（原称 B0） | `--center image`，DeCo 选簇策略套 ReLaGS 场 | 58.5869 / 51.8076 / 79.3896 / 65.9209 | 0.6392625 |
| 路线 B（未本机复核） | 我方重建语义场 + gap 补丁（hn01） | — | 0.6471 |

gateE-TF 分场景：figurines 0.6493 / ramen 0.6221 / teatime 0.8097 / waldo 0.6051。
原 gateE（center=image，学习器）分场景：0.6129 / 0.6092 / 0.8223 / 0.6597。

### 0.3 最终指标状态（必须说清）

- **training-free 最好：官方 208 条 0.6716（+3.15 vs 0.6401）**，已通过配对 bootstrap 与分族门控。
- 原 0.6760 含 LOSO 学习器（module C），只能作学习层消融；0.6393 不是 DeCo 基线，是策略移植。
- Part 方向的最新 +2.1 是 **GT-parent oracle 上界**，有数据泄露，不能算方法结果。
- 无 GT parent 推断失败；part-level 特征/选择器门控失败。
- 因此可报的 leakage-free 增长是 **+3.15（0.6716）**，仍未到 >0.70；分析/诊断贡献仍成立。

### 0.4 已经证明了什么

1. **读端/选择层已经接近饱和**：固定池 oracle 0.7094，当前 0.6760 已接近池内上限。
2. **瓶颈在候选/实例生成**：recall 29/208 池内最佳仅 0.027；greedy20 0.7818。
3. **一致性/中心化/白化不能用于评分**：found 命中从 65.3%/52.4% 掉到 13.7%/15.3%。
4. **多 token late interaction 有效**：单向量 +0.56，late interaction +2.26，gateE +3.66。
5. **边界不是主因**：好的 2D mask lift 后仍损失约 0.06 BIoU；边界假阳性全删只有 +0.04 IoU3D；DIGS 已占边界。
6. **ReLaGS 的候选单位是超点**：中位 2 个、87.5%≤3；扩池是负收益（0.553→0.432）。
7. **Part 在 2D/3D leaf 候选里存在**：2D part oracle 0.813、集合 0.856；3D leaf best IoU 0.55–0.94；root 常 0.07–0.51。
8. **Part 的主因是候选策略**：root-first + leaf>root 才提升，把 leaf part 挡在池外；特征是第二瓶颈，聚合第三。
9. **Part grounding 仍不可部署**：CLIP/AlphaCLIP/NAG/GroundingDINO/多视角 track 都选不动；
   GT parent 下 part mask 选择 0.90–0.99，但无 GT parent 推断失败。

### 0.5 还没证明什么

- 没有任何方法把 208 条推过 0.70；training-free 当前 0.6716（官方锚点 +3.15），已排除学习器。
- 没有可部署的 part-level 方法（parent 推断、part 选择、多实例集合均未过门控）。
- ScanNet 上没有正迁移（late interaction 负迁移）。
- C1-off/center-image 的受控消融已由 gateE-TF（center=none）部分补齐；路线 B 未复核。

### 0.6 五分钟上手

1. 先读本文 §0–§3，建立“0.6760 + 候选/part 头寸”的地图。
2. 需要复现/定位：读 `reports/EXPERIMENT_INDEX.md` 与 §12。
3. 需要继续 Part：只读 `reports/PART_MULTIGRANULARITY_GATE.md` §2.9–2.14，先看候选策略/特征/选择器三层结论。
4. 不要重跑基线；不要混 ReLaGS/B0；不要把 oracle 数字当方法。
5. 任何新方法先过 subset 门控，再跑 208；必须报 CI/分族/每场景。
6. 服务器数据盘已装 GroundingDINO-tiny；本地 `claude/part_grounding/` 还有一份 659MB 权重。

---

## 1. 任务、数据、评测协议

### 1.1 LERF-OVS

- 4 场景：figurines / ramen / teatime / waldo_kitchen。
- 22 张标注图、208 条查询；每场景查询数：56 / 71 / 59 / 22。
- 指标：query mIoU；先按图平均，再跨图/跨场景 macro。
- 官方评测器：`REPO/deco/lerf_ovs/eval_lerf_ovs.py`。
- 固定配置：`--renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 --sel gap`。
- 基线读端：
  - ReLaGS：`--center none`（C1 关），0.6401。
  - DeCo B0：`--center image`（C1 开），0.6392625。

### 1.2 ScanNet-10

- 10 场景，19 类闭集 per-root argmax；GT 为每高斯类别。
- 与 LERF 的 query 候选选择结构不同；gateE/module C 不能直接应用。

### 1.3 研究约束（用户明确要求）

1. LERF 逐 query 开放词汇；**不能**用场景词表/其他类别/class bank 推断。
2. 方法 training-free；GT 只用于 evaluation/headroom。
3. 严格 LOSO/预注册门控；禁止 GT 监督 reranker 当方法。
4. 不要被 ReLaGS 实现细节限制，不做“在别人方法上打补丁”。
5. 发现问题要建立在大量实验数据上；有 GT headroom、有现成方法/数学可借。
6. 不混 baseline、不混读出方式、不混重建/官方读权。

### 1.4 有效性分级

- **A（可报绝对数）**：官方读权 `ref/relags_hf/<scene>/sai_nag.pt` + 官方 GT + 官方评测器。
- **B（仅同底座相对可比）**：自己重跑的 `merge_proj` 读权；绝对数不可比。
  - 官方 level-0 7852 超点 vs 发布 `nag-l1.pt` 7936；ramen 重建 0.4935 vs 官方 0.5115。
- **C（仅机理诊断）**：DeCo 观测/候选上的代理实验。

---

## 2. 核心问题与成功标准

### 2.1 问题陈述

现有 3D-OVS 被实现成“用文本在已有候选里选区域”，而不是“先形成 3D 实例、再做开放词汇识别”。
后果：

- 候选池没有的物体永远救不回来（recall 29/208，池内最佳 0.027）。
- 跨视角身份用特征一致性关联不可靠（共享/背景语义更一致）。
- 2D mask 很强（found rank1 77.4%），但 3D 聚合/候选策略把它丢掉。
- 评测只在查询视角渲染，容易隐藏 recall/实例发现问题。

### 2.2 成功标准（建议）

- **方法门控**：training-free；无 GT/场景词表；预注册；逐查询开放词汇。
- **统计**：配对 bootstrap CI 下界 >0；无场景恶化 >−0.01；found 不退化。
- **指标**：macro >0.70（第一步），>0.72（有竞争力）；recall/deep 至少一个族显著提升。
- **可复现**：固定读权/评测；公开脚本/CSV；说明读出方式（单向量 vs late interaction）。
- **论文**：一句话原则 + 统一目标 + 解决 recall/deep 的机制；不是 A+B gating。

---

## 3. 实验总览（按阶段）

| 阶段 | 问题 | 状态 | 结论 |
|---|---|---|---|
| Phase 0 | 基线复现/口径 | 完成 | ReLaGS 0.6401、B0 0.6393；不能混 |
| Phase 1 | 读端/选择/gating | 完成 | gateE 0.6760；读端接近饱和 |
| Phase 2 | 2D/3D 诊断/头寸 | 完成 | 瓶颈在候选/实例生成；一致性不能评分 |
| Phase 3 | 边界/BIoU | 完成 | 边界与候选耦合；好 mask lift 损失 ~0.06；非主因 |
| Phase 4 | 高斯单位/DIGS | 完成 | 扩池负收益；边界+0.04；DIGS 已占边界/本体 |
| Phase 5 | Part 多粒度 | **门控失败/未完成** | part 候选在 leaf 存在；root-first 策略是主因；特征/选择器仍不行 |
| Phase 6 | ScanNet 迁移 | 完成（负） | late interaction 不迁移 |
| Phase 7 | 负结果/协议 | 完成 | 多条路线已排除 |

---

## 4. Phase 0：基线与复现

### 4.1 设计

- 固定官方读权、官方 GT、官方评测器。
- 分别复现 ReLaGS（C1 off）与 DeCo B0（C1 on）。
- 用逐查询 CSV 和 B0_REPRODUCTION_GUIDE 核对。

### 4.2 结果

| 基线 | scene mIoU | macro |
|---|---|---:|
| ReLaGS | 63.29 / 51.15 / 80.97 / 60.62 | 0.6401 |
| DeCo B0 | 58.5869 / 51.8076 / 79.3896 / 65.9209 | 0.6392625 |

### 4.3 想证明/结论

- 两个基线在单场景可差 4–5 分，但 macro 接近；**混用会产生假差异**。
- 官方读权无法由发布产物完全重建；重建只用于同底座相对 A/B。
- 早期把 64.01（C1 off）当 B0 是错误；B0 是 63.9263（C1 on）。

---

## 5. Phase 1：读端/选择/gating（当前最好 0.6760）

### 5.1 掩膜级聚合重排（PASS）

**设计**：固定逐视角 mask token 场；`S = z(NAG) + 1.0·z(id_top3_mask)`；保持 released gap 的 k，仅在 k 内重排。

**结果**：

| 基线 | main | mask rerank | Δ | CI |
|---|---:|---:|---:|---|
| ReLaGS | 0.6401 | **0.6583** | +0.0182 | [+0.0089,+0.0282] |
| B0 | 0.6393 | **0.6619** | +0.0226 | [+0.0136,+0.0319] |

分族：found/rank2/recall 持平，deep +25.4/+30.2；收益集中 ramen +6.1。

**想证明**：物体级 mask 证据 + 多视角保留，比一致性聚合更保身份。
**结论**：PASS；但 λ 敏感（0.5 +0.37、1.0 +1.82、2.0 −2.00）。

### 5.2 模块 C（模块 C = guarded prefix pruning + scale-gated challenger inclusion）

> **协议修正（2026-11-01 晚）**：`pv` 是冻结 LOSO GBM 的**预测 IoU**（`preds.csv` 的 `pred` 列，
> 训练标签为其他 3 场景 GT IoU），因此 module C 是**学习组件**，不满足 training-free。
> 只用 released NAG score 时 0.5·max 剪枝从不触发，module C 退化为
> **尺度门控挑战者并入**（training-free）：官方锚点 0.6401 → 0.6572（20/208 触发）。
> 详见 `reports/CONTRIBUTION_REANCHOR_20261101.md`。

**设计**：在 NAG 前缀上保留 `pv ≥ 0.5·max_pred` 的成员；当 `area[0] < 0.001` 且
`max area > 2×area[0]` 时加入 area champion。

**结果**：B0 main 0.6393 → **0.6603**；deep 0.4446。
**想证明**：截断/尺度保护能修复 deep 的候选选择。
**结论**：PASS（但含学习器）；单独不如 mask rerank 的 deep；两者互补。

### 5.3 逐查询 gating（gateB/gateE）

> **协议修正（2026-11-01 晚）**：下表 gateB/gateE 的 S_C 来自学习型 module C（见 5.2）。
> training-free 版本（C_tf = 尺度并入）在官方锚点 center=none 下重跑：
> main 0.6401 → C_tf 0.6572 → mask 0.6582 → **gateE-TF 0.6716（+3.15，CI [+1.20,+5.39]）**；
> deep 0.2914→0.6515，found −0.0045，无场景恶化 >−0.01。详见 `reports/CONTRIBUTION_REANCHOR_20261101.md`。

**gateB（启发式）**：mask 改变 released 选择 → 用 mask；否则用 C。

**gateE（主推，有原理）**：

```
e_C = max_{i in S_C} id_top3(i)
e_M = max_{i in S_M} id_top3(i)
S_gateE = S_M   if set(S_M) != set(S0) and e_M > e_C
          S_C   otherwise
```

即两套选择在**同一 mask 证据空间**比较最高 `id_top3`，谁强用谁。

**结果**（B0）：

| 选择 | macro | figurines | ramen | teatime | waldo |
|---|---:|---:|---:|---:|---:|
| main | 0.6393 | 0.5859 | 0.5181 | 0.7939 | 0.6592 |
| 模块 C | 0.6603 | 0.5941 | 0.5649 | 0.8223 | 0.6597 |
| mask rerank | 0.6619 | 0.6160 | 0.5791 | 0.7939 | 0.6585 |
| gateB | 0.6757 | 0.6129 | 0.6092 | 0.8223 | 0.6585 |
| **gateE** | **0.6760** | **0.6129** | **0.6092** | **0.8223** | **0.6597** |

CI：gateE−main +0.0367 [+0.0247,+0.0482]；−C +0.0158 [+0.0068,+0.0248]；−mask +0.0141 [+0.0056,+0.0240]。
分族：found 0.8312、rank2 0.6402、deep 0.6607、recall 0.0198。

**想证明**：只在物体证据更强时改选择，保护 found；同尺度比较避免跨融合 margin 不可比。
**结论**：PASS；但 0.6760 含学习型 S_C，不能作 training-free 主张。
training-free 对应结果 **0.6716（官方锚点 +3.15）**，见 5.3 顶部修正框。

### 5.4 失败的组合（不要重复）

| 实验 | 结果 | 原因 |
|---|---|---|
| 秩融合替代 z 融合 | ReLaGS +0.0069（CI 含 0）、B0 +0.0080 | 丢 ramen 的尺度信息 |
| C + mask 排序叠加 | 0.6478，比 C −1.25；waldo −8.2 | 先 mask 排序再套 C 的 prune，破坏 guard |
| margin gating | 0.6603/0.6600（=C） | 两路 margin 跨尺度不可比；mask 改变选择时从不触发 |
| top-k 扩池（全部候选） | 0.432 vs 0.553（3D IoU） | 精度 0.707→0.493；越加越差 |
| 2D proposal 硬 union | 0.5854/0.6501 | found 被拉低 |

### 5.5 读出方式协议（必须写清）

- `id_top3 = mean(top-3 per-view <f_o, t>)`：固定 token 场 + **late interaction**；
  不是“文本进入聚合”，也不是 query-specific field。
- 同一 token 场：
  - 单向量 `id_mean`：B0 +0.0056。
  - late interaction `id_top3`：+0.0226。
  - gateE：+0.0366。
- **不能声称“同读出下超过 baseline”**；baseline 是单向量读出。正确表述：
  “在相同 released 场/候选池上，把单向量换成多视角 late interaction，macro +2.26；加 gating +3.66。”
- leave-query-view-out：gateE 0.6753（+0.0360，CI [+0.0228,+0.0480]），说明不是查询视角 2D 捷径；
  但仍使用了额外逐视角 mask 证据，论文需声明。

---

## 6. Phase 2：2D/3D 诊断与头寸

### 6.1 2D vs 3D：不是表示层，是聚合/选择

- 2D 逐视角 mask 文本 rank1：found **96/124=77.4%**、deep **11/19=57.9%**、rank2 38.9%、recall 27.6%。
- 2D→3D lift 全部 208 查询平均只损失 **0.049 IoU**（found −0.05、deep −0.03、recall −0.09）。
- 3D 最佳候选 IoU **0.708** > 2D 最佳 mask **0.635** → 表示层不是瓶颈。

**想证明**：2D mask 信息足够；3D 聚合/候选选择才是损失来源。
**结论**：支持；后续边界/高斯实验进一步确认。

### 6.2 同物体/不同类余弦

| 场景 | same 中位 | diff 中位 | 间隔 | 跨视角检索 top1 |
|---|---:|---:|---:|---:|
| figurines | 0.734 | 0.495 | 0.239 | 35.5% |
| ramen | 0.629 | 0.563 | **0.066** | 24.9% |
| teatime | 0.759 | 0.496 | 0.263 | 41.5% |
| waldo | 0.603 | 0.498 | 0.105 | 40.0% |

**结论**：ramen 类间间隔仅 0.066；open-vocab 特征的“跨视角一致”不是身份。

### 6.3 一致性为什么不能用于评分

| 信号 | Phase A found | Phase B found | deep |
|---|---:|---:|---:|
| id_top3（文本 top-3） | **65.3%** | **52.4%** | 73.7%/26.3% |
| id_mean | 62.1% | 50.0% | 21.1%/5.3% |
| agree_c（中心化一致性） | 34.7% | 33.9% | 26.3%/21.1% |
| agree（一致性） | **13.7%** | **15.3%** | 0%/21.1% |

- 逐场景：Phase B id_top3 found = fig 0.806 / ramen 0.810 / **teatime 0.204** / waldo 0.615。
- teatime 占 found 44%，单独把总命中拖低；去掉 teatime，Phase B id_top3 = 77.1%。
- teatime found：main 0.858、GT 最佳候选 0.866、id_top3 选中 0.178、选错 43/54。

**想证明**：一致性是 query-agnostic 的关联先验，不是评分权重。
**结论**：一致性只能用于 grouping/association；不能用于聚合评分。

### 6.4 失败族与 GT 头寸

族定义：

- found：池内最佳候选 rank1 且 IoU≥0.3。
- rank2：最佳候选 rank2。
- deep：最佳候选 rank≥3。
- recall：池内最佳 IoU<0.3。

n=124/36/19/29。

| 层级 | macro | 相对 0.676 |
|---|---:|---:|
| gateE | **0.6760** | — |
| 池内 oracle | **0.7094** | +3.3 |
| 2D/3D 族级 oracle | **0.7145** | +3.9 |
| greedy5 | **0.7500** | +7.4 |
| greedy20 | **0.7818** | +10.6 |

分族：

| 族 | n | gateE | 2D GT-best lift | greedy20 |
|---|---:|---:|---:|---:|
| found | 124 | 0.8312 | 0.6654 | 0.8737 |
| rank2 | 36 | 0.6402 | 0.4726 | 0.8218 |
| deep | 19 | 0.6607 | **0.8247** | 0.9108 |
| recall | 29 | **0.0198** | **0.2352** | 0.1959 |

**结论**：

- 当前已接近池内 oracle；真正头寸在候选/实例生成。
- recall 29 条池内最佳 0.027；必须做 proposal。
- deep/recall 的 2D 上界高（0.825/0.235），但选不到。

---

## 7. Phase 3：2D-3D-2D 边界审计（BIoU）

### 7.1 指标与设计

- 对每条查询，在查询视角所有 level-3 SAM mask 中，按**面积 IoU**选 GT-best mask。
- `B(mask)`：mask XOR 1-pixel erosion 得到边界，膨胀 `d = round(ratio·diag)`；
  `BIoU = |B(pred)∩B(gt)| / |B(pred)∪B(gt)|`；ratio = 2%、1%。
- lift：可见 Gaussian（`render_point weight>1e-4`）投影中心落在 mask 内 → 渲染 → BIoU。
- GT 只用于 oracle 选择。

### 7.2 结果

| 口径 | mean | median | ≥0.95 |
|---|---:|---:|---:|
| 2D 按面积 IoU 选 best | IoU 0.635 | 0.833 | 31.7% |
| 同一 mask BIoU 2% | **0.678** | 0.870 | 41.8% |
| 同一 mask BIoU 1% | 0.633 | 0.787 | 21.6% |
| BIoU 2% 重选（oracle） | **0.694** | 0.881 | 41.8% |
| BIoU 1% 重选（oracle） | 0.645 | 0.787 | 21.6% |
| lift 后 BIoU 2% | **0.636** | 0.772 | 12.5% |
| lift 后 BIoU 1% | 0.566 | 0.643 | 2.4% |

分层：

| 2D mask 覆盖 | n | 2D BIoU 2% | lift BIoU 2% |
|---|---:|---:|---:|
| IoU≥0.5 | 137 | **0.900**（中位 0.966） | 0.839（中位 0.897） |
| IoU<0.5 | 71 | **0.252**（中位 0.174） | 0.245 |

分族 2D BIoU 2%：found 0.760、rank2 0.536、deep 0.871、recall 0.382。

### 7.3 想证明/结论

- 0.678 是 208 条均值，不是“每张最好 mask 的典型值”；中位 0.870。
- 按 BIoU 重选只到 0.694，说明不是按 IoU 选 mask 的选择偏差。
- 分层后：有覆盖的 137 条 2D BIoU 0.900；71 条低覆盖只有 0.252；**边界损失与候选缺失高度耦合**。
- 好 mask lift 后仍损失约 0.06；1% band 下 found/deep 只有约 3.5% 到 0.95。
- **像素 BIoU 只是评测探针；方法空间应是高斯边界，但“边界高斯 + top-M 分布 + late-argmax”已是 DIGS 核心**。
- 边界不是当前主因；不要把它作为新方法主张。

---

## 8. Phase 4：高斯级单位、边界与 DIGS 定位

### 8.1 ReLaGS 实际选了什么

代码路径：

- `search_matched_superpoint_in_mhtree(topk=5, level_until=3, filter_small=50)`；
- `_filter_by_similarity_gap` 取前缀 `g`；
- `_get_related_gaussians` 把整个超点全部置 1；**没有高斯级/部分级选择**。

208 条查询（可见口径）：

| 指标 | 值 |
|---|---:|
| 选中候选 mean/median | 2.11 / **2** |
| ≤3 的比例 | **87.5%** |
| 可见选中高斯 mean/median | 5,389 / 2,824 |
| GT 可见高斯 mean/median | 4,785 / 2,514 |
| precision / recall / IoU3D | **0.707 / 0.674 / 0.553** |
| GT 落在已选 root 内的比例 | 0.714 |
| 边界假阳性 / 全部可见假阳性 | 0.642（median 0.872） |

### 8.2 头寸

| 改动 | 3D IoU | 相对当前 |
|---|---:|---:|
| 当前 gap 前缀 | **0.553** | — |
| 全候选池 | 0.432 | **−0.121** |
| GT 最好前缀 | 0.632 | +0.079 |
| GT 最好单候选 | 0.633 | +0.081 |
| 边界假阳性全删 | 0.593 | **+0.040** |
| 已选 root 拆叶（oracle） | 0.585（n=143） | +0.035 |
| 已选 root 内完美挑高斯 | **0.714** | **+0.161** |
| 全场景完美选高斯 | 1.000 | +0.286（相对 root 上界） |

分族：

| 族 | n | precision | recall | IoU3D | 边界增益 |
|---|---:|---:|---:|---:|---:|
| found | 124 | 0.841 | 0.824 | 0.708 | +0.048 |
| rank2 | 36 | 0.713 | 0.828 | 0.607 | +0.063 |
| deep | 19 | 0.337 | 0.324 | 0.197 | +0.003 |
| recall | 29 | 0.198 | 0.066 | 0.056 | +0.004 |

### 8.3 想证明/结论

- “只选 2–3 个超点”是真的，但**扩池是负收益**；单位太混。
- 边界只对 found/rank2 有小增益；deep/recall 是 root/单位错误。
- 真正的头寸是已选 root 内精度/root 覆盖（+0.161/+0.286），不是边界。
- DIGS 已经占了“边界高斯 + top-M 分布 + late-argmax + class-agnostic ontology”；
  DIGS LERF-OVS **71.95**（论文摘要，未本地复现），残余失败 F1 透明/薄结构、F2 SAM 粒度封顶 part。
- **不要做边界高斯/多峰分布/late-argmax 作为新方法**；DIGS 在前。

---

## 9. Phase 5：Part 多粒度（当前主动方向，但门控失败）

### 9.1 2D 多粒度 oracle（PASS）

四层 `seg_map` 全部 mask；GT 选：

| 组 | n | L3 单 mask | 四层单 mask | 贪心集合 | 平均集合 |
|---|---:|---:|---:|---:|---:|
| part | 16 | 0.372 | **0.813** | 0.856 | 2.62 |
| set | 34 | 0.729 | 0.910 | 0.931 | 3.12 |
| part+set | 9 | 0.256 | **0.871** | 0.898 | 3.33 |
| attr | 18 | 0.821 | 0.936 | 0.950 | 3.67 |
| whole | 131 | 0.644 | 0.880 | 0.948 | 3.45 |
| **全部** | **208** | **0.635** | **0.884** | **0.936** | **3.35** |

最佳层级分布：L0 57 / L1 38 / L2 58 / L3 55；**没有固定 level 可用**。
**想证明**：part 的 2D 证据存在，且需要按 query 选粒度/集合。
**结论**：PASS；这是头寸，不是方法。

### 9.2 文本选择器（FAIL）

| 组 | n | top1 | top3 | top5 | oracle 单 mask |
|---|---:|---:|---:|---:|---:|
| part | 16 | 0.364 | 0.343 | 0.261 | 0.813 |
| part+set | 9 | **0.000** | 0.061 | 0.059 | 0.871 |
| set | 34 | 0.629 | 0.689 | 0.575 | 0.910 |
| whole | 131 | 0.508 | 0.585 | 0.493 | 0.880 |
| **全部** | **208** | **0.511** | **0.586** | **0.489** | **0.884** |

top1 层级匹配 37.5%。AlphaCLIP：onion top1 3/7、top3 4/7；bear 1/3；knife 0/3。
NAG/多视角 track 也只部分改善（tesla/porcelain rank1，pirate 68）。
**结论**：raw CLIP/AlphaCLIP/NAG 选不动 part。

### 9.3 3D 候选覆盖率：part 存在，但只在 leaf（关键）

GT part 投到可见 Gaussian，与各层超点最佳 IoU：

| query | root | **leaf** | L1 | L0 |
|---|---:|---:|---:|---:|
| pirate ×4 | 0.301–0.509 | **0.545–0.641** | 0.545–0.641 | 0.09–0.12 |
| bear ×3 | 0.069–0.089 | **0.826–0.943** | 0.49–0.58 | 0.05 |
| hooves ×2 | 0.221 / 0.871 | 0.695 / 0.392 | 0.67/0.37 | 0.10/0.05 |
| brand ×2 | 0.199–0.211 | **0.779–0.788** | 0.47–0.49 | 0.17–0.18 |

**想证明**：part 不是没有候选，而是 root-first 策略把 leaf part 挡在池外。
**结论**：候选策略是主因；特征是第二瓶颈；聚合第三。

### 9.4 GroundingDINO + parent crop（部分 PASS）

- 全图检测 part：只给大 box（bear nose 覆盖 100%、box IoU 0.03；pirate hat box 0）。
- **GT parent** 裁图 + part 检测 + **box 中心最小 mask**（下表是 top-scoring part box，不是 oracle）：

| query | frame | 2D part IoU | lift 渲染 IoU |
|---|---|---:|---:|
| pirate hat | 00041 | 0.929 | **0.903** |
| pirate hat | 00105 | 0.054 | 0.041 |
| pirate hat | 00152 | 0.947 | **0.940** |
| pirate hat | 00195 | 0.941 | **0.906** |
| bear nose | 00002 | 0.982 | **0.953** |
| bear nose | 00025 | 0.989 | **0.965** |
| bear nose | 00107 | 0.000 | 0.000 |
| hooves | 00025 | NaN | NaN（oracle lift 0.878） |

注：`part`/`part+set` 是按 query 关键词分组；**true-part 11 条**是 pirate hat 4、bear nose 3、
hooves 2、dall-e brand 2。`onion segments` 7 条是独立小块，`tesla door handle`/`porcelain hand`/`hand`
更像整体物体；+2.1 只针对 8 条 true-part，不是全部 16 条 part。

- **检测到的 parent**（top-1 box）：pirate 3/4、bear 1/3、hooves 0/2、brand 0/2。
- 多假设 support：pirate 3/4、bear 1/3；大 mask 会刷 support。
- **结论**：parent 正确时 part mask 选择已解决；**parent 定位是瓶颈**。

### 9.5 无 GT parent 推断（FAIL）

- SAM parent 候选 + part 检测 support：pirate 0/4、bear 2/3、hooves 0/2、brand 0/2。
- CLIP 给正确 parent 的排名 3–60；只有 4/11 进 top-4。
- **结论**：当前 CLIP/GroundingDINO + SAM 候选无法可靠推断 parent；
  +2.1 是 GT-parent oracle，不可部署。

### 9.6 语义 enrichment（部分改善，FAIL top1）

表示：local `_f`、context（bbox/1.5×）、parent、residual（local−parent），late interaction max。

| 表示 | mean rank | median | top1 | top3 | top5 | top10 | top1 IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 53.4 | 35 | 2/11 | 2/11 | 3/11 | 3/11 | 0.116 |
| +context | 34.3 | 17 | 1/11 | 4/11 | 5/11 | 5/11 | 0.116 |
| +parent/residual | 39.3 | 16 | 2/11 | 2/11 | 4/11 | 5/11 | 0.100 |
| all late-interaction | **27.2** | **5** | 1/11 | 4/11 | **6/11** | **6/11** | 0.100 |

**结论**：enrichment 提高 recall@5（3/11→6/11），但 top1 仍 1/11；不能单独完成选择。

### 9.7 Part-whole 选择器（FAIL）

enrichment top-50 + 多视角 track（8 近邻）+ parent + 几何：

| 信号 | mean rank | top1 IoU |
|---|---:|---:|
| text | 12.1 | 0.100 |
| track_good | 18.2 | **0.175** |
| track_cos_mean | 31.0 | 0.083 |
| track_cov_mean | 15.2 | 0.032 |
| parent | 16.7 | 0.100 |
| resid | 20.6 | 0.048 |
| area_ratio | 30.5 | 0.084 |
| level | 18.3 | 0.005 |

没有组合规则把 top1 IoU 抬到 0.45；门控 top1≥0.5/top3≥0.7 未过。
**结论**：多视角 track 不具判别力；当前 class-agnostic 特征 + 几何无法完成 part 选择。

### 9.8 Part 方向总结

- **2D/3D leaf part 候选存在**：2D oracle 0.813/0.856；3D leaf 0.55–0.94。
- **主因是候选策略**：root-first + leaf>root 才提升，把 part 挡在池外。
- **第二是初始语义**：单 mask CLIP 压成单向量，text rank 3–60。
- **第三是聚合**：leaf→root 混合；不是主要问题。
- **可部署方法尚未成立**：parent 推断、part 选择、多实例集合都未过门控。
- **当前 208 指标仍 0.6760**；+2.1 是 GT-parent oracle；+1.6 的检测 parent 仍用 GT parent prompt。

---

## 10. Phase 6：ScanNet 迁移（负结果）

### 10.1 设计

把 LERF 的 late interaction/多视角 token 读出移植到 ScanNet-10 的 19 类 per-root argmax；
baseline 是 root 3D 特征点积。

### 10.2 结果

| 信号 | scene macro | pooled |
|---|---:|---:|
| baseline（root 3D 特征） | **40.36%** | **43.88%** |
| top1 late interaction | 19.54% | 28.22% |
| top3 late interaction | 22.85% | 31.94% |
| top5 late interaction | 25.58% | 33.04% |
| meanfeat | 40.15% | 42.83% |
| top3 + 背景对比 | 1.58% | 2.66% |

### 10.3 结论

- late interaction 不迁移到 ScanNet 闭集分类；meanfeat 接近 baseline。
- 任务结构不同：LERF 是 query 候选选择；ScanNet 是 per-root 19 类分类。
- 方法作用域是 **open-vocabulary query selection**，不是闭集分类。
- 不要在 ScanNet 上硬套 gateE/module C。

---

## 11. 负结果清单（不要重复）

| 路线 | 结果 | 原因 |
|---|---|---|
| ROFA 一致性剔除 | 官方读权中性/空操作；重建场景波动 ±14 | 一致性优化共享语义 |
| C1/center/白化单独用 | rank-preserving 或破坏对齐 | 常数平移/共享成分 |
| 一致性聚合评分 | found 13.7–15.3% | query-agnostic，选到共享/背景 |
| 跨视角一致性选 2D mask | text 0.4598→xview 0.4003/0.3038 | 共享 mask 更一致 |
| 秩融合 | +0.0069/+0.0080 | 丢 ramen 尺度 |
| C+mask 排序叠加 | 0.6478，−1.25 vs C | 破坏 C 的 guard |
| margin gating | 0.6603/0.6600 | 跨尺度 margin 不可比 |
| 扩大 top-k/全池 | 3D IoU 0.553→0.432 | 精度崩 |
| 2D proposal 硬 union | 0.585/0.650 | found 下降 |
| SREM Phase A | found 87.9→75.8、deep 73.7→68.4 | 背景项惩罚正确证据；冗余项换掉正确 token |
| qgate 模板一致性 | 0.6603（=C）；deep 0.4446 | reliability 0.90–0.93，无区分度 |
| 文本侧场景 bank 校准 | 违反开放词汇 | 不能用场景类别 |
| ScanNet late interaction | 40.36→22.85 | 闭集分类不适配 |
| 边界作为新方法 | 边界全删 +0.04；DIGS 已做 | 边界不是主因，且被 DIGS 占 |
| Part CLIP/AlphaCLIP/NAG 直接选 | top1 0.511；part+set 0.000 | part 语义弱 |
| 无 GT parent 推断 | pirate 0/4、hooves 0/2 | CLIP/GDINO+SAM 选不出 parent |
| Part-whole selector | top1 IoU 0.175 | 多视角 track/几何不判别 part |

---

## 12. 当前最好的精确复现

### 12.1 配置

- 底座：DeCo B0（`--center image`，C1 开）。
- 评测：`--renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 --sel gap`。
- 读出：`id_top3` 多 token late interaction；`S = z(NAG)+z(id_top3)`，λ=1；released k 不变。
- gateE：见 5.3。
- module C：guarded prefix pruning + scale-gated challenger inclusion。

### 12.2 关键脚本/结果

| 内容 | 路径 |
|---|---|
| gateE 脚本 | `analysis/lerf_c_mask_gate.py` |
| gateE 结果 | `analysis/results/20261031/gatee/<scene>_cmaskgate_image.csv` |
| mask rerank | `analysis/lerf_maskagg_rerank.py`、`results/20261031/maskagg/` |
| 基线 B0/ReLaGS | `results/20261031/baselines/` |
| 诊断 CSV | `results/20261031/diagnostics/` |
| 边界 | `analysis/lerf_boundary_probe.py`、`results/20261101/boundary*/` |
| 高斯头寸 | `analysis/lerf_gaussian_headroom.py`、`results/20261101/gauss_headroom3/` |
| 多粒度 oracle | `analysis/lerf_multigranularity_gate.py`、`results/20261101/mg_gate2/` |
| 文本选择 | `analysis/lerf_mg_select.py`、`results/20261101/mg_select/` |
| GroundingDINO 门控 | `analysis/lerf_part_grounding_gate.py`、`analysis/lerf_part_grounding_mh.py` |
| 无 GT parent | `analysis/lerf_part_grounding_nogt.py` |
| enrichment | `analysis/lerf_part_enrichment_gate.py`、`results/20261101/enrichment/` |
| part-whole selector | `analysis/lerf_part_selector_gate.py`、`results/20261101/selector/` |
| 3D 候选覆盖 | `code/part_3d_recall2.py`（服务器） |

### 12.3 GitHub / 备份

- GitHub：`https://github.com/Youyuan287/DeCo-3DGS/tree/main/experiments/lerf_6760`
- 提交：`1aa0bf1`（父 `839d23e`）。
- 包内：README、scripts、inputs、results、manifest、docs。
- 本地备份：`CVPR-2026/claude-repo-20261031.bundle`、`claude_backup_20261031.zip`。

### 12.4 服务器路径

- 读权：`/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/<scene>/sai_nag.pt`
- 数据：`/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/<scene>`
- GT：`/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label`
- 结果：`/root/rivermind-data/repro/deco-b123/`
- Part grounding（数据盘）：`/root/rivermind-data/part_grounding_20261101/`
  - `pylibs/`：transformers 4.40.2 + tokenizers 0.19 + hub 0.23 + loralib
  - `model/grounding-dino-tiny/`：658MB 权重
  - `code/`、`out/`、`hf_cache/`
- 本地 `claude/part_grounding/` 还留了一份 659MB 权重；服务器数据盘已有，可手动删。

### 12.5 CSV schema（关键）

| 路径 | 内容 |
|---|---|
| `gatee/<scene>_cmaskgate_image.csv` | main/c/mask/gate/gateb/gatee 的 IoU、e_c/e_m、gap_len |
| `diagnostics/<scene>_segidentity.csv` | 每候选 id_mean/id_top3/agree/iou/rank/level |
| `diagnostics/<scene>_liftprobe.csv` | cov2d/lifted_iou |
| `boundary_oracle/*.csv` | 2D IoU/BIoU/oracle/lift |
| `gauss_headroom3/*.csv` | 选中超点/高斯、precision/recall、leaf split、boundary prune |
| `mg_gate2/*.csv` | 每查询 best0..3/all/union、层级 |
| `mg_select/*.csv` | top1/top3/top5 CLIP 选择 |
| `enrichment/enrichment_gate.csv` | local/ctx/partwhole/all 的 rank/top IoU |
| `selector/part_selector_candidates.csv` | top-50 候选的 text/track/parent/geometry/GT IoU |

---

## 13. 当前计划与下一步

### 13.1 立即结论

- **不要继续在固定池上调 gating/λ/特征聚合**；已饱和。
- **不要做边界高斯/多峰分布/late-argmax**；DIGS 71.95 在前。
- **不要直接把 CLIP 多粒度 top-k 写成方法**；top1 0.511 < L3 oracle 0.635。
- **不要把 GT-parent +2.1 当方法结果**；有泄露。

### 13.2 三条可选路线（按风险/收益）

#### A. Part：leaf-first 候选池 + 更强 part 语义（高风险，高收益）

1. **候选策略**：leaf/part 作为独立候选，不再只 root top-5；目标 part recall@pool。
2. **语义源**：
   - HQ-SAM/SAM2 更细 mask；
   - GroundingDINO/OWL-ViT parent/part 定位（GT parent 下已 0.90–0.99；无 GT parent 失败）；
   - AlphaCLIP/GRIT 区域特征；
   - LLM/VLM 属性描述（DIGS 路线，注意协议/成本）；
   - 几何 part 分割（法向/曲率/附肢）。
3. **选择器**：part-whole 多假设 + 多视角一致性 + 几何先验；当前 track/几何失败，需更强信号。
4. **门控**：true-part top1≥0.5、top3≥0.7；2D→3D lift；208 mIoU >0.70。
5. **风险**：DIGS F2 已说明 part 是 SAM 粒度 ceiling；可能必须换 2D 源。

#### B. Set/plural（低风险，收益中）

- set 组 34 条：CLIP top1 0.629、oracle 0.910；现有方法 0.767；
- 做集合选择（submodular/coverage/PCST）而不是 argmax；预计 +1.5–2 分；
- novelty 较弱，但可作为方法的一部分。

#### C. 透明/薄结构（DIGS F1，窄）

- 透明盒/杯子：alpha compositing 让前景投票泄漏到背景；
- 问题清晰，但覆盖面小；可作为 focused contribution。

#### D. 分析/benchmark 论文（保底）

- 已有证据：一致性不是身份、2D/3D 脱节、候选策略 root-first 挡 part、边界与候选耦合、part 候选存在但选不动；
- 可写成“诊断 + 负结果 + protocol/指标”论文；但需要更强的 benchmark/novel-view/统计。

### 13.3 建议顺序

1. **先决定是否继续 Part**：如果继续，先做 leaf-first 候选池 + 更强 part 2D 源的门控；
   如果 2D 源（HQ-SAM/SAM2/GDINO/AlphaCLIP）仍不能把 true-part top1 拉到 0.5，放弃 Part 主方法。
2. **同时补 C1-off × C × gate 两格**（低成本，完善 0.6760 的受控叙事）。
3. **若 Part 失败**：转 set/plural 或透明；或把现有诊断写成分析论文。
4. **任何方法**：必须跑 208 条、配对 bootstrap CI、分族、无场景恶化；不能只报 subset/oracle。

### 13.4 必须补的实验（如果继续方法）

- leaf-first 候选池的 **part recall@k**（无文本选择）。
- 更强 2D 源的 **part grounding 门控**：top1≥0.5、top3≥0.7。
- part mask → 3D lift 的端到端 mIoU（subset → 208）。
- 多实例（hooves）集合选择。
- 消融：单粒度 vs 多粒度；leaf-first vs root-first；单向量 vs late interaction；enrichment 各 token；track/几何。
- 统计：配对 bootstrap（B=2000，seed 固定）；分族；每场景。
- 协议：query-only open-vocab；不用场景 class bank；novel-view 检查。

---

## 14. 数据泄露 / 协议审计清单

### 14.1 已确认的泄露（只能当 oracle/诊断）

| 结果 | 泄露点 |
|---|---|
| 2D 多粒度 oracle 0.884/0.936 | GT 选 mask/level/集合 |
| BIoU 0.678/0.694 | GT 选 best mask / BIoU oracle |
| 高斯头寸 +0.161/+0.286 | GT 选高斯/root |
| Part GT-parent +2.1 | GT parent 类别/crop |
| Part 检测 parent +1.6 | parent prompt 仍来自 GT 类别 |
| Part-whole 选择器分析 | GT 选 top-50 候选用于评估 |
| 任何“池内 oracle/greedy” | GT 选择 |

### 14.2 无泄露但未成功

- CLIP/AlphaCLIP/NAG 文本选择：无 GT，但效果差。
- 无 GT parent 推断：无 GT，但失败。
- enrichment/selector：无 GT 选择，但 top1 失败。
- ScanNet late interaction：无 GT 选择，但负迁移。

### 14.3 协议检查项

- baseline 是 ReLaGS 还是 B0？不能混。
- 读出是单向量还是 late interaction？必须声明。
- 读权是官方 `sai_nag.pt` 还是重建？绝对数不可比。
- 是否用了查询视角 2D mask 证据？leave-query-view-out 检查。
- 是否用了场景 class bank/词表？LERF-OVS 禁止。
- 是否用 GT 选候选/阈值/λ？只能 oracle/诊断。
- 是否有 CI、分族、每场景、无场景恶化？
- 是否只在 subset 上报？必须 208 条。

---

## 15. 文件索引（按阅读顺序）

### 15.1 先读

1. `reports/PROJECT_HANDOFF_20261101.md`（本文）
2. `reports/EXPERIMENT_RECORD.md`（总实验表）
3. `reports/GPT_HANDOFF_DOSSIER.md`（上一版交接，含协议/路径）
4. `reports/REAL_PROBLEM_AND_WHAT_WE_SOLVED.md`（阶段反思：只做到选择层）
5. `reports/RELAGS_ANALYSIS_AND_HEADROOM.md`（ReLaGS 不足/头寸）
6. `reports/3DOVS_OPEN_PROBLEMS.md`（领域问题）
7. `reports/PART_MULTIGRANULARITY_GATE.md`（Part 最新，401 行）
8. `reports/GAUSSIAN_HEADROOM_AND_DIGS_POSITIONING.md`（高斯/DIGS）
9. `reports/BOUNDARY_LOSS_ANALYSIS.md`（BIoU）
10. `reports/SCANNET_LATE_INTERACTION_REPORT.md`（ScanNet 负迁移）

### 15.2 方法/叙事

- `reports/CVPR_DIRECTION_QRES.md`（SREM 目标）
- `reports/PAPER_STORY_AND_NEXT.md`（2×2×2 消融缺口/claim）
- `reports/NEXT_NOVEL_DIRECTION_QCMG.md`（objectness-first/QCMG）
- `reports/RELAGS_FRAMEWORK_OPPORTUNITIES.md`（逐阶段机会）
- `reports/WHY_CONSISTENCY_HURTS.md`（一致性负结果）
- `reports/PROTOCOL_QUERY_CONDITIONED.md`（late interaction 协议）

### 15.3 负结果/专项

- `reports/SREM_PHASE_A_REPORT.md`
- `reports/QUERY_ONLY_RELIABILITY_REPORT.md`
- `reports/LERF_MASKAGG_REPORT.md`
- `reports/LERF_MODULE_PRUNE_INCLUDE_REPORT.md`
- `reports/LERF_ROFA_REPORT.md`
- `reports/LERF_STEP0_ATTRIBUTION_REPORT.md`
- `reports/SCANNET_*`

### 15.4 结果目录

- `analysis/results/20261031/`：基线、gateE、maskagg、diagnostics、rofa、srem。
- `analysis/results/20261101/`：boundary、gauss_headroom、mg_gate、mg_select、enrichment、selector、scannet。

---

## 16. 开放问题（下一任需要判断）

1. Part 是否值得继续？
   - 2D/3D leaf 候选存在，但无 GT parent 推断 + part 选择都失败；
   - 若 HQ-SAM/SAM2/GDINO/LLM/VLM 仍不能把 true-part top1 拉到 0.5，放弃 Part 主方法。
2. 是否先补 C1-off × C × gate 两格，完善 0.6760 的受控叙事？
3. Set/plural 是否更容易？set top1 0.629、oracle 0.910，34 条；收益约 +1.5–2 分。
4. 透明/薄结构是否更适合作为 focused contribution？
5. 是否需要 novel-view/instance-level 评测，避免查询视角 2D 捷径？
6. 是否把现有诊断写成分析论文（一致性不是身份、2D/3D 脱节、候选策略挡 part）？
7. DIGS 71.95 在前：任何新方法必须与 DIGS 同协议对比，并说明不是它的增量。

---

## 17. 一句话总结

**training-free 最好结果 0.6716（官方锚点 0.6401 +3.15）来自”尺度门控并入 + 多视角 mask late interaction +
同尺度证据 gating”；原 0.6760 含 LOSO 学习器，只能作学习层消融。**
它把选择层做到了接近池内 oracle，但没有解决”实例/part 从哪里来”。
最新 Part 诊断表明：part 候选在 2D/3D leaf 里存在，主因是 root-first 候选策略把 leaf part 挡在池外；
特征是第二瓶颈，聚合第三。无 GT parent 推断、part 语义选择、多实例集合都未过门控，
所以最终 208 指标仍未上涨；+2.1 是 GT-parent oracle，不能当方法结果。**