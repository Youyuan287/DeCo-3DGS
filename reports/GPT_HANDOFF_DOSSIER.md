# GPT 交接档案：LERF-OVS 3D-OVS 实验脉络、最新设计与原始数据

日期：2026-11-01
用途：把当前项目的实验状态完整交给另一个分析者（GPT），用于继续设计实验。
原则：所有数字都注明口径；**不要把不同 baseline/读出方式混用**；不确定的地方看原始 CSV。

---

## 0. 任务与数据

- 任务：LERF-OVS 开放词汇 3D 分割，4 场景（figurines / ramen / teatime / waldo_kitchen），
  22 张标注图、208 条查询；指标 query mIoU（先按图平均，再跨图/跨场景 macro）。
- 底座：ReLaGS/DeCo 的 released 3D 场（`sai_nag.pt` + 2D SAM mask 特征 + GT 多边形）。
- 约束：训练-free；GT 只用于 IoU/分族，不能进入选择；不得用 GT 监督重排器当方法。

## 1. 两个官方基线（绝对不能混用）

评测器：`eval_lerf_ovs.py --renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 --sel gap`。

| 基线 | 读端 | 四场景 scene mIoU | macro |
|---|---|---|---:|
| **ReLaGS** | `--center none`（C1 关） | 63.29 / 51.15 / 80.97 / 60.62 | **0.6401** |
| **DeCo B0** | `--center image`（C1 开） | 58.5869 / 51.8076 / 79.3896 / 65.9209 | **0.6392625** |

- 单场景可差 4–5 分（fig 58.59 vs 63.29；waldo 65.92 vs 60.62），macro 却接近。
- 官方读权无法由发布产物重建：官方 level-0 **7852** 超点 vs 发布 `nag-l1.pt` **7936**；
  ramen 重建 0.4935 vs 官方 0.5115。→ **不要用重建读权报绝对数**。

## 2. 当前最好结果：DeCo B0 + gateE = 0.6760

精确配置：B0（center image, canon, topk5, level3, filter50, gap）；λ=1 的 z 融合；released k 不变。

| 场景 | main | module C | mask rerank | **gateE** |
|---|---:|---:|---:|---:|
| figurines | 0.5859 | 0.5941 | 0.6160 | **0.6129** |
| ramen | 0.5181 | 0.5649 | 0.5791 | **0.6092** |
| teatime | 0.7939 | 0.8223 | 0.7939 | **0.8223** |
| waldo | 0.6592 | 0.6597 | 0.6585 | **0.6597** |
| **macro** | **0.6393** | **0.6603** | **0.6619** | **0.6760** |

配对按图分层 bootstrap（B=2000，seed=20261030）：
- gateE − main **+0.0367 [+0.0247,+0.0482]**
- gateE − module C **+0.0158 [+0.0068,+0.0248]**
- gateE − mask rerank **+0.0141 [+0.0056,+0.0240]**

分族（found 124 / rank2 36 / deep 19 / recall 29）：

| 族 | main | C | mask rerank | gateE |
|---|---:|---:|---:|---:|
| found | 0.8160 | 0.8312 | 0.8160 | **0.8312** |
| rank2 | 0.6635 | 0.6402 | 0.6631 | 0.6402 |
| deep | 0.2263 | 0.4446 | 0.5281 | **0.6607** |
| recall | 0.0266 | 0.0198 | 0.0261 | 0.0198 |

## 3. gateE 算法（当前最好，可直接复现）

记号：`p5` = released top-5 pool（NAG 分排序，含 leaf 提升）；`g = gap_len(NAG)`；
`S0 = p5[:g]`；`S_C` = module C 的 prune+include；`S_M = top-g by S = z(NAG)+z(id_top3)`；
`id_top3 = mean(top-3 per-view <f_o,t>)`（**late interaction**，固定 token 场，文本只在读出端）。

```
e_C = max_{i in S_C} id_top3(i)
e_M = max_{i in S_M} id_top3(i)
S_gateE = S_M   if set(S_M) != set(S0) and e_M > e_C
          S_C   otherwise
渲染 S_gateE 的并集，IoU vs GT
```

module C = guarded prefix pruning（保留 pv ≥ 0.5·max_pred 的前缀成员）+
R1 scale-gated challenger inclusion（area[0]<0.001 且 max area>2×area[0] 时加入 area champion）。

## 4. 关键实验脉络（按时间，含结论）

| # | 实验 | 结果 | 判定 |
|---|---|---|---|
| 1 | ROFA 忠实复现（DeCo 候选，raw/center） | 205/208 逐条相同；剔除 4.8% 视角；raw −0.034 vs main | 作废（底座/协议不对） |
| 2 | 官方读权 bank B2 ROFA | off/raw/center 逐条相同；bank 融合 −4 分 | ROFA 在评分路径无效 |
| 3 | 重建 merge_proj ROFA A/B | raw−off +2.7 但场景 ±14；绝对数不可比 | 仅相对、不可用 |
| 4 | 候选间排斥（readout） | figurines 0.5468 vs 0.6329 | FAIL |
| 5 | 掩膜级聚合 rerank | B0 0.6393→0.6619（+0.0226）；ReLaGS 0.6401→0.6583 | PASS |
| 6 | 秩融合替代 z 融合 | ReLaGS +0.0069（CI 含 0）、B0 +0.0080 | FAIL（丢 ramen） |
| 7 | C+mask 排序叠加 | 0.6478，比 C 低 1.25；waldo −8.2 | FAIL（交互破坏 C） |
| 8 | 逐查询 gating（gateB） | 0.6757 | PASS（启发式） |
| 9 | 同尺度证据 gating（gateE） | **0.6760** | PASS（有原理，当前最好） |
| 10 | 几何约束 2D mask 选择 | text 0.4598→0.5259；deep 0.610；仍低于 3D | Stage 1 FAIL |
| 11 | leave-query-view-out | gateE 0.6753（+0.0360） | 不是查询视角捷径 |
| 12 | **SREM Phase A** | found 87.9→75.8、deep 73.7→68.4；λ0μ1 recall 31.0 但 found 76.6 | **Stage 1 FAIL** |

## 5. 关键诊断（决定方法方向）

### 5.1 2D vs 3D：不是“3D 特征差”，而是“聚合/选择”

- 2D 逐视角 mask 文本 rank1：found **96/124=77.4%**、deep 11/19=57.9%、rank2 14/36=38.9%、recall 8/29=27.6%。
- 2D→3D lifting：全部 208 查询平均只损失 **0.049 IoU**（found −0.05、deep −0.03、recall −0.09）。
- 3D 最佳候选 IoU（0.708）甚至高于 2D 最佳 mask（0.635）→ **表示层不是瓶颈**。

### 5.2 同物体/不同类余弦

| 场景 | same 中位 | diff 中位 | 间隔 | 跨视角检索 top1 |
|---|---:|---:|---:|---:|
| figurines | 0.734 | 0.495 | 0.239 | 35.5% |
| ramen | 0.629 | 0.563 | **0.066** | 24.9% |
| teatime | 0.759 | 0.496 | 0.263 | 41.5% |
| waldo | 0.603 | 0.498 | 0.105 | 40.0% |

### 5.3 一致性为什么不能用于评分

Phase A（观测 token）/ Phase B（mask token），found 命中：

| 信号 | Phase A | Phase B |
|---|---:|---:|
| id_top3（文本 top-3） | **65.3%** | **52.4%** |
| id_mean | 62.1% | 50.0% |
| agree_c（中心化一致性） | 34.7% | 33.9% |
| agree（一致性） | **13.7%** | **15.3%** |

- 逐场景：Phase B id_top3 found = fig 0.806 / ramen 0.810 / **teatime 0.204** / waldo 0.615；
  teatime 占 found 的 44%，单独把总命中拖低；去掉 teatime，Phase B id_top3 = 54/70=77.1%，接近 2D。
- **结论**：一致性是 query-agnostic 的，优化共享/背景语义；只能用于**关联/分组**，不能用于评分聚合。

### 5.4 剩余头寸

- released pool oracle **0.7094**；oracle greedy5 **0.7500**；greedy20 **0.7818**。
- GT 最佳 2D mask 提升上界：deep **0.8247**、recall **0.2352**（3D 对应 0.6607/0.0198）。
- 2D/3D 族级 oracle 混合 ≈ **0.7145**。
- recall 29 条池内最佳仅 0.027 → **必须做候选生成**，rerank 救不了。

### 5.5 teatime 高 mIoU vs 低 id_top3 命中（不矛盾）

teatime found（n=54）：released main 0.858、GT 最佳候选 0.866、id_top3 选中 **0.178**、选错 43/54。
高 mIoU 是 **NAG 主路径**；id_top3 从未在 teatime 部署（mask 改 0 条，gateE 退回 C）。

## 6. 最新实验设计：SREM（Scene-Relative Evidence Maximization）

### 6.1 动机

把 67.60 从 A+B 工程收敛成一个目标：**身份 = 候选视角证据集合相对 query + 场景背景的关系**。

### 6.2 Phase A 设计（已跑，FAIL）

- 固定 token 场 `X={f_o}`；背景 `B` = **不同 root** 的其他候选 token；
- 全局 z（整个查询所有候选 token）：
  `base_o = z(<f_o,t>) − λ z(max_{b∈B}<f_o,b>)`；
- 贪心选 token：边际 `base_o − μ max_{j∈S}<f_o,f_j>`，k≤8，增益≤0 停；
- 分数 = 选中 mean base − μ × 选中两两平均余弦；
- λ=1, μ=1 主设置；消融 (1,0)、(0,1)。

结果（候选命中率）：

| 信号 | found | rank2 | deep | recall |
|---|---:|---:|---:|---:|
| id_top3 | **87.9%** | 27.8% | **73.7%** | 24.1% |
| SREM λ1μ1 | 75.8% | 25.0% | 68.4% | 17.2% |
| λ1μ0 | 80.6% | 25.0% | 68.4% | 17.2% |
| λ0μ1 | 76.6% | 27.8% | 73.7% | **31.0%** |

失败原因：背景项惩罚正确证据；冗余项换掉正确 top token；自适应 k≈6 稀释；Phase A 是区域混合特征。

### 6.3 下一版：Phase B SREM（尚未跑，建议 GPT 设计）

在 `lerf_segment_identity_dump.py` 的 **object-level mask token** 上跑同一 SREM：
- 每个候选的逐视角 mask 特征 `X`（Phase B 已有）；
- 背景 B = 不同 root 的其他候选 mask token；
- 全局 z；λ/μ 预注册（不得在看到结果后调）；
- Stage 1 门控：相对 Phase B id_top3（found 52.4%、deep 26.3%、rank2 52.8%、recall 34.5%）
  found 或 deep +3 且 found 不降 >2。

## 7. 下一步候选（给 GPT 选择）

1. **Phase B SREM**：最直接延续；若失败，说明背景对比不是核心。
2. **Class-agnostic mask graph proposal**：逐视角 SAM mask 图 + 几何兼容（lift 后支撑重叠/可见性），
   连通分量/谱聚类 → 3D tracks；解决 recall 29 条（池内 0.027，greedy20 0.196）。
3. **late interaction 消融**：max / top-1 / top-3 / top-5 / logsumexp；token 过滤（coverage/visibility）；
   多原型（候选内 k-means）；这些都在固定 token 场上，协议安全。
4. **公平协议**：单向量 vs 多 token 读出必须声明；给 baseline 同 token/读出；
   novel-view 评测，避免查询视角 2D 捷径。
5. **理论**：late interaction 的排序保持/扰动稳定性；SREM 子模性（线性 + 负成对 = 子模；
   非单调时贪心只有 1/e/局部保证）；SPRT 停止准则。

## 8. 不要做的事（已证伪）

- 用一致性/中心化/白化直接改评分空间（found 13.7–15.3%）。
- 用观测特征聚合分替换 NAG 排序（raw −0.034）。
- 把 2D proposal 硬 union 进 3D（found 0.585/0.650）。
- 用 text-in-aggregation 的说法描述 id_top3（它是 fixed field + late interaction）。
- 直接把重建读权与官方 64.01/63.93 比较（不可比）。
- 只在单场景下结论（teatime 会翻转）。

## 9. 原始数据与代码位置

### 9.1 GitHub（已推送）

- `https://github.com/Youyuan287/DeCo-3DGS/tree/main/experiments/lerf_6760`
- 远端 main 提交 `1aa0bf1`（父 `839d23e`）。
- 包内：README、scripts、inputs（segidentity/preds/evidence/identity）、results（gateE/B0）、manifest、
  docs（PREREG、WHY_CONSISTENCY_HURTS、PROTOCOL_QUERY_CONDITIONED）。

### 9.2 本地仓库

`C:\Users\TYY\Documents\MyDocument\CVPR-2026\claude`
- `reports/`：全部中文报告（EXPERIMENT_RECORD、WHY_CONSISTENCY_HURTS、PROTOCOL、HEADROOM、3DOVS_OPEN_PROBLEMS、SREM_PHASE_A_REPORT、CVPR_DIRECTION_QRES）。
- `analysis/`：脚本 + PREREG。
- `analysis/results/20261031/`：逐查询 CSV（见下）。

### 9.3 CSV schema

| 路径 | 内容 |
|---|---|
| `gatee/<scene>_cmaskgate_image.csv` | main_iou, c_iou, mask_iou, gatee_iou, gateb/gatec/gated, m_nag/m_mask, e_c/e_m |
| `baselines/b0/<scene>.csv` | B0 逐查询 iou（per_query） |
| `diagnostics/<scene>_segidentity.csv` | 每候选 id_mean/id_top3/agree/agree_c/iou/rank/level/root_id/cand_id |
| `diagnostics/<scene>_identity.csv` | Phase A 同结构 |
| `diagnostics/<scene>_crossview.csv` / `_xform.csv` | same/diff 余弦、检索、变换 |
| `diagnostics/<scene>_maskrank.csv` | 2D mask 文本排名 |
| `diagnostics/<scene>_liftprobe.csv` | cov2d / lifted_iou |
| `maskagg*/`, `maskcstack/`, `maskgeom*/`, `maskxview/` | 各读出/几何/跨视角实验 |
| `srem/<scene>_srem.csv` | srem / srem_l1m0 / srem_l0m1 / iou / id_top3 |
| `segid_lo/` | leave-query-view-out token 证据 |
| `gatee_lo/` | leave-out 端到端结果 |

### 9.4 服务器（fj01）

- 读权：`/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/<scene>/sai_nag.pt`
- 数据：`/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/<scene>`
- GT：`/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label`
- 结果：`/root/rivermind-data/repro/deco-b123/` 下 `gatee_20261031/`、`srem2_20261031/` 等。

## 10. 历史修正（防止重复踩坑）

1. 早期把 64.01（C1 off）当 B0；B0 是 63.9263（C1 on）。
2. 早期汇总 bug 得到 id_top3 86.3%/agree 31.5%；正确为 Phase A 65.3%/13.7%、
   Phase B 52.4%/15.3%（逐条重算）。
3. id_top3 不是文本进聚合，而是固定 token 场 + late interaction（`mean(top-3 sims)`）。
4. SREM 第一版候选内 z-score 崩盘（found 0.19）；已改全局 z；`toks` 错位已修。
5. merge_proj 重建与官方读权不可比（7852 vs 7936）。

## 11. 一句话给 GPT

**当前最好 0.6760 是“固定逐视角 token 场 + top-3 late interaction + 同尺度证据 gating”；
标准单向量读出只有 +0.56。下一步请在 Phase B object-level token 上验证 SREM；
若仍失败，转 class-agnostic mask graph proposal 解决 recall，而不是继续调融合权重。**