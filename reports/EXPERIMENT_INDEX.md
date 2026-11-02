# 实验索引：脚本、运行指令、结果位置

日期：2026-11-01
目的：以后要复跑或找结果时，从这张表直接定位。
所有 GPU 实验都在服务器 `fj01` 上跑；本地仓库只存脚本/报告/CSV 副本。

> **先读总交接**：`reports/PROJECT_HANDOFF_20261101.md`（目标、全部实验设计/结果/结论、当前计划、泄露审计、文件索引）。

---

## 0. 统一环境（所有命令的前缀）

```bash
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
REL=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
SAS=/root/rivermind-data/repro/deco-b123/m1_sas_20261029
export RELAGS_REPO=$REL HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$REL:$REL/ext:$REL/ext/spt/dependencies/grid_graph/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
```

结果目录约定：

| 层 | 路径 |
|---|---|
| 服务器原始 | `/root/rivermind-data/repro/deco-b123/<exp>_2026103x/` |
| 本地 ZCY 副本 | `C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro\<exp>_2026103x\` |
| 仓库归档 CSV | `C:\Users\TYY\Documents\MyDocument\CVPR-2026\claude\analysis\results\20261031\<group>\` |
| 报告 | `C:\Users\TYY\Documents\MyDocument\CVPR-2026\claude\reports\` |
| GitHub | `Youyuan287/DeCo-3DGS` → `experiments/lerf_6760/` |

---

## 1. 基线（先跑这两个，所有比较的锚点）

### 1.1 ReLaGS 基线（C1 off，macro 64.01）

```bash
$PY $REPO/deco/lerf_ovs/eval_lerf_ovs.py --scene <sc> --source $SRC/<sc> --model $MODEL/<sc> \
  --nag $MODEL/<sc>/sai_nag.pt --gt_root $GT --out <out>/<sc> --csv <out>/<sc>/per_query.csv \
  --center none --renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 \
  --sel gap --norm none --text-center none --tau 0.02 --text-agg lse
```

结果：服务器 `/root/rivermind-data/relags_baseline_20261031/`；仓库 `analysis/results/20261031/baselines/relags/`。

### 1.2 DeCo B0（C1 on，macro 63.9263）

```bash
bash /root/rivermind-data/reproduce_b0_four_scenes.sh
# 或 SCENES=ramen bash /root/rivermind-data/reproduce_b0_four_scenes.sh
```

结果：`/root/rivermind-data/b0_reproduce_XXXX/<sc>/per_query.csv`；仓库 `analysis/results/20261031/baselines/b0/`。
指南：`C:\Users\TYY\Documents\ChatGPT\ZCY-4090\B0_REPRODUCTION_GUIDE.md`。

---

## 2. 当前最好结果：gateE 0.6760

**脚本**：`analysis/lerf_c_mask_gate.py`（服务器 `$REPO/deco/lerf_ovs/`）

```bash
for sc in figurines ramen teatime waldo_kitchen; do
  $PY $REPO/deco/lerf_ovs/lerf_c_mask_gate.py --scene $sc --source $SRC/$sc --model $MODEL/$sc \
    --nag $MODEL/$sc/sai_nag.pt --gt_root $GT --out <OUT> \
    --preds /root/rivermind-data/repro/deco-b123/lerf_subset_20261030/preds.csv \
    --evidence /root/rivermind-data/repro/deco-b123/lerf_evidence_20261030 \
    --id_csv /root/rivermind-data/repro/deco-b123/lerf_segidentity_20261030/${sc}_segidentity.csv \
    --center image
done
```

- 服务器结果：`/root/rivermind-data/repro/deco-b123/gatee_20261031/`
- 本地 ZCY：`cmaskgate2_20261031/`
- 仓库 CSV：`analysis/results/20261031/gatee/`
- GitHub 复现包：`experiments/lerf_6760/`
- 汇总：`$PY scripts/summarize_6760.py --results results/gatee --identity inputs/identity`

---

## 3. 实验总表

| # | 实验 | 脚本 / runner | 结果目录（服务器 / 本地 ZCY） | 判定 |
|---|---|---|---|---|
| 1 | Phase A 观测证据 dump | `lerf_identity_dump.py` / `run_lerf_identity_dump.sh` | `lerf_identity_20261030` | 证据 |
| 2 | Phase B 段级证据 dump | `lerf_segment_identity_dump.py` / `run_lerf_segment_identity_dump.sh` | `lerf_segidentity_20261030` | 证据 |
| 3 | leave-query-view-out dump | `lerf_segment_identity_dump_lo.py` | `segid_lo_20261031` | 公平性检查 |
| 4 | 2D mask 文本排名 | `lerf_mask_rank_probe.py` / `run_lerf_mask_rank_probe.sh` | `lerf_maskrank_20261031` | 诊断 |
| 5 | 2D mask 中心化/排名 | `lerf_mask_feature_diag.py` / `run_lerf_mask_feature_diag.sh` | `lerf_maskdiag_20261031` | 诊断 |
| 6 | 2D→3D lifting | `lerf_lift_probe.py` / `run_lerf_lift_probe.sh` | `lerf_lift_20261031` | 诊断 |
| 7 | 跨视角同/异类余弦 | `lerf_crossview_feature_consistency.py` / 同名 runner | `lerf_crossview_20261031` | 诊断 |
| 8 | 变换（viewz/white） | `lerf_crossview_transforms.py` / 同名 runner | `lerf_crossview_xform_20261031` | 诊断 |
| 9 | viewz 文本排名 | `lerf_viewz_text_rank.py` / 同名 runner | `lerf_viewz_20261031` | 诊断 |
| 10 | 2D mask 覆盖率 | `lerf_2dmask_coverage.py` / 同名 runner | `lerf_2dmaskcov_20261031` | 诊断 |
| 11 | module C（prune+include） | `lerf_prune_include.py` / `run_lerf_prune_include.sh` | `lerf_pruneinclude_20261030` | PASS（+2.10） |
| 12 | 掩膜级 rerank（id_top3） | `lerf_maskagg_rerank.py --column id_top3 --center image` | `maskagg_20261031` | PASS（+2.26） |
| 13 | 掩膜级 rerank（单向量 id_mean） | 同上 `--column id_mean` | `maskagg_mean_20261031` | +0.56 |
| 14 | 掩膜级 rerank（id_mean_c） | 同上 `--column id_mean_c` | `maskagg_meanc_20261031` | +0.28 |
| 15 | 秩融合 | 同上 `--fusion rank` | `maskagg_20261031/*_rank.csv` | FAIL |
| 16 | C + mask 排序叠加 | `lerf_mask_c_stack.py` | `maskcstack_20261031` | FAIL |
| 17 | gateB/gateE | `lerf_c_mask_gate.py` | `cmaskgate2_20261031` / `gatee_20261031` | **PASS 0.6760** |
| 18 | leave-out 端到端 | 同 17，`--id_csv segid_lo...` | `gatee_lo_20261031` | 0.6753（不是捷径） |
| 19 | 几何约束 2D mask 选择 | `lerf_mask_geom.py` | `maskgeom_20261031` / `maskgeom2_20261031` | Stage 1 FAIL |
| 20 | 查询视角 mask proposal | `lerf_mask_proposal.py` | `maskprop_20261031` | recall 0.108 |
| 21 | proposal + gate 组合 | `lerf_maskprop_gate.py` | `maskprop_gate_20261031` / `maskprop_gate2_20261031` | union FAIL；证据竞争 +0.002 |
| 22 | 跨视角 mask 选择 | `lerf_mask_xview.py` | `maskxview_20261031` | FAIL（0.40/0.30/0.025） |
| 23 | SREM Phase A | `lerf_srem_dump.py` | `srem_20261031`（初版）/ `srem2_20261031`（修正） | **Stage 1 FAIL** |
| 24 | ROFA 忠实复现（DeCo 候选） | `lerf_rofa_eval.py` | `lerf_rofa_20261031` | 作废（底座不对） |
| 25 | ROFA merge_proj A/B | `analysis/relags_rofa/rebuild_mode.sh`、`run_relags_rofa.sh` | `work_rofa/`、`results_rofa/` | 仅相对；绝对不可比 |
| 26 | bank B2 ROFA | `analysis/run_rofa_bank.sh` | `rofa_bank_20261031` | 中性；C1-off 版无效 |
| 27 | 候选间排斥 readout | `eval_lerf_ovs_repulse.py` | `rofa_bank_20261031/repulse*` | FAIL |
| 28 | oracle/头寸 | `lerf_oracle.py` | `lerf_oracle_20261030` | 诊断 |

---

## 4. 关键命令（复制即用）

### 4.1 掩膜级 rerank（id_top3，B0）

```bash
$PY $REPO/deco/lerf_ovs/lerf_maskagg_rerank.py --scene <sc> --source $SRC/<sc> --model $MODEL/<sc> \
  --nag $MODEL/<sc>/sai_nag.pt --gt_root $GT --out <OUT> \
  --id_csv /root/rivermind-data/repro/deco-b123/lerf_segidentity_20261030/<sc>_segidentity.csv \
  --center image --column id_top3 --fusion z --lam 1.0
```

### 4.2 SREM dump（Phase A）

```bash
$PY $REPO/deco/lerf_ovs/lerf_srem_dump.py --scene <sc> --source $SRC/<sc> --model $MODEL/<sc> \
  --nag $MODEL/<sc>/sai_nag.pt --gt_root $GT --sas $SAS/<sc>_new --out <OUT>
```

### 4.3 Phase B 证据 dump（gateE 的输入）

```bash
$PY $REPO/deco/lerf_ovs/lerf_segment_identity_dump.py --scene <sc> --source $SRC/<sc> \
  --model $MODEL/<sc> --nag $MODEL/<sc>/sai_nag.pt --gt_root $GT --sas $SAS/<sc>_new --out <OUT>
```

### 4.4 几何 mask 选择

```bash
$PY $REPO/deco/lerf_ovs/lerf_mask_geom.py --scene <sc> --source $SRC/<sc> --model $MODEL/<sc> \
  --nag $MODEL/<sc>/sai_nag.pt --gt_root $GT --out <OUT> \
  --preds /root/rivermind-data/repro/deco-b123/lerf_subset_20261030/preds.csv \
  --evidence /root/rivermind-data/repro/deco-b123/lerf_evidence_20261030 \
  --id_csv /root/rivermind-data/repro/deco-b123/lerf_segidentity_20261030/<sc>_segidentity.csv
```

### 4.5 结果汇总

```bash
# gateE
$PY analysis/results/.../summarize_6760.py --results ... --identity ...
# 或按 reports/EXPERIMENT_RECORD.md 的 Python 片段
```

---

## 5. 结果目录与仓库归档对应

| 仓库 `analysis/results/20261031/` | 内容 | 来源服务器/ZCY |
|---|---|---|
| `baselines/relags`、`baselines/b0` | 两条基线逐查询 | `relags_baseline_20261031`、`b0_reproduce_*` |
| `gatee`、`cmaskgate`、`margingate`、`gatee_lo` | gating 逐查询 | `gatee_20261031`、`cmaskgate2_20261031`、`margin2_20261031`、`gatee_lo_20261031` |
| `maskagg`、`maskagg_mean`、`maskagg_meanc`、`maskagg_rank` | 掩膜 rerank 各读出 | `maskagg_*_20261031` |
| `maskcstack`、`maskgeom`、`maskprop`、`maskprop_gate`、`maskxview` | 叠加/几何/proposal | 同名 ZCY |
| `diagnostics` | identity/segidentity/crossview/xform/maskrank/maskdiag/viewz/lift | `lerf_*_20261030/31` |
| `segid_lo` | leave-out token 证据 | `segid_lo_20261031` |
| `srem` | SREM Phase A 修正版 | `srem2_20261031` |
| `rofa` | ROFA A/B 逐查询 | `relags_mp_ab` |

---

## 6. 报告索引

| 报告 | 内容 |
|---|---|
| `reports/PROJECT_HANDOFF_20261101.md` | **总交接**：目标/全部实验/结论/计划/泄露审计/文件索引 |
| `reports/CONTRIBUTION_REANCHOR_20261101.md` | **贡献重锚定**：出处核验、module C 学习性质审计、training-free gateE-TF 0.6716 |
| `reports/REFRAME_DECISION_AUDIT_20261101.md` | **叙事重构**：判决审计、配对证书 0.6711、四条可证伪预测、非重合表述词表 |
| `reports/EXPERIMENT_DOC_LERF_SCANNET_20261102.md` | **简要实验文档**：摘要/引言/方法/实验（LERF+ScanNet）+ 小记录；§3.4.2.1 视角中心化归因（独立增益 +0.39，上界 ~0.8，收口）；§3.4.3 局部对比 B0–B6 全负（随机≈空间邻域）；§6 下一步=二维→三维映射可辨识性实验（需先有 level 0/1 观测） |
| `reports/CVPR_DRAFT_DECISION_AUDIT_20261102.md` | **CVPR 论文草稿**（中文，CVPR 结构）：判决审计 + 层匹配原则 + OpenSeg-768 ScanNet 记录 + 补充材料 |
| `reports/SCANNET_AUDIT_NEGATIVE_20261101.md` | **ScanNet 负结果**：审计器不迁移闭集分类（40.36→36.73），配对前提不成立 |
| `reports/3DOVS_DECISION_AUDIT_NEGATIVE_20261102.md` | **3D-OVS 负结果**：证书 0.98–1.00 时改判仍掉分（70.05→63.51），证据≠IoU |
| `reports/3DOVS_REPRO_GAP_AND_AUDIT_20261102.md` | **3D-OVS 复现差距根因**：三级 oracle 分解，特征分辨率（level-1 天花板 0.7735） |
| `reports/EXPERIMENT_RECORD.md` | 总记录（双基线、实验判定、协议警告） |
| `reports/GPT_HANDOFF_DOSSIER.md` | 给 GPT 的完整交接 |
| `reports/PART_MULTIGRANULARITY_GATE.md` | Part 多粒度：2D/3D 候选、grounding、enrichment、selector、lift |
| `reports/GAUSSIAN_HEADROOM_AND_DIGS_POSITIONING.md` | 高斯级头寸、DIGS 定位 |
| `reports/BOUNDARY_LOSS_ANALYSIS.md` | BIoU 边界审计 |
| `reports/WHY_CONSISTENCY_HURTS.md` | 2D/3D、一致性崩塌的逐场景分析 |
| `reports/PROTOCOL_QUERY_CONDITIONED.md` | late interaction vs 单向量、协议公平性 |
| `reports/HEADROOM_ANALYSIS.md` | 池内/2D/贪心 oracle 头寸 |
| `reports/3DOVS_OPEN_PROBLEMS.md` | 领域问题清单 |
| `reports/CVPR_DIRECTION_QRES.md` | SREM/objectness 方向 |
| `reports/SREM_PHASE_A_REPORT.md` | SREM Phase A 负结果 |
| `reports/LERF_MASKAGG_REPORT.md` | mask rerank 0.6619 |
| `reports/LERF_ROFA_EXPERIMENT_LOG.md` | ROFA/重建读权作废范围 |

---

## 7. 状态标记（避免找错）

- **有效绝对数**：只有官方读权 + 上述 B0/ReLaGS 配置。
- **仅相对**：`work_rofa`、`results_rofa/mp_*`（重建读权）。
- **作废**：`lerf_rofa_20261031`（DeCo 候选/错误底座）、`maskagg_*` 的 C1-off bank 数字。
- **修正后**：SREM 用 `srem2_20261031`，不要用 `srem_20261031` 初版。
- **当前最好**：`gatee_20261031` = 0.6760。

---

## 8. Part 多粒度 / grounding / 边界（2026-11-01 新增）

服务器数据盘：`/root/rivermind-data/part_grounding_20261101/`
（`pylibs/`、`model/grounding-dino-tiny/`、`code/`、`out/`、`hf_cache/`；**不要放系统盘**）

| 内容 | 脚本 | 结果 |
|---|---|---|
| 2D 多粒度 oracle | `analysis/lerf_multigranularity_gate.py` | `analysis/results/20261101/mg_gate2/` |
| 文本选 mask | `analysis/lerf_mg_select.py` | `analysis/results/20261101/mg_select/` |
| 高斯级头寸 | `analysis/lerf_gaussian_headroom.py` | `analysis/results/20261101/gauss_headroom3/` |
| 边界 BIoU | `analysis/lerf_boundary_probe.py` | `analysis/results/20261101/boundary_oracle/` |
| GroundingDINO parent/part | `analysis/lerf_part_grounding_gate.py`、`lerf_part_grounding_mh.py` | 服务器 `part_grounding_20261101/out/` |
| 无 GT parent | `analysis/lerf_part_grounding_nogt.py` | 服务器 `out/part_grounding_nogt.csv` |
| enrichment | `analysis/lerf_part_enrichment_gate.py` | `analysis/results/20261101/enrichment/` |
| part-whole selector | `analysis/lerf_part_selector_gate.py` | `analysis/results/20261101/selector/` |
| 3D 候选覆盖 | 服务器 `code/part_3d_recall2.py` | `part_3d_recall` 日志 |

当前状态：2D/3D leaf part 候选存在；root-first 候选策略是主因；无 GT parent/part 选择器均失败；
208 最终指标仍 0.6760；+2.1 是 GT-parent oracle。详见 `reports/PART_MULTIGRANULARITY_GATE.md`。

---

## 9. 贡献重锚定与 training-free gateE（2026-11-01 新增）

**背景**：module C 的分支使用冻结 LOSO GBM 的预测 IoU（`lerf_subset_20261030/preds.csv`
的 `pred` 列），属于学习组件；gateE 的回退分支是 S_C，因此 0.6760 不能作为 training-free 结果。
详见 `reports/CONTRIBUTION_REANCHOR_20261101.md`。

**新主结果（官方锚点 center=none，208 条）**：main 0.6401 → C_tf 0.6572 → mask 0.6582
→ **审计器 0.6711（+3.13，CI [+1.14,+5.48]）**；确定性门控变体 0.6716。
跨场（center=image 63.93）0.6691（+2.99）；leave-query-view-out 0.6675（+2.77）。
机制与表述见 `reports/REFRAME_DECISION_AUDIT_20261101.md`。

| 内容 | 脚本 / runner | 结果 |
|---|---|---|
| center=none 三份 dump | `run_cnone_dumps.sh`（服务器；dump 脚本 `*_cnone.py`） | `cnone_20261101/{cand,ev,segid}/` |
| gateE-TF（pred=NAG root score） | `run_gate_cnone_tf.sh` | `cnone_20261101/gate/` |
| 离线门控分析 | `analysis/lerf_gate_trainingfree.py` | `analysis/results/20261101/cnone_gate_tf/` |
| 判决审计（配对证书） | `analysis/lerf_witness_certificate.py` | `analysis/results/20261101/witness/`（witness dump）+ 同目录 |
| 跨场复核（center=image） | 服务器 `run_crossfield_image.sh` | `analysis/results/20261101/{witness_image,gate_image,ev_image,segid_image}/` |

口径提醒：0.6716（官方 64.01 锚点 + training-free）与 0.6760（center=image 策略移植 + 学习器）
不属同一协议，不能并列。