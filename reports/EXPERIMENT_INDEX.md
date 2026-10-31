# 实验索引：脚本、运行指令、结果位置

日期：2026-11-01
目的：以后要复跑或找结果时，从这张表直接定位。
所有 GPU 实验都在服务器 `fj01` 上跑；本地仓库只存脚本/报告/CSV 副本。

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
| `reports/EXPERIMENT_RECORD.md` | 总记录（双基线、实验判定、协议警告） |
| `reports/GPT_HANDOFF_DOSSIER.md` | 给 GPT 的完整交接 |
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