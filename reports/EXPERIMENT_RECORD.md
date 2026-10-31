# LERF-OVS 实验总记录（2026-10-31 归档）

本文件是 2026-10-31 这一轮所有 LERF-OVS 实验的总索引：基线口径、实验结论、可复跑命令、数据与代码位置。
详细分析见：
- `reports/LERF_ROFA_EXPERIMENT_LOG.md`（ROFA/跨视角实验过程与作废范围）
- `reports/LERF_2D3D_FEATURE_DIAGNOSTICS.md`（2D/3D 特征区分度诊断，汇报用）
- `reports/LERF_MASKAGG_REPORT.md`（掩膜级聚合重排，预注册 PASS）
- `reports/LERF_MODULE_PRUNE_INCLUDE_REPORT.md`（模块 C，另一个过门控结果）

---

## 1. 两个官方基线（不能混用）

评测器：`REPO/deco/lerf_ovs/eval_lerf_ovs.py`，`--topk 5 --level_until 3 --filter_small 50 --sel gap --text canon --renorm 0`。

| 基线 | 读端 | 四场景 scene mIoU | macro | 用途 |
|---|---|---|---:|---|
| **ReLaGS** | `--center none`（C1 关闭） | 0.6329 / 0.5115 / 0.8097 / 0.6062 | **0.6401** | ROFA、特征侧、读端改 ReLaGS |
| **DeCo B0** | `--center image`（C1 开启） | 0.5859 / 0.5181 / 0.7939 / 0.6592 | **0.6393** | DeCo bank/evidence/M1–M3 |

B0 的六位核对值：58.5869 / 51.8076 / 79.3896 / 65.9209，macro 0.6392625，与
`B0_REPRODUCTION_GUIDE.md` 一致；复跑脚本 `ZCY-4090/reproduce_b0_four_scenes.sh`。

---

## 2. 有效性分级

- **A（可用绝对数）**：官方读权 `ref/relags_hf/<scene>/sai_nag.pt` + 官方 GT + 上述评测器。
- **B（仅同底座相对可比）**：自己重跑的 `merge_proj` 读权。ramen 重建 0.4935–0.4962 vs ReLaGS 0.5115；
  官方 `sai_nag.pt` 的 level-0 是 7852 个超点，而发布 `nag-l1.pt` 是 7936，**官方读权不可由发布产物完全重建**。
- **C（仅机理诊断）**：DeCo 观测/候选上的代理实验（ROFA、center、bank 的部分设置）。

---

## 3. 实验索引与结论

| # | 实验 | 底座 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | ROFA 忠实复现（DeCo 候选，raw/center） | C | raw −0.0341、rofa −0.0434、rofac −0.0329 vs main；205/208 条相同；剔除 4.8% 观测 | 作废（底座不对） |
| 2 | 官方读权上 B2 ROFA（bank） | A | figurines/ramen 的 off/raw/center 逐条相同；bank 融合本身 −4 分 | ROFA 无效；bank 需按 B0 重跑 |
| 3 | 重建 merge_proj ROFA A/B | B | macro off 0.5359 / raw 0.5630 / center 0.5527；raw−off +0.0271 | 仅相对有效，绝对数不可比 |
| 4 | 官方 level-0 分区重建 | B | ramen 0.4788；level-1 1835 vs 官方 1853 | 仍不可对齐 |
| 5 | 候选间排斥读端（方案 3） | A | figurines 0.5468 vs 0.6329 | FAIL，保留负结果 |
| 6 | 2D/3D 特征诊断 | C | 同物体余弦 0.60–0.76 vs 不同类 0.50–0.56；2D rank1 found 77.4%；提升损失 0.049 IoU；一致性聚合 found 31.5% | 机理证据 |
| 7 | **掩膜级聚合重排** | **A** | **ReLaGS 0.6401→0.6583（+0.0182, CI +0.0089~+0.0282）**；**B0 0.6393→0.6619（+0.0226, CI +0.0136~+0.0319）**；found/rank2/recall 持平，deep +25.4/+30.2 | **PASS** |
| 8 | 秩融合替代 z 融合 | A | ReLaGS +0.0069（CI 含 0）；B0 +0.0080（未达 +0.01） | FAIL（修掉 λ 敏感但丢 ramen 收益） |
| 9 | 模块 C + mask 排序叠加 | A（B0） | main 0.6393 / C 0.6603 / stack 0.6478；stack−C −0.0125（CI 含 0），found 0.8177 vs C 0.8312，waldo −8.2 | FAIL（交互破坏 C 的 guard；且 mask 排序存在 nan 填充 bug，结果仅作记录） |
| 10 | 逐查询 gating（C vs mask） | A（B0） | **gateB（mask 改动用 mask，否则 C）：0.6757**，+3.64 vs main，+1.54 vs C，+1.38 vs mask；found 0.8312（与 C 持平），deep 0.6607 | **PASS（组合自逐查询实测；teatime/waldo 待 GPU 空闲后整体重跑确认）** |

### 3.1 掩膜级聚合重排（当前最好结果）

- 方法：`S = z(NAG) + 1.0·z(id_top3_mask)`，保持 released gap 的 k，仅在 k 内重排。
- 分场景 Δ（ReLaGS / B0）：figurines +1.27 / +3.02；ramen +6.14 / +6.10；teatime 0 / 0；waldo −0.11 / −0.07。
- 门控：Δ≥+0.01、CI 下界>0、无场景恶化、found 不退化，全部 PASS。
- λ 敏感性：0.5 → +0.37；1.0 → +1.82；2.0 → −2.00。**λ 敏感**，下一步做秩融合。

### 3.2 剩余头寸与两条失败路线（2026-10-31 补充）

- **池内选择头寸**：released pool 的 oracle（每查询最佳候选）macro **0.7094**，
  当前 mask rerank 0.6583（ReLaGS）/0.6619（B0），池内仍有约 **+5 macro** 可拿；
  分场景 oracle：figurines 0.734、ramen 0.628、teatime 0.801、waldo 0.667。
- **秩融合 FAIL**：ReLaGS +0.0069（CI [−0.0002,+0.0169]）；B0 +0.0080（CI [+0.0030,+0.0120]）。
  修掉了 λ 敏感，但丢掉 ramen 的 +6.1，说明 z 融合的尺度信息有用。
- **模块 C + mask 排序叠加 FAIL**（B0，208 条完整）：main 0.6393 / C 0.6603 / stack 0.6478；
  stack−C −0.0125（CI [−0.0369,+0.0101]）；found 0.8177 vs C 0.8312（−1.35）；waldo −8.2。
  两者改动查询几乎不重叠（32 vs 12，重叠 5），但“先按 mask 排序再套 C 的 prune”破坏了 C 的 guard。
- center none 的叠加结果无效：`preds/evidence` 由 C1-on 生成，候选 rank 与 center-none 池不匹配，
  多条查询被跳过（46/56、47/71、54/59、20/22），不能使用。

结论：信号互补，但不能靠改排序叠加；下一步应做 **per-query gating** 或 **feature-level 聚合**，
而不是把 mask 顺序直接塞进 C。

### 3.3 逐查询 gating（当前最好，0.6757）

规则（训练-free，只看“哪一路改变了 released 选择”）：
- gateA：C 改变选择 → 用 C；否则用 mask 重排（预注册主规则）。
- **gateB（更好）：mask 改变选择 → 用 mask；否则用 C。**

B0 底座，四场景（按图平均）：

| 选择 | macro | figurines | ramen | teatime | waldo |
|---|---:|---:|---:|---:|---:|
| main | 0.6393 | 0.5859 | 0.5181 | 0.7939 | 0.6592 |
| 模块 C | 0.6603 | 0.5941 | 0.5649 | 0.8223 | 0.6597 |
| mask 重排 | 0.6619 | 0.6160 | 0.5791 | 0.7939 | 0.6585 |
| gateA | 0.6713 | 0.5941 | 0.6092 | 0.8223 | 0.6597 |
| **gateB** | **0.6757** | **0.6129** | **0.6092** | **0.8223** | **0.6585** |

配对场景分层 bootstrap（B=2000）：gateB−C **+0.0155 [+0.0044,+0.0286]**；
gateB−mask **+0.0139 [+0.0022,+0.0259]**；gateB−main **+0.0364**。
分族：found 0.8312（与 C 持平）、rank2 0.6395、deep **0.6607**（C 0.4446 / mask 0.5281）、recall 0.0198。

**状态说明**：figurines/ramen 来自修复后的完整 gate 运行；teatime/waldo 因 GPU 被另一会话占用
（三个 `enc_i16.py` 各占 ~7GB），由已实测的 C/mask 选择精确组合（teatime mask 不改选择 → C；
waldo 两路改同两条 → mask）。需在 GPU 空闲后整体重跑确认。另：gate 脚本此前的 `nan` 填充 bug
已修复（缺失 id_top3 用有限值均值填充），修复前的 mask 分支会少救 3 条 deep 查询。

---

## 4. 可复跑命令（fj01）

### 4.1 ReLaGS 基线（C1 关闭）

```bash
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
REL=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
M=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
export RELAGS_REPO=$REL HF_HUB_OFFLINE=1 PYTHONPATH="$REL:$REL/ext:$REPO/deco:$REPO/deco/lerf_ovs"
$PY $REPO/deco/lerf_ovs/eval_lerf_ovs.py --scene ramen --source $SRC/ramen --model $M/ramen \
  --nag $M/ramen/sai_nag.pt --gt_root $GT --out /tmp/out --csv /tmp/out/q.csv \
  --center none --renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 \
  --sel gap --norm none --text-center none --tau 0.02 --text-agg lse
```

### 4.2 DeCo B0（C1 开启）

```bash
# 使用归档脚本（会新建 mktemp 输出目录）
scp ZCY-4090/reproduce_b0_four_scenes.sh fj01:/root/rivermind-data/
ssh fj01 "bash /root/rivermind-data/reproduce_b0_four_scenes.sh"
```

### 4.3 掩膜级聚合重排

```bash
$PY $REPO/deco/lerf_ovs/lerf_maskagg_rerank.py --scene ramen --source $SRC/ramen --model $M/ramen \
  --nag $M/ramen/sai_nag.pt --gt_root $GT --out /root/rivermind-data/repro/deco-b123/maskagg_20261031 \
  --id_csv /root/rivermind-data/repro/deco-b123/lerf_segidentity_20261030/ramen_segidentity.csv \
  --center none --lam 1.0
```

`--center image` 即 B0 读端。脚本与预注册：`analysis/lerf_maskagg_rerank.py`、`analysis/PREREG_maskagg.md`。

### 4.4 重建 merge_proj（仅 B 级）

```bash
# 需先提取官方 level-0 分区或用发布 nag-l1；脚本：
# analysis/relags_rofa/rebuild_official_part.sh / rebuild_mode.sh
```

---

## 5. 数据与代码位置

仓库内（本次归档副本）：

| 内容 | 路径 |
|---|---|
| ReLaGS 基线逐查询 | `analysis/results/20261031/baselines/relags/<scene>.csv` |
| B0 基线逐查询 | `analysis/results/20261031/baselines/b0/<scene>.csv` |
| 掩膜级重排逐查询 | `analysis/results/20261031/maskagg/` |
| 2D/3D 诊断 CSV | `analysis/results/20261031/diagnostics/` |
| ROFA 四场景 A/B 逐查询 | `analysis/results/20261031/rofa/` |
| 预注册文件 | `analysis/PREREG_rofa.md`、`PREREG_maskagg.md`、`PREREG_crossview_three.md` |
| 脚本 | `analysis/lerf_*.py`、`analysis/relags_rofa/` |

服务器原始产物：

| 内容 | 路径 |
|---|---|
| 官方读权 | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/<scene>/sai_nag.pt` |
| B0 复跑输出 | `/root/rivermind-data/b0_reproduce_CD9FUU8X/` |
| ReLaGS 基线输出 | `/root/rivermind-data/relags_baseline_20261031/` |
| 掩膜级重排 | `/root/rivermind-data/repro/deco-b123/maskagg_20261031/` |
| 2D/3D 诊断 | `/root/rivermind-data/repro/deco-b123/lerf_*_20261031/` |
| 重建读权 | `/root/rivermind-data/ReLaGS_Reproduction/work_rofa/`、`work_official_part/` |

---

## 6. 关键结论

1. **两个基线必须分开**：ReLaGS = C1 关闭 64.01；DeCo B0 = C1 开启 63.93。混用会产生 4–5 分的单场景假差异。
2. **ROFA 本身不是瓶颈**：在可编辑路径上 raw ROFA 中性偏正，一致性改进（center）无收益；官方读权无法做锚定消融。
3. **问题在聚合单元**：区域混合观测的一致性聚合把 found 命中拉到 31.5%，而物体级 mask 特征的简单 top-3 聚合保留 84.7%。
4. **掩膜级聚合重排过门控**：两条官方基线 macro +1.8/+2.3，收益集中在 deep（+25/+30）与 ramen（+6.1），found 不退化。
5. **λ 敏感**：0.5/+0.37、1.0/+1.82、2.0/−2.00；下一步做秩融合（需独立预注册）。

---

## 7. 归档与备份

- 本记录与三份详细报告：`reports/`。
- 结果 CSV：`analysis/results/20261031/`。
- 备份：`CVPR-2026/claude-repo-20261031.bundle`（git 全历史）与
  `CVPR-2026/claude_backup_20261031.zip`（工作树，含结果 CSV）。