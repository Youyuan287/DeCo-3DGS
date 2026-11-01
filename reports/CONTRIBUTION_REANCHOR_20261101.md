# 贡献重锚定与协议修正报告（2026-11-01）

目的：按已核验的 LERF-OVS 出处链重报基线，审计 module C / gateE 的协议合规性，
给出 **training-free 版本** 的贡献与确认后的收益。

数据/评测：官方读权 `ref/relags_hf/<scene>/sai_nag.pt` + 官方 GT + 官方评测口径
（`--topk 5 --level_until 3 --filter_small 50`；每图平均 → 场景平均 → 四场景 macro）。
全部数字在服务器 `fj01` 上重算，结果 CSV 归档在 `analysis/results/20261101/cnone_gate_tf/`。

---

## 1. 出处核验（provenance）

| 口径 | macro | 来源 / 核验状态 |
|---|---:|---|
| 路线 A：官方权重 `ref/relags_hf`，center=none | **0.6401** | `results/lerf_scene_metrics.csv`（63.29/51.15/80.97/60.62），本次复跑逐位一致 ✓ |
| center=image（DeCo 选簇策略**移植**） | 0.6393（嵌套）/ 0.6396（flat） | 本仓库 gatee CSV 重聚合；与出处文档 63.93/63.37 同批产物 ✓ |
| 路线 B：我方重建语义场（gap 补丁） | 0.6471 | 出处文档记录；位于 **hn01**，本机不可达，未复核 |

结论：

1. **唯一可直接对外的基线是路线 A 的 64.01**；本次在 fj01 上独立复跑得到 0.6401，
   且 center=none 候选 dump 的 `main` 逐场景等于官方值（如 figurines 0.6329 = 63.29）。
2. 0.6393 是"DeCo 选簇策略套在 ReLaGS 语义场上"的数字，**不是 DeCo 论文复现**，
   也不应作为论文主基线；只能作辅助消融。
3. 路线 B 的 64.71 未在本机复核，引用时须标注来源（hn01）。

---

## 2. 协议审计：module C 与 gateE 的真实性质（关键发现）

`lerf_subset_20261030/preds.csv` 的 `pred` 列是**冻结的 LOSO IoU 回归器（GBM）的预测 IoU**，
不是 NAG 相似度。实测（figurines frame_00041）：

| 查询 | rank | pred（预测） | 真实 IoU |
|---|---:|---:|---:|
| toy elephant | 1 | 0.8522 | 0.9104 |
| waldo | 1 | 0.6936 | 0.7301 |
| old camera | 2 | 0.2004 | 0.6114 |

`lerf_c_mask_gate.py` 中 module C 的剪枝规则 `pv ≥ 0.5·max_pred` 使用的正是这一列。
因此：

- **module C = 学习组件**（训练集为其他 3 场景的 GT IoU 标签，LOSO），
  与"training-free、GT 不进选择"的硬约束冲突；
- **gateE 的回退分支就是 S_C**，所以此前报告的 **0.6760 含学习组件**，
  不能作为 training-free 方法结果；
- 纯 training-free 门控（released S0 vs mask S_M，按 id_top3 证据比较）只有
  **0.6621**（center=image 锚点），相对 mask rerank 的 0.6619 无增益 →
  此前 gateE 的增益主要来自学习型 S_C。

---

## 3. 修正：training-free 替换后的管线与收益

### 3.1 改动

1. `pred` 改用 **NAG root score**（官方 released 特征，无 GT、无学习器）；
   在该定义下 `0.5·max` 剪枝从不触发（root score 全部 > 0.5·max）。
2. 于是 module C 退化为 **尺度门控挑战者并入**（training-free）：
   若 `area[0] < 0.001` 且 `max(area) > 2·area[0]`，把 `z(area)+z(mean_vis)` 最大的候选并入选择集。
   触发于 **20/208** 查询。
3. mask rerank 与 gateE 公式不变：`S = z(NAG)+z(id_top3)`，
   `S_gateE = S_M if set(S_M)≠set(S0) and e_M>e_C else S_C`。
4. 全部在 **center=none（官方 64.01 锚点）** 下重跑；三份静态 dump
   （candidates / evidence / segidentity）也在 center=none 下重新生成。

### 3.2 主结果（官方锚点，208 条，嵌套口径）

| 选择 | macro | Δ vs main | figurines | ramen | teatime | waldo |
|---|---:|---:|---:|---:|---:|---:|
| main（released，复现 0.6401 ✓） | 0.6401 | — | 0.6329 | 0.5115 | 0.8097 | 0.6062 |
| C_tf（尺度并入，training-free） | 0.6572 | +0.0171 | 0.6349 | 0.5779 | 0.8097 | 0.6062 |
| mask rerank（late interaction） | 0.6582 | +0.0181 | 0.6473 | 0.5709 | 0.8097 | 0.6051 |
| gateB-TF | 0.6675 | +0.0274 | 0.6350 | 0.6203 | 0.8097 | 0.6051 |
| gateE-TF（确定性变体） | 0.6716 | +0.0315 | 0.6493 | 0.6221 | 0.8097 | 0.6051 |
| **审计器（主，配对证书）** | **0.6711** | **+0.0313** | 0.6493 | 0.6203 | 0.8097 | 0.6051 |

- **主方法为审计器**（配对半样本多数证书 + 范围支配），见
  `reports/REFRAME_DECISION_AUDIT_20261101.md`；gateE-TF 是不做配对证书、
  直接用证据极值的确定性变体，两者差 0.0005，说明增益来自选择性而非阈值。
- 审计器配对 bootstrap（场景分层，B=2000）：**Δ=+0.0313，95% CI [+0.0114, +0.0548]**。
- 无场景恶化超过 −0.01（waldo −0.0011）；found 族 −0.0045（阈值 0.005 内）。

分族：

| 族 | n | main | C_tf | mask | gateE-TF |
|---|---:|---:|---:|---:|---:|
| found | 129 | 0.8316 | 0.8271 | 0.8320 | 0.8271 |
| rank2 | 24 | 0.6915 | 0.6915 | 0.6826 | 0.6904 |
| deep | 25 | 0.2914 | 0.4960 | 0.4847 | **0.6515** |
| recall | 30 | 0.0218 | 0.0255 | 0.0218 | 0.0255 |

### 3.3 机理

- mask rerank 只在 **26/208** 条查询上改变选择集；gateE-TF 决定用或不用。
- 增益集中于 ramen（逐查询平均 +0.1078）与 figurines（+0.0156）；
  teatime/waldo 持平。
- deep 族 0.2914 → 0.6515：尺度并入 + mask 证据门控把深层候选救回来；
  found 基本不动（0.8316 → 0.8271）。

### 3.4 口径对照（不要混用）

| 数字 | 底座 | C 的性质 | 用途 |
|---|---:|---|---|
| **0.6711（审计器）** | **0.6401 官方** | training-free | **可作方法主结果** |
| 0.6716（gateE-TF） | 0.6401 官方 | training-free | 确定性变体（差 0.0005） |
| 0.6760（原 gateE） | 0.6393 策略移植 | LOSO 学习器 | 只能作"学习层"消融/上界 |
| 0.6603 / 0.6619（原 C / mask，center=image） | 0.6393 | 学习 / training-free | 辅助消融 |
| 0.6471（路线 B） | 我方重建 | — | 未复核，引用须注明 |

---

## 4. 现在可以主张 / 不能主张

**可以主张（training-free，官方锚点）：**

1. 多视角 mask late interaction（`id_top3`）重排：64.01 → 65.82。
2. 尺度门控挑战者并入：64.01 → 65.72。
3. 判决审计（配对证书 + 范围支配）：**64.01 → 67.11（+3.13）**，
   CI 下界 +1.14；deep 0.2914→0.6515；found 0.8316→0.8275（−0.41）；无场景恶化 >−0.01。
   改判集中在场排得深的查询（24%）而非 found（2.3%）；π∈[0.5,0.95] 结果仅差 0.0036。

**不能主张：**

- 0.6760 作为 training-free 方法结果（含 LOSO 学习器）；
- GT-parent +2.1（泄露，见 PART 报告）；
- 0.6393 作为 "DeCo 基线"（仅是策略移植）；
- 路线 B 64.71 的复核结论（本机未验证）。

---

## 5. 复现命令（fj01）

```bash
# 1) center=none 三份 dump（脚本为原 dump 去掉 center_torch 行）
bash /root/rivermind-data/repro/deco-b123/run_cnone_dumps.sh
# 产物: cnone_20261101/{cand,ev,segid}/<scene>_*.csv

# 2) gateE-TF（pred = NAG root score）
bash /root/rivermind-data/repro/deco-b123/run_gate_cnone_tf.sh
# 产物: cnone_20261101/gate/<scene>_cmaskgate_none.csv
```

本地聚合与分析：

```bash
python analysis/lerf_gate_trainingfree.py analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_gate_tf none
```

---

## 6. 归档

| 内容 | 路径 |
|---|---|
| 本次 gate 结果（4 场景 CSV） | `analysis/results/20261101/cnone_gate_tf/<scene>_cmaskgate_none.csv` |
| 对应 segidentity dump | `analysis/results/20261101/cnone_gate_tf/<scene>_segidentity.csv` |
| 离线 training-free 门控分析脚本 | `analysis/lerf_gate_trainingfree.py` |
| 服务器产物根目录 | `/root/rivermind-data/repro/deco-b123/cnone_20261101/` |
| 原 center=image 学习版结果 | `analysis/results/20261031/gatee/` |