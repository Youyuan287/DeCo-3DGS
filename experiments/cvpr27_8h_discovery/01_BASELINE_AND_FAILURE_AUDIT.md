# 01 — 基线与失败审计

时间起点：2026-11-03T01:27:15+08:00（本地单调计时另行保留）。
证据标签：**[V]** 本会话由代码/输出复核；**[R]** 仅见历史报告，未独立复核；**[L]** 文献；**[H]** 假设。

## 1. 环境与产物定位

| 项 | 状态 |
|---|---|
| 仓库（本地） | `C:\Users\TYY\Documents\MyDocument\CVPR-2026\claude`（reports/analysis 为主体，不含 3DGS 训练代码） |
| GitHub | `Youyuan287/DeCo-3DGS`，67.60 已推送 `experiments/lerf_6760`（1aa0bf1）[R] |
| 服务器 fj01 | 在线；ScanNet 认证场 `/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029`、LERF 官方读权 `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf`、逐视角观测 `m1_sas_20261029/*_new/observations_level3.npz` [V] |
| 服务器 hn01 | 在线（30833），但只有 LERF figurines/ramen 观测，无 ScanNet 认证场 [V] |
| GPU | fj01 4090 24G，实验时基本空闲 [V] |

**任务书中点名的 7 份文档全部缺失**：`DIAGNOSIS_LEDGER.md`、`DIAGNOSIS_WHY_MASKS_FAIL.md`、
`FINDINGS_DIFFICULT_SAMPLES.md`、`FINDINGS_E12_SCORE.md`、`FINDINGS_MASK_COS.md`、
`FINDINGS_SUBDIV.md`、`FINDINGS_VALA_XCHECK.md` 在仓库内均不存在 [V]。
等价材料散落在 `reports/` 的 40+ 份报告中（如 `LERF_HARD_QUERY_REPORT.md`、
`WHY_CONSISTENCY_HURTS.md`、`RELAGS_ANALYSIS_AND_HEADROOM.md`、`SCANNET_STEP0_ATTRIBUTION_REPORT.md`）。
→ **结论：不能声称已读取"账本原文"；本审计只基于可定位的报告 + 本次复核。**

## 2. 权威基线（必须使用）

### LERF-OVS（官方口径：topk5 / level_until3 / filter_small50；per-image → per-scene → macro；208 查询）

| 配置 | macro mIoU | 状态 |
|---|---:|---|
| ReLaGS released（center=none） | **0.6401** | [V] 本会话重跑，逐位一致 |
| 判决审计（配对证书 + 范围支配，training-free） | **0.6711** | [R] 历史输出，本轮未重跑 |
| 确定性门控变体 | 0.6716 | [R] |
| LOSO-GBM module C | 0.6760 | [R] **含学习组件，禁止作为 training-free 基线** |
| VCLS（逐视角中心化） | 0.6401 − 0.002 | [V] 本轮重跑，约 −0.2 pp |

### ScanNet-10（认证协议：19 类 / 逐 root argmax / present-class macro / 10 场景）

| 配置 | mIoU | 状态 |
|---|---:|---|
| C1/C2 off（ReLaGS 场） | **40.36** | [V] |
| + 逐类 z-score（zR） | **45.98** | [V] |
| zR + 原始多视角证据 E0 | **48.77** | [V] 本轮复现，与历史一致 |
| zR + 逐视角中心化 Ev | 49.16 | [V] |
| C1+C2 on（DeCo 场） | 50.18 | [R] |

## 3. 已否证 / 已收口的路线

| 路线 | 结果 | 证据 |
|---|---|---|
| 判决审计迁移到 ScanNet | 0.4036→0.3673；0.5018→0.4591；同 query 重构 0.4926 | [R][V 部分] |
| SAM 独立证人 | 保守证书 0 触发；激进 −14.6 | [R] |
| 3D-OVS 复现/审计 | 未达文献水平；level-1 天花板 0.7735 | [R] |
| 视角中心化（VCLS）作为方法 | 独立增益 +0.39，CI [−0.03,+0.82] | [V] |
| 局部对比 B0–B6（两数据集） | 全部低于无对比基线；随机参考 ≈ 空间邻域 | [V] |

**局部对比的细节（本轮新增，重要）**：ScanNet 上 Erand −2.96 / Eknn −2.89 / Ehard −2.58 /
Epair −11.84（均相对 zR+E0 = 48.77）；LERF 上 E0 −0.32 / Erand −0.88 / Eknn −3.72（显著）/ Epair −0.51。
随机参考与空间邻域同量级 ⇒ 损害来自"减掉参考均值"本身，而非局部性 [V]。

## 4. 协议污染清单（必须持续检查）

1. **LOSO-GBM**：用真实 IoU 训练的选择器。任何 0.6760 派生数字都不是 training-free [R]。
2. **四场景设计污染**：LERF 四个场景都曾参与方法设计，**不能称为独立测试集**；只能称"复现场景上的评测" [R]。
3. **超参来源**：α=0.5 / β=0.5 / topk=3 / γ=1 均在本数据上选定过，属于"验证集选择"，必须在论文中披露。
4. **无观测填零**：`evidence()` 对无观测 root 填 0 会显著改变 z-score；已统一为"全 root 标准化 + 零填"以复现 48.77 [V]。
5. **null 的几何解释**：任何"根级"结果都受"root 是粗区域"影响（ScanNet root 380 个，跨物体）[V]。
6. **投影 R/Rᵀ、RGB/BGR**：历史曾出错；本轮未使用新的投影代码，沿用已认证产物 [R]。
7. **Oracle**：0.7169 是单候选 oracle，不是集合推断上限；集合 oracle 需按子集枚举 [R]。

## 5. 本阶段结论

- 两个数据集各有一条**强基线**且不可混用：LERF `released 0.6401`、ScanNet `zR+E0 48.77`。
- 两个数据集上**迄今唯一稳健的正结果分别是不同层的**：LERF 靠"决策层证书"（+3.13），
  ScanNet 靠"分数层逐类标准化"（+5.62）。**这正是需要统一的地方，也是本案的入口。**
- 所有"再加一层分数变换"的尝试（中心化、局部对比、证据变体）在两个数据集上都没有稳定收益，
  说明问题不在"变换的形式"，而在**分数里到底混进了什么**。