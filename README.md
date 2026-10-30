# 3D-GS Open-Vocabulary Segmentation — DeCo / LERF-OVS + ScanNet 分析与方法

本仓库保存 2026-10 期间对 DeCo-3DGS 在 LERF-OVS 与 ScanNet 上的失败诊断、预注册实验与最终涨点模块。所有实验均遵循：测试集 GT 不参与训练/特征/选参；留一场景（LOSO）评估；场景分层 bootstrap 置信区间；预注册（PREREG）先冻结规则与门控，再跑实验。

## 核心结果

| 数据集 | 方法 | 指标 | 提升 |
|---|---|---|---|
| ScanNet（10 场景） | 学习式候选选择器（LOSO） | pooled mIoU 55.04 → 56.90；root 准确率 34.5% → 38.1% | +1.86 / +3.6pp |
| LERF-OVS（4 场景 208 查询） | **Prune-and-Include（预注册 PASS）** | 场景平均 IoU 0.6396 → 0.6590（95% CI [+0.0061, +0.0452]） | **+2.40** |

LERF 最终模块（`analysis/lerf_prune_include.py`）：
1. **学习式前缀剪枝**：LOSO IoU 回归器预测候选质量，删除 gap 前缀中预测 IoU < 0.5×max 的非 top-1 候选（top-1 永远保留）；仅在 max_pred ≥ 0.5 时启用。
2. **尺度门控挑战者并入**：top-1 面积 <0.1% 且存在 >2 倍面积候选时，并入尺度证据最高的挑战者。
3. 机理对应诊断：cut_early +0.173、cut_late +0.055、deep 族 +0.218，四场景全部提升（ramen +3.9、teatime +3.0、figurines +0.9、waldo +0.1）。

## 关键诊断（为什么提升空间有限）

- V2 分数修正上限仅 +1.96：修正量 ~1e-3，而排序分差 0.01–0.05（小 1–1.5 个数量级）；证据与主分数同源（主导模式余弦 >0.95）。
- 失败分族（LERF 208 查询）：59% 已正确、18% rank-2、9% 深层排序（池内正确候选 IoU 0.8–0.96）、14% 池内无正确候选（候选生成问题）。
- 决策层切换经济学：错误切换 −0.85 IoU/次 vs 正确救援 +0.47；零参数切换规则不存在正收益操作点。
- 候选生成阈值网格、池深、分布截断、身份证据（区域级/段级）均未通过预注册门控。

## 仓库结构

```
analysis/                     实验脚本与预注册文档（PREREG_*.md）
reports/                      中文实验报告（诊断、各轮实验、最终模块）
results/                      逐查询结果 CSV、审计输出、预注册产物（按实验分目录）
OLD_SESSION_*.md              （不纳入版本管理的会话记录）
```

关键脚本：
- `analysis/lerf_identity_dump.py`：候选级身份特征（多视角观测证据）
- `analysis/lerf_subset_selector.py`：LOSO IoU 回归器预测 → 集合选择
- `analysis/lerf_prune_include.py`：最终模块（剪枝 + 并入）选择与渲染
- `analysis/lerf_prune_include_eval.py`：预注册门控评估
- `analysis/lerf_pool_depth.py` / `lerf_poolgen.py` / `lerf_dist_trunc.py`：失败的对照实验

## 复现要点

- 服务器：fj01（RTX 4090）。仓库脚本运行于 DeCo-3DGS 的 `deco/lerf_ovs/`，数据依赖 LERF-OVS 场景与 `m1_sas_20261029` 导出（observations_level3.npz、sidecar）。
- 运行顺序：身份/证据特征导出 → LOSO 回归器预测（`lerf_subset_20261030/preds.csv`）→ 模块选择与渲染 → 预注册评估。
- 结果目录中的 `*_pruneinclude.csv` 与 `audit_summary` 为论文数字来源。

## 局限

- 回归器为监督组件（LOSO 逐折重训，仅用训练场景 GT 标签），论文须披露。
- LERF 仅 4 场景 208 查询，置信区间较宽；需多种子/更多场景确认。
- ScanNet 为单标签协议，LERF 多标签剪枝/并入无法直接移植；两套结果独立成立。