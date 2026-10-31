# ScanNet 上移植 67.60（LERF）读出的结果：负迁移

日期：2026-11-01
目的：把 LERF-OVS 上最好结果（DeCo B0 gateE，0.6760）的可移植部分放到 ScanNet-10 上测试。
结论：**late interaction / 多视角 token 读出不能迁移到 ScanNet 19 类 per-root 分类。**

## 1. 协议差异（先说清）

- LERF-OVS：开放词汇 query → 候选池 → gap/gating；67.60 的核心是 **逐视角 mask token +
  late interaction + 同尺度证据 gating**。
- ScanNet-10：**19 类闭集 per-root argmax**；每个 root 有 19 类分数，GT 为每高斯类别；
  没有 LERF 的 query 候选选择，因此 **gateE/module C 无法直接应用**。
- 本次可移植的是：同一 root 的逐视角观测 token 上做 `top-k <f_o, t_class>` 的 late interaction，
  与 baseline（root 3D 特征 `x @ one_word.T`）比较。

## 2. 实验设置

- 数据：`scannet_v2_certified_20261029` 10 场景；`export/observations_level3.npz` 逐视角 token；
  GT 用最近 GT 点；模型 `/root/verify58_20261029/full10/<scene>/model/sai_nag_3.0.pt`。
- Baseline：`x = l2(nag_feat[-1])`，`scores = x @ one_word.T`（C1/C2 off，认证 DeCo off）。
- 变体：
  - `top1/top3/top5`：每个 root 的逐视角 token 对每个类别取 top-k 相似度均值；
  - `meanfeat`：可见性加权平均 token 后点积（对照，接近 baseline 的聚合方式）；
  - `top3_bg`：逐类别 z-score − 背景对比（其他 root token 的 max 相似度），再 top-3。
- 指标：scene macro mIoU（10 场景平均）与 pooled mIoU（跨场景按类汇总 inter/union）。

## 3. 结果

| 信号 | scene macro | pooled |
|---|---:|---:|
| **baseline（root 3D 特征）** | **40.36%** | **43.88%** |
| top1 late interaction | 19.54% | 28.22% |
| top3 late interaction | 22.85% | 31.94% |
| top5 late interaction | 25.58% | 33.04% |
| meanfeat（逐视角均值） | 40.15% | 42.83% |
| top3 + 背景对比 | 1.58% | 2.66% |

逐场景 baseline/top3（scene mIoU）：

| 场景 | baseline | top3 |
|---|---:|---:|
| scene0000_00 | 0.4629 | 0.3749 |
| scene0062_00 | 0.3687 | 0.2017 |
| scene0070_00 | 0.3914 | 0.1343 |
| scene0097_00 | 0.3974 | 0.2026 |
| scene0140_00 | 0.3580 | 0.1620 |
| scene0200_00 | 0.3585 | 0.3574 |
| scene0347_00 | 0.4831 | 0.1737 |
| scene0400_00 | 0.4568 | 0.0971 |
| scene0590_00 | 0.4255 | 0.3287 |
| scene0645_00 | 0.3334 | 0.2529 |

Baseline 与认证报告一致（pooled 43.88、macro 40.36），说明评测链正确。

## 4. 解释

1. **ScanNet 的 root 3D 特征已经是强而校准好的描述子**：闭集 19 类 argmax 依赖绝对分数刻度；
   逐视角 token 的 top-k 只保留“最像某类”的视角，去掉了跨类校准，导致 argmax 大量翻转。
2. **逐视角观测是区域混合特征**：对 LERF 的候选选择有用，但对 19 类闭集分类噪声太大；
   `meanfeat` 接近 baseline（−0.21 macro）说明“聚合方式”不是主要问题，**top-k 选择本身**才是。
3. **背景对比在 ScanNet 上不可用**：z-score 后减背景会把类间差异压掉（1.58%）。
4. **任务不同**：LERF 是“给定 query 选候选”，ScanNet 是“每 root 19 类分类”。
   67.60 的 gateE/module C 依赖 query 候选池，ScanNet 没有对应结构，不能直接搬。

## 5. 结论

- **67.60 的 LERF 读出（late interaction/gating）不迁移到 ScanNet**；ScanNet 上 baseline 仍最好。
- 若要在 ScanNet 上涨点，应改 **root 级特征/分类器/校准**，而不是逐视角 token 读出；
  这与 LERF 的问题结构不同。
- 不要把这个负迁移写成方法失败；它说明方法的作用域是 **open-vocabulary query selection**，
  不是闭集 per-root 分类。

## 6. 产物

- 脚本：`analysis/scannet_late_eval.py`（服务器 `deco/scannet/`）
- 结果：服务器 `/root/rivermind-data/repro/deco-b123/scannet_late_20261101/<scene>.json`
- 仓库副本：`analysis/results/20261101/scannet_late/`