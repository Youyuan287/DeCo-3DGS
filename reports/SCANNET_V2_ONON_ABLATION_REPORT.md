# ScanNet-10 DeCo V2 消融（DeCo 主链：C1=on, C2=on）

上一版 `SCANNET_V2_CERTIFIED_ABLATION_REPORT.md` 的五组全部跑在 `SCANNET_C1=off SCANNET_C2=off` 上，"DeCo off" 实际是 43.88 的 base 臂，不是 55.04 的 DeCo 臂。本次在相同认证输入上改为 C1（均值减除）与 C2（both_lse 文本库）全开重跑，其余不变。

## 运行信息

- 服务器：fj01（RTX 4090），2026-10-30 03:17–03:24 UTC
- 驱动：`/root/rivermind-data/repro/deco-b123/run_certified_v2_eval_onon.sh`（与 `run_certified_v2_eval.sh` 只差 C1/C2=on 和输出目录；不重写 EXPORT_MANIFEST）
- 评测脚本：`scannet_v2_certified_20261029/scannet_c1c2_m1m2m3_evaluate_certified.py`（未修改）
- 输出：`/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029/ablation/v2_on_on/<scene>/`
- 日志：`.../scannet_v2_certified_20261029/batch_eval_onon.log`
- 汇总脚本：`/root/rivermind-data/repro/deco-b123/analyze_v2_ablation.py v2_on_on`
- 固定参数：K19、renorm=0、temperature=0.08、coefficient=0.25、max_modes=3、geo_weight=0.25；observation、B1 bank、NAG、Gaussian、GT 与 off/off 版相同（EXPORT_MANIFEST SHA 校验通过）

## Baseline parity

十个场景全部打印 `BASELINE_PARITY_PASS`：off 组与未修改的已发布评分器逐类指标完全相等（代码内 `assert offmetric==reference`），且与 `verify_expected.py` 的 EXP_DECO 逐场一致。汇总 off = pooled 55.0405 / macro 50.1783，对应认证 DeCo 55.04 / 50.18。

## 汇总结果

| 组别 | pooled mIoU | scene macro | pooled 相对 off | macro 相对 off | 场景 胜/负/平 |
|---|---:|---:|---:|---:|:---:|
| DeCo (C1+C2), 无 V2 | 55.0405 | 50.1783 | — | — | — |
| + M1 Fixed | 55.1866 | 50.2550 | +0.146 | +0.077 | 6/3/1 |
| + M1+M2 Fixed | 55.1744 | 50.2957 | +0.134 | +0.117 | 7/2/1 |
| + M1+M3 Adaptive | 55.2824 | 50.0059 | +0.242 | **−0.172** | 5/4/1 |
| + M1+M2+M3 Adaptive | 55.1960 | 50.2997 | +0.156 | +0.121 | 8/1/1 |

## 逐场 scene mIoU

| 场景 | off | M1 Fixed | M1+M2 Fixed | M1+M3 Adaptive | M1+M2+M3 Adaptive |
|---|---:|---:|---:|---:|---:|
| scene0000_00 | 51.2800 | 51.1771 | 51.0238 | 49.5718 | 51.3030 |
| scene0062_00 | 82.5013 | 82.4098 | 82.4098 | 82.5213 | 82.5213 |
| scene0070_00 | 35.7754 | 35.9025 | 35.9075 | 34.8369 | 34.8419 |
| scene0097_00 | 43.3788 | 43.4211 | 43.4211 | 43.4211 | 43.4211 |
| scene0140_00 | 51.9783 | 51.4471 | 52.8866 | 51.4083 | 52.9309 |
| scene0200_00 | 49.7671 | 49.8876 | 49.8876 | 49.8876 | 49.8876 |
| scene0347_00 | 43.4688 | 43.5580 | 43.5591 | 43.3305 | 44.0437 |
| scene0400_00 | 44.5428 | 44.5428 | 44.5428 | 44.5428 | 44.5428 |
| scene0590_00 | 52.1750 | 53.0365 | 52.3985 | 52.9998 | 52.3880 |
| scene0645_00 | 46.9160 | 47.1671 | 46.9202 | 47.5392 | 47.1164 |

## 修正幅度诊断

| 量 | off/off 版 | on/on 版（本次） |
|---|---:|---:|
| 改变预测的 root（共 1951） | 13–27 | 52–119 |
| 受影响 Gaussian（共约 672 万） | 1.3–1.9 万 | 3.7–13.3 万 |
| root top1−top2 margin 中位数 | 0.0187 | 0.0128 |
| \|correction\| 中位数 / p99 / max（Adaptive） | 0.0002 / 0.0066 / 0.019 | 0.0012 / 0.014 / 0.059 |
| M2 reliability p10/p50/p90 | 0.74 / 0.81 / 0.89 | 同左 |

C1/C2 后分数更集中、修正更大，V2 能实际改变 3–6% 的 root，比 off/off 版有效得多，但仍属小幅扰动。

## 结论

1. 在真正的 DeCo 主链上，V2 四组 pooled 均为正（+0.13 ~ +0.24），但幅度都在 0.25 个百分点以内。
2. **M1+M3 Adaptive 不可取**：pooled 最高，但 macro 为负（−0.17），因 scene0000（−1.71）和 scene0070（−0.94）明显下降；pooled 增益来自少数大类。
3. **M1+M2+M3 Adaptive 最稳**：pooled +0.16、macro +0.12、8 胜 1 负 1 平。但逐场差值均值 +0.12、主要由 scene0140（+0.95）与 scene0347（+0.58）贡献，并被 scene0070（−0.93）抵消；10 个场景的配对 t 检验远不显著（t≈0.80，差值标准差 0.48），符号检验 8/9 处于边缘。
4. **M2 在 Adaptive 路径上有可见作用**：把 M1+M3 的 macro 从 −0.17 拉回 +0.12（scene0000 +1.73、scene0140 +1.52），起到抑制错误修正的作用；在 Fixed 路径上作用很小且方向不一（scene0140 +1.44、scene0590 −0.64）。
5. 可以写成"V2 在 DeCo 上带来一致但很小的 pooled 增益，M2 稳定了自适应权重"，**不能**写成显著提升；若要支撑更强结论，需要更多场景或多 seed。
6. 上一版 off/off 表格应改名为"base 臂上的 V2"，不应以"DeCo off"出现在论文中。

## 裕量分析（GT oracle，2026-10-30）

脚本：`/root/rivermind-data/repro/deco-b123/headroom_oracles.py`（DeCo C1+C2 臂）。每个 oracle 对每组高斯，在给定候选里选与 GT 多数类相同的标签。

| oracle | 含义 | pooled | macro | 相对 DeCo pooled |
|---|---|---:|---:|---:|
| DeCo（实际） | — | 55.04 | 50.18 | — |
| V2 实际最好 | M1+M3 Adaptive / M1+M2+M3 | 55.28 / 55.20 | 50.01 / 50.30 | +0.24 / +0.16 |
| root_modes | root 在 {主 top-1, 各 eligible mode argmax} 中选对：V2 root 级读出的上限 | 59.61 | 54.80 | +4.57 |
| mode_split | 按 mode 足迹拆 root，每块在 {root 标签, 该 mode argmax} 中选对 | 56.65 | 51.99 | +1.61 |
| root_top2 | root 在主分数 top-1/top-2 中选对 | 64.31 | 60.58 | +9.27 |
| root_any | root 取 GT 多数类（分区上限） | 81.02 | 75.09 | +25.98 |
| mode_split_any | mode 足迹块取 GT 多数类 | 82.22 | 76.10 | +27.18 |

- 分区不是瓶颈：root 级上限 81.0，比 DeCo 高 26 个百分点；按 mode 再拆只多 1.2，混合 root 问题很小。
- 瓶颈是分类：主分数的 top-2 已含正确类的 root 能带来 +9.3；mode 候选只能带来 +4.6，且 V2 实际只拿到其中约 3–5%。
- mode 级拆分上限只有 +1.6，不值得做。

LERF-OVS 现有 V2 结果（`m1_sas_m2m3_20261029`，参数同 ScanNet）：186 个查询里只有 4 个改善、3 个变差；figurines 58.04→58.79，teatime 不变，ramen 51.17→51.14。与 ScanNet 一致，V2 在 LERF 上也只是在打平时起作用。### LERF-OVS 候选 oracle（2026-10-30）

脚本：`c1_b123_run_20261028/DeCo-3DGS/deco/lerf_ovs/lerf_oracle.py`，驱动 `run_lerf_oracle.sh`，输出 `/root/rivermind-data/repro/deco-b123/lerf_oracle_20261030/`。DeCo 主链（center=image、canon、renorm=0、gap、top-5）。main 与 V2 c=0.25 与已有 per_query.csv 逐查询完全一致（208 个查询，差异 0）。

| oracle（四场景 flat mIoU 平均） | mIoU | 相对 DeCo |
|---|---:|---:|
| DeCo main（gap 选择） | 63.96 | — |
| V2 c=0.25（现行） | 64.15 | +0.19 |
| V2 全局最佳系数 | 64.15（c=0.25） | +0.19 |
| V2 每查询最佳系数（c∈{0,.25,.5,1,2,4}） | 65.92 | +1.96 |
| 最佳前缀（最优截断，替代 gap 规则） | 71.23 | +7.27 |
| 最佳单候选 | 70.94 | +6.98 |
| top-5 池内贪心子集 | 75.12 | +11.16 |
| top-20 池内贪心子集 | 77.91 | +13.95 |

| 场景 | main | V2 c=.25 | 最佳前缀 | 贪心 top-5 | 贪心 top-20 |
|---|---:|---:|---:|---:|---:|
| figurines | 58.04 | 58.79 | 69.40 | 77.00 | 79.06 |
| ramen | 51.17 | 51.14 | 60.35 | 68.14 | 68.99 |
| teatime | 78.52 | 78.52 | 82.66 | 82.66 | 85.56 |
| waldo_kitchen | 68.12 | 68.12 | 72.52 | 72.70 | 78.03 |

- V2 这一族的上限约 +2，而且系数 ≥1 时结果崩溃（c=4 只剩 7.07）：修正会抬高大量 root，破坏 gap 选择器依赖的分数落差。
- 裕量主要在选择：只改截断规则就有 +7.3；在现有 top-5 候选里选对子集有 +11.2；扩大候选池再多 +2.8。
- 208 个查询中 29 个 IoU=0，其中 20 个在 top-5 池里根本没有正确候选（top-20 池仍有 13 个），属于召回失败，任何选择规则都救不了。

## 未覆盖的风险

- 认证 DeCo 链本身状态为 UNGATED；评分器对每场打印 6 种文本变体的 GT 分数，both_lse 是否在测试集上挑选出来的需单独确认。
- temperature=0.08、coefficient=0.25 未扫描（按设计）；本次结论只对这一组参数成立。
- 仅 seed=1234 一个种子。
