# waldo_kitchen 冻结 NAG B123 复现实验报告

## 结论

在同一冻结 NAG、同一批 12,772 条观测、同一 source/model/GT 和同一评测协议下，五组实验均已完成。原始 DeCo B0 的 mIoU 为 **0.4664**；K0 为 **0.4491**；B1 为 **0.4884**；B2 为 **0.5064**；B3 为 **0.4408**。每组均为 22 个 query、5 张图。

**历史口径纠正：** 用户确认历史 DeCo waldo_kitchen 基准为 **65.18%**。本次结果不能作为该基准的复现：本次使用的是隔离 B123 evaluator `eval_lerf_ovs_b123.py`、冻结 `sai_nag_3.0.pt` 和 22-query 新协议；历史主链路使用 `scripts/eval_seg.py` 及其原始 query/聚合方式。此前报告将当前 0.4664 与历史 DeCo 直接比较，已作废。历史表中还存在 65.90%/65.92% 的四场景/其他指标记录，不能混用。

因此，本文件中的 B0/K0/B1/B2/B3 数字仅是“当前冻结 NAG B123 协议诊断”，不是历史 DeCo waldo_kitchen 复现结果；在恢复 `scripts/eval_seg.py`、历史输入和 65.18% 指标聚合前，不应据此报告方法增益。

## 官方 DeCo 输入复核（新增）

使用完整仓库的官方 `deco/lerf_ovs/eval_lerf_ovs.py`、`ref/relags_hf/waldo_kitchen/sai_nag.pt`、官方模型和相同 `topk=5 / level_until=3 / filter_small=50` 协议重新运行，得到：

```text
SCENE_RESULT waldo_kitchen mIoU=0.6062 flat=0.6137 queries=22 images=5
```

该结果与 DeCo 仓库自带的 `reproduction/results/lerf_scene_metrics.csv` 中 waldo_kitchen **60.62** 完全一致。由此确认先前 46.64% 的原因是误用了冻结 `sai_nag_3.0.pt`，而不是官方 DeCo B0 输入。旧的 46.64% B123 对照不再用于历史 DeCo 结论。

| 组别 | mIoU | flat mIoU | 相对上一组 |
|---|---:|---:|---:|
| B0 | 0.4664 | 0.4629 | — |
| K0 | 0.4491 | 0.4472 | -0.0156 vs B0 |
| B1 | 0.4884 | 0.4954 | +0.0482 vs K0 |
| B2 | 0.5064 | 0.5200 | +0.0246 vs B1 |
| B3 | 0.4408 | 0.4444 | -0.0756 vs B2 |

逐 query 原始 CSV 位于 [waldo_b123_results](./waldo_b123_results/)。配对差值按 `(image, prompt)` 对齐后计算。B2 实际删除观测数为 **0**，因此 B2 的 ROFA 净作用来自模式内评分而非删观测；没有调大阈值。

## 固定输入与一致性

- 场景：`waldo_kitchen`；187 个训练视角，374 个二维特征文件。
- Gaussian：905,012 个点；冻结 NAG 四层标签均为 905,012，根区域数为 385。
- NAG：`sai_nag_3.0.pt`，SHA-256 `76c48aca3f384a2fc28909c6f23d1d04df98d4bd0f56c1eea1158426761d8083`。
- 新观测：12,772 条，381 个唯一区域；member CSR 尾 offset=68,225,036，visible CSR 尾 offset=23,506,297；特征和正权重均有限。
- 观测 SHA-256：`6fc6e6e4d7c812ac26e5961cb2945c9ae15145bf7d01d6c118c758885876263d`。
- K0/B1/B2 均绑定上述观测和 NAG；B3 直接复用 B2 bank。B2/B3 bank SHA 均为 `b7c8eec04e4623819e9efd815b765b88a08d085a1ee0a4dd0b0c9cda84ec1c36`。
- K0/B1/B2 bank SHA：`530a00f04ad18a9d917888e4cbbe23f2e22f3046ffd5a986b7f987743666f677`、`580dc86adecca773a7452125b45f9c2ac27aeb539bb5e63e4f082fa6f23969a8`、`b7c8eec04e4623819e9efd815b765b88a08d085a1ee0a4dd0b0c9cda84ec1c36`。

## 运行协议

```bash
--topk 5 --level_until 3 --filter_small 50
--center image --text canon --renorm 0 --sel gap
--evidence-temperature 0.08
```

B0 使用 `--b1 0 --b2 0 --b3 0`；K0 使用 `--control-k1 1` 和 K0 bank；B1 使用 `--b1 1` 和 B1 bank；B2 使用 `--b1 1 --b2 1` 和 B2 bank；B3 使用 `--b1 1 --b2 1 --b3 1`，仍加载 B2 bank。

## 结果解释

B1 相对 K0 提升 4.82 个百分点，说明去偏语义与几何支持联合分组在该场景有效；B2 再提升 2.46 个百分点，但由于删除数为 0，应解释为模式内重评分作用。B3 比 B2 下降 7.56 个百分点，说明本次固定 query-conditioned weighting 在 waldo_kitchen 上没有带来正收益，不能把它写成普遍改进。

服务器原始目录：`/root/rivermind-data/repro/deco-b123/waldo_b123_20261028/`。
