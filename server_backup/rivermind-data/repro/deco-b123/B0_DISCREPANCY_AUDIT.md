# ramen B0 结果差异审计

## 结论

此前 B123 报告中的 `B0 mIoU=0.457138` 与历史记录的 `0.5180756` 不可直接比较，因为两者使用了不同的 NAG 输入。

| 评测 | NAG | NAG SHA256 | mIoU |
|---|---|---|---:|
| 历史官方 DeCo/ReLaGS 基准 | `sai_nag.pt` | `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324` | `0.5180756`（DeCo 记录） |
| 历史 ReLaGS evaluator 复核 | `sai_nag.pt` | 同上 | `0.5114938` |
| 本次 B123 冻结 NAG B0 | `sai_nag_3.0.pt` | `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535` | `0.4571384` |

## 已完成的最小诊断

使用历史 `eval_lerf_relags.py`，固定当前 ramen 图像、相机、Gaussian 模型和 GT，仅把 NAG 换成官方 `sai_nag.pt`：

```bash
export RELAGS_REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
export PYTHONPATH=$RELAGS_REPO:$RELAGS_REPO/ext

/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3 \
  /root/rivermind-data/repro/deco-3dgs-38620a7d/code/reproduction/scripts/eval_lerf_relags.py \
  --scene ramen \
  --source /root/rivermind-data/repro/deco-b123/frozen_ramen_source \
  --model /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen \
  --nag /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag.pt \
  --gt_root /root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label \
  --out /root/rivermind-data/repro/deco-b123/diagnostic_official_nag \
  --csv /root/rivermind-data/repro/deco-b123/diagnostic_official_nag/per_query.csv \
  --topk 5 --level_until 3 --filter_small 50
```

诊断输出：

```text
SCENE_RESULT ramen mIoU=0.5115 flat=0.5046 queries=71 images=7
```

## 输入一致性排除

- VALA ramen 图像与 HF ramen 图像：`131/131` 文件 MD5 完全一致。
- VALA 与 HF 的 `sparse/0/cameras.bin`：MD5 一致。
- VALA 与 HF 的 `sparse/0/images.bin`：MD5 一致。
- 当前 GT 仍为 `/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label/ramen`。

因此当前已确认的主要差异是 NAG 文件，而不是图像、相机或 GT。

## 正确表述

本次 B123 的 B0 应标为：

> 冻结 `sai_nag_3.0.pt` 条件下的 controlled B0

不能标为：

> 官方 `sai_nag.pt` 条件下的原始 DeCo B0

历史 `0.5180756` 只能作为官方 NAG 基准参考，不能直接作为本次冻结 NAG B123 的同条件对照。
