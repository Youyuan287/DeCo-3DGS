# 冻结 ramen NAG 与 B123 实验报告

## 1. 实验目标

在冻结 NAG、Gaussian 模型、131 个训练视角和已有二维 CLIP 特征不变的条件下，重新导出冻结 NAG 对应的生产观测，并基于同一份观测运行 B0、K0、B1、B2、B3 五组实验。未执行重新训练、`image_encoding.py`、NAG 聚类或历史观测恢复。

## 2. 输入与一致性

| 项目 | 路径 | SHA256/数值 |
|---|---|---|
| 冻结 NAG | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt` | `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535` |
| Gaussian iteration-0 PLY | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/point_cloud/iteration_0/point_cloud.ply` | `e7d2fb7a725b882efe3c8f4cc9f0f64fc5cbe5a36866f0d8bee353fbd9b68276` |
| `nag-l1.pt` | `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/nag-l1.pt` | `878b36dbace822d27197e6371d11361380d973d9182460de8bafc2c383c01288` |

- Gaussian 数与四层 NAG 标签长度均为 `372270`。
- `nag[0]` 与 `nag-l1.pt` 逐元素一致。
- NAG 区域数（level 0/1/2/3）：`7936 / 1860 / 490 / 245`。
- 训练视角 `131` 个；二维特征文件 `262` 个，即每个视角一对 `_f.npy`/`_s.npy`。
- 生产目录未修改；隔离 DeCo commit 为 `fbbb7f9`。

## 3. 冻结观测导出

输出：

```text
/root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz
```

观测 SHA256：

```text
3c857c05d101029fd332fbdbb4e499b4d0c8d1b67a643121a9b7fdb7d32a7fc6
```

| 项目 | 数值 |
|---|---:|
| 观测条数 | 8486 |
| 唯一区域数 | 244 |
| 视角数 | 131 |
| 特征维度 | 512 |
| member CSR ID 数 | 33421977 |
| visible CSR ID 数 | 14674402 |

校验结果：features 为 `(8486, 512)` 且全部有限；权重全部有限且为正，最小值为 `2.4933925e-05`；两套 offsets 长度均为 `8487`，尾 offset 与 ID 数量一致；region ID 和 Gaussian ID 均在合法范围内。

本文件是基于当前冻结 NAG 的新观测导出，不是历史 304 区域观测恢复，也没有与旧 bank 混用。

## 4. 统一评测协议

```text
scene=ramen, center=image, text=canon, renorm=0, sel=gap
topk=5, level_until=3, filter_small=50, temperature=0.08
```

- B0：原始完整 DeCo。
- K0：同一观测的单证据聚合。
- B1：去偏语义 + 几何支持联合分组。
- B2：B1 + 模式内 ROFA。
- B3：直接加载 B2 bank，仅开启 query-conditioned weighting。

## 5. 总体结果

| 组别 | mIoU | flat mIoU | 查询数 | 平均候选数 | 平均预测像素 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.457138 | 0.449149 | 71 | 2.070 | 23057.3 |
| K0 | 0.457138 | 0.449149 | 71 | 2.141 | 23057.3 |
| B1 | 0.457135 | 0.449144 | 71 | 1.789 | 14998.9 |
| B2 | 0.457135 | 0.449144 | 71 | 1.789 | 14998.9 |
| B3 | 0.437974 | 0.426904 | 71 | 2.042 | 21903.7 |

## 6. 配对消融

| 配对 | 平均 IoU 差值 | 最小 | 最大 | 正向查询数 | 负向查询数 |
|---|---:|---:|---:|---:|---:|
| K0 − B0 | 0.00000000 | 0.00000000 | 0.00000000 | 0 | 0 |
| B1 − K0 | −0.00000510 | −0.00036200 | 0.00000000 | 0 | 1 |
| B2 − B1 | 0.00000000 | 0.00000000 | 0.00000000 | 0 | 0 |
| B3 − B2 | −0.02224021 | −0.56645600 | 0.05264500 | 6 | 16 |

## 7. ROFA 与 bank 审计

| Bank | geometry 生效 | 删除观测数 |
|---|---|---:|
| K0 | false | 0 |
| B1 | true | 0 |
| B2 | true | 0 |

B2 实际删除观测数为 `0`，因此没有调整 ROFA 阈值。B3 直接加载 B2.npz；B2 bank SHA256 为：

```text
7978967c6709c7c424b7fb9568be47014ea32032837c43224994184cbe793a37
```

## 8. 复现命令

### 观测导出

```bash
export PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
export SRC=/root/rivermind-data/repro/deco-b123/frozen_ramen_source
export MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
export NAG=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt
export OBS=/root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz

HF_HUB_OFFLINE=1 $PY \
  /root/rivermind-data/repro/deco-b123/export_frozen_ramen_observations.py \
  --source "$SRC" --model "$MODEL" --nag "$NAG" --output "$OBS" \
  --iteration 0 --seed 1234
```

### B0/K0/B1/B2/B3

```bash
export DECO_ROOT=/root/rivermind-data/repro/deco-b123/DeCo-3DGS
export RELAGS_REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
export NAG=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt
export OBS=/root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz
export SRC=/root/rivermind-data/repro/deco-b123/frozen_ramen_source
export MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
export GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
export OUT=/root/rivermind-data/repro/deco-b123/b123_eval_20261028
export PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
export DECO_CENTER=image DECO_TEXT=canon DECO_RENORM=0 DECO_SEL=gap
export GEO_WEIGHT=0.25 TOPK=5 MAX_MODES=3 TEMP=0.08

bash /root/rivermind-data/repro/deco-b123/DeCo_B123_Patch/run_b123.sh
```

## 9. 结果位置

```text
/root/rivermind-data/repro/deco-b123/FROZEN_NAG_OBSERVATION_EXPORT.md
/root/rivermind-data/repro/deco-b123/B123_RESULTS.md
/root/rivermind-data/repro/deco-b123/b123_eval_20261028/B0/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_20261028/K0/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_20261028/B1/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_20261028/B2/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_20261028/B3/per_query.csv
```

## 10. 结论

B0 与 K0 完全一致；B1 相比 K0 没有实际提升；B2 没有删除观测，因此 B2 与 B1 完全一致；B3 的 query-conditioned weighting 使 mIoU 下降约 `0.02224`。几何支持在 B1/B2 中实际生效。所有结果使用同一冻结 NAG、同一观测文件和统一评测协议，未混用历史观测或旧 bank，也未进行调参。

## 11. 与历史 0.5181 基准的差异说明

本报告中的 B0 是“冻结 `sai_nag_3.0.pt` 条件下的 controlled B0”，不是官方 `sai_nag.pt` 条件下的历史原始 DeCo B0。两者 NAG SHA256 不同：官方 `sai_nag.pt` 为 `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324`，冻结 `sai_nag_3.0.pt` 为 `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535`。

使用历史 ReLaGS evaluator、相同 ramen 图像/相机/GT，仅换回官方 `sai_nag.pt`，得到 `mIoU=0.5114938`；图像 131/131、相机和 GT 输入均与当前一致。因此 `0.457138` 与历史 `0.5180756` 的主要差异来自 NAG 版本，不能把两者当作同 NAG 的性能差分。详细审计见 `B0_DISCREPANCY_AUDIT.md`。
