# ramen 官方 NAG B123 重跑实验报告

## 结论先行

此前基于 `sai_nag_3.0.pt` 的整批 B123 结果作废。本报告使用历史官方 `sai_nag.pt`，并重新导出观测、重新构建 K0/B1/B2 bank、重新运行 B0/K0/B1/B2/B3。

官方 NAG 条件下 B0 为 **mIoU 0.518076**，与历史记录的 `0.5181` 一致。

## 固定输入

```text
NAG: /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag.pt
NAG SHA256: 10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324
Gaussian model: /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
Source: /root/rivermind-data/repro/deco-b123/frozen_ramen_source
GT: /root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
```

评测协议：`center=image`、`text=canon`、`renorm=0`、`sel=gap`、`topk=5`、`level_until=3`、`filter_small=50`、`temperature=0.08`，共 71 个 query。

## 重新导出的观测

```text
/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz
SHA256: 6ae30f6e03b01177b14e900a447d71ee788ef57a31cac9909b0abf0d33358232
```

- 观测条数：`8470`
- level-3 区域：`248/249` 个区域被观测到
- 视角：`131`
- 特征：`(8470, 512)`
- member CSR ID 数：`34766011`
- visible CSR ID 数：`14672473`
- weights 全部有限且为正
- CSR offsets 长度为 `8471`，尾部与 ID 数一致
- Gaussian 数：`372270`

## 五组结果

| 组别 | mIoU | flat mIoU | query 数 | 平均候选数 | 平均预测像素 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.518076 | 0.511650 | 71 | 2.113 | 20602.0 |
| K0 | 0.494300 | 0.483218 | 71 | 1.746 | 24459.4 |
| B1 | 0.494300 | 0.483218 | 71 | 1.676 | 24459.4 |
| B2 | 0.494300 | 0.483218 | 71 | 1.676 | 24459.4 |
| B3 | 0.492156 | 0.481220 | 71 | 2.465 | 24693.4 |

## 配对消融

| 配对 | 平均 IoU 差值 | 最小 | 最大 | 正向 query | 负向 query |
|---|---:|---:|---:|---:|---:|
| K0 − B0 | −0.02843215 | −0.92936700 | 0.23062600 | 11 | 7 |
| B1 − K0 | 0.00000000 | 0.00000000 | 0.00000000 | 0 | 0 |
| B2 − B1 | 0.00000000 | 0.00000000 | 0.00000000 | 0 | 0 |
| B3 − B2 | −0.00199866 | −0.09013900 | 0.04880700 | 3 | 9 |

## Bank/ROFA 审计

| Bank | geometry 生效 | 删除观测数 | SHA256 |
|---|---|---:|---|
| K0 | false | 0 | `844c4b04aa2494d64baab9b67ca2fdaa438c8e84cad6bbcfa76c74847b32a8c1` |
| B1 | true | 0 | `73e4511df01725fb7b6e6ed235971c509434aa6603d1aaf233eba5cdd6f54aa1` |
| B2 | true | 0 | `68f1b7ad3eee33670539caba1d49c853cc6bbef587f3bf4e4b0ec80235bdc81e` |

B3 直接加载 B2 bank，B2/B3 bank SHA256 完全一致。B2 实际 ROFA 删除数为 `0`，没有调整阈值。

## 复现命令

### 观测导出

```bash
export PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
export SRC=/root/rivermind-data/repro/deco-b123/frozen_ramen_source
export MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
export NAG=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag.pt
export OBS=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz

HF_HUB_OFFLINE=1 $PY \
  /root/rivermind-data/repro/deco-b123/export_frozen_ramen_observations.py \
  --source "$SRC" --model "$MODEL" --nag "$NAG" --output "$OBS" \
  --iteration 0 --seed 1234
```

### B0/K0/B1/B2/B3

```bash
export DECO_ROOT=/root/rivermind-data/repro/deco-b123/DeCo-3DGS
export RELAGS_REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
export NAG=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag.pt
export OBS=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz
export SRC=/root/rivermind-data/repro/deco-b123/frozen_ramen_source
export MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
export GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
export OUT=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028
export PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
export DECO_CENTER=image DECO_TEXT=canon DECO_RENORM=0 DECO_SEL=gap
export GEO_WEIGHT=0.25 TOPK=5 MAX_MODES=3 TEMP=0.08

bash /root/rivermind-data/repro/deco-b123/DeCo_B123_Patch/run_b123.sh
```

## 结果文件

```text
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B123_RESULTS_OFFICIAL.md
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B0/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/K0/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B2/per_query.csv
/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B3/per_query.csv
```

旧的 `b123_eval_20261028` 目录仅作为作废批次保留，不应引用。
