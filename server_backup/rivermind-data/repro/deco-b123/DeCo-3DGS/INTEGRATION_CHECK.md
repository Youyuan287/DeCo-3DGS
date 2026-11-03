# DeCo-3DGS B123 集成检查

日期：2026-10-28

## 基线与隔离

- 上游来源：`https://github.com/Youyuan287/DeCo-3DGS`
- 可复用代码归档提交：`38620a7d84ff21db28ab19a2e067cf663b00698a`
- 服务器生产仓库 `/root/rivermind-data/ReLaGS-1023` 当前 HEAD：`65dfba925abbd45cb68ddc8cc6959669e821f961`
- 生产仓库状态：干净；未修改。
- 独立实验副本：`/root/rivermind-data/repro/deco-b123/DeCo-3DGS`
- 独立分支：`b123-integration`
- 安装生成的基线快照提交：`bf80da352f300363ebdb040bbcfa417b47db49b7`

## 补丁安装

使用补丁自带命令：

```bash
PATH=/root/rivermind-data/ReLaGS_Reproduction/venv/bin:$PATH \
  bash /root/rivermind-data/repro/deco-b123/DeCo_B123_Patch/install.sh \
  /root/rivermind-data/repro/deco-b123/DeCo-3DGS
```

安装成功生成：

- `deco/lerf_ovs/evidence_modes.py`
- `deco/lerf_ovs/evidence_readout.py`
- `deco/lerf_ovs/build_bank_b123.py`
- `deco/lerf_ovs/eval_lerf_ovs_b123.py`

原始 `deco/lerf_ovs/eval_lerf_ovs.py`、`deco/deco_ops.py`、`nag_data.py`、`merge_proj.py` 未覆盖。

## 正确性检查

CPU 单元测试：通过。

```text
Ran 4 tests in 0.065s
OK
```

已通过的测试：

- `test_b1_b2_construction_and_provenance`
- `test_b3_switch_only_scoring`
- `test_rofa_single_and_majority`
- `test_root_only`

Python 语法检查：通过：

```text
deco/lerf_ovs/eval_lerf_ovs_b123.py
deco/lerf_ovs/evidence_readout.py
deco/lerf_ovs/evidence_modes.py
deco/lerf_ovs/build_bank_b123.py
```

生成 evaluator 已确认保留：

- `SemanticNAG` 加载；
- 原始 DeCo `text_sims`；
- C1 image center 与 C2 text/canon 评分路径；
- NAG 层级搜索和 root/leaf 候选构建；
- 原始渲染评测与 Gaussian 相关路径；
- B1/B2 evidence bank 和 B3 query-conditioned replacement。

## 输入一致性检查

要求的 ramen 输入应包括：

```text
sai_nag_3.0.pt
observations_level3.npz
visible_gaussian_offsets/ids
ramen source/images
ramen Gaussian model
LERF-OVS GT
```

已找到：

- `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt`
- `/root/rivermind-data/THGS/ready_scenes/ramen.zip`，含 `ramen/sai_nag.pt`、Gaussian PLY 和输入相机
- LERF-OVS 相关 GT/reference 目录

未找到：

- `/root/rivermind-data/**/observations_level3.npz`
- 任何包含 `visible_gaussian_offsets` 或 `visible_gaussian_ids` 的现有观测文件
- 本地备份 `E:\CVPR 2026\backup_20261027` 中对应的 ramen production observation

现有 `c1_origin_results/.../ramen/fill/paired_features.npz` 只有 `bb/mb/bm/mm` 的 `(36,512)` 特征，不能替代逐观测 production observation，也没有可验证的 visible Gaussian CSR。现有 ScanNet `fields.npz` 也没有这些字段，不能跨数据集替代。

## 阻塞结论

根据 `CODEX_TASK.md`：缺失 visible Gaussian CSR 时必须停止，不得关闭几何约束，也不得声称 B1 已启用空间支持。因此本轮没有运行 `build_bank_b123.py`、`run_b123.sh` 或 B0/K0/B1/B2/B3 端到端实验，也没有生成 `B123_RESULTS.md`。

待补齐同一 ramen NAG 绑定的 `observations_level3.npz`（包含 region IDs、原始 CLIP 特征、visibility 权重、visible Gaussian offsets/ids）后，才能继续 SHA 绑定、最小查询检查和五臂实验。
