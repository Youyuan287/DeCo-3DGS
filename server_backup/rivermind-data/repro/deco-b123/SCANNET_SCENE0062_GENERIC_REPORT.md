# ScanNet scene0062_00 通用实验报告

## 状态说明

本报告整理的是服务器/本地备份中已经存在且可审计的 `scene0062_00` 通用实验缓存，不冒充一次新的 B123 端到端重跑。

当前无法直接重跑 ramen 式 B0/K0/B1/B2/B3 的原因：

- 训练后的 Gaussian PLY 已不在服务器或本地备份中；
- 旧 NAG `/root/rivermind-data/repro/deco-3dgs-38620a7d/recovery/variants/post_rofa/output/scannet/scene0062_00/sai_nag_3.0.pt` 只有单层标签（shape `[805980]`，85 个区域），不是 ramen B123 所需的四层 NAG；
- 本地 `datasets_scannet.tgz` 只包含源图像、相机、语言特征和初始 `points3d.ply`，不包含训练后 Gaussian PLY。

因此以下结果是已有 generic ScanNet evaluator/cache 的复核，不能标为新的官方 B123 结果。

## 缓存来源

主结果：

```text
E:\CVPR 2026\backup_20261027\evidence_run_01\eval\scene0062_00\scene_results.json
```

字段缓存：

```text
E:\CVPR 2026\backup_20261027\evidence_run_01\fields\scene0062_00\field.json
```

原始缓存 provenance 中记录：

- ReLaGS commit：`38620a7d84ff21db28ab19a2e067cf663b00698a`
- Gaussian 数：`805980`
- 训练视角：`219`
- 图像文件：`244`
- level：`3`
- 旧模型 PLY SHA256：`a6ce2abd77a3be2b875ba56f992891751280ad8c3d2a5184399ca94f20da45c1`
- 旧 NAG SHA256：`ff0d00c0b3f2fa5964d93a9ac7c7321290251629f0c03e5048bab7aa50369c42`

## 通用 ScanNet 结果

场景：`scene0062_00`。评测包含 19 类设置（其中 6 类在该场景出现），同时保留 mAcc 和 OA。

| 方法 | mIoU | mAcc | OA |
|---|---:|---:|---:|
| legacy_raw_dot（原始基线） | 0.363972 | 0.652304 | 0.313345 |
| legacy_c1_dot（C1） | **0.685575** | **0.788450** | **0.835684** |
| mean_raw | 0.338632 | 0.662854 | 0.428901 |
| mean_c1 | 0.451241 | 0.682742 | 0.479244 |
| evidence_uniform | 0.339221 | 0.662178 | 0.428947 |
| raw_attention | 0.329714 | 0.659232 | 0.412786 |
| scale_attention | 0.295261 | 0.635004 | 0.377621 |
| residual_attention | 0.290707 | 0.630423 | 0.371179 |

## 结果解读

- C1 相对 legacy_raw_dot 的 mIoU 提升：`+0.321603`，即约 `32.16` 个百分点。
- C1 的 OA 从 `0.313345` 提升到 `0.835684`。
- 在该缓存实验中，C1 明显有效；但这不是新导出的 B123 bank 实验。
- evidence/attention 变体均低于 legacy_c1_dot，不能据此宣称 B1/B2/B3 的官方 ScanNet 性能。

## 当前缺失与真正重跑所需输入

要进行与 ramen 相同的 ScanNet B0/K0/B1/B2/B3，需要恢复：

1. 与旧 NAG SHA256 绑定的训练后 Gaussian PLY；
2. 完整层级 NAG 或明确的四层层级标签；
3. 219 个训练视角及其 language features；
4. 对应 GT 和相机文件；
5. 基于同一 NAG/模型重新导出的 observation CSR。

在这些输入恢复前，本报告中的 generic cache 结果是可报告的 ScanNet 0062 结果，但不应与 ramen 官方 NAG B123 结果混在同一张表中。
