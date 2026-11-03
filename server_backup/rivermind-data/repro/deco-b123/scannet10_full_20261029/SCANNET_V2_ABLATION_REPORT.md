# ScanNet-10：认证基线上的 V2 候选与消融

本报告使用已通过 `verify_expected.py` 的 seed=1234 ScanNet-10 分区、Gaussian、GT 和官方 DeCo scorer。新增消融不重新训练、不重建 NAG，不混用旧缓存。

## 组别

- DeCo off：原始 both_lse + C1/C2 主路径。
- M1 Fixed：M1 debiased semantic bank + query-independent prior 权重。
- M1+M2 Fixed：Fixed 读出加多视角 reliability。
- M1+M3 Adaptive：M1 + query-adaptive Softmax 权重，M2 OFF；这是与 LERF-OVS 认证 V2 最接近的配置。
- M1+M2+M3 Adaptive：同时启用 M2 reliability 和 M3 adaptive weighting。

所有组使用相同 NAG、Gaussian、observation、bank、C1/C2、temperature=0.08 和 coefficient=0.25。

## 十场景 19-class scene mIoU

| 场景 | DeCo off | M1 Fixed | M1+M2 Fixed | M1+M3 Adaptive | M1+M2+M3 Adaptive |
|---|---:|---:|---:|---:|---:|
| scene0000_00 | 0.508481 | 0.508382 | 0.508607 | 0.496413 | 0.509905 |
| scene0062_00 | 0.823382 | 0.822478 | 0.822478 | 0.823583 | 0.822478 |
| scene0070_00 | 0.360460 | 0.359337 | 0.359469 | 0.349726 | 0.349858 |
| scene0097_00 | 0.436054 | 0.436082 | 0.436082 | 0.435313 | 0.435313 |
| scene0140_00 | 0.525714 | 0.511766 | 0.512414 | 0.512662 | 0.511886 |
| scene0200_00 | 0.498026 | 0.497978 | 0.497978 | 0.499230 | 0.499230 |
| scene0347_00 | 0.387115 | 0.390418 | 0.390418 | 0.369512 | 0.392228 |
| scene0400_00 | 0.445153 | 0.445153 | 0.445153 | 0.445153 | 0.445153 |
| scene0590_00 | 0.519395 | 0.521686 | 0.520393 | 0.521359 | 0.521643 |
| scene0645_00 | 0.485317 | 0.486781 | 0.486905 | 0.489710 | 0.489705 |
| 十场景等权 macro | **0.498910** | 0.498006 | 0.497990 | 0.494266 | 0.497740 |

## 结论

当前 ScanNet-10 上，认证 DeCo off 仍是宏平均最高组。M1 Fixed 为 −0.000904，M2 在 Fixed 分支的独立差异为 −0.000016；M1+M3 Adaptive 为 −0.004644。加入 M2 后 Adaptive 分支恢复 +0.003474，但仍比 DeCo off 低 −0.001170。因此不能声称 V2 在 ScanNet-10 上带来整体提升；scene0062 的 +0.000200 仅是局部结果。

## 候选审计

候选级记录保存在服务器：

`/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/sceneXXXX_00/SCANNET_M3_CANDIDATE_CHANGE.csv`

每行记录 root/class/query 的五组分数、五组类别选择和 Gaussian 数量；权重与 mask 记录在同目录的 `SCANNET_M3_WEIGHT_TRACE.csv` 与 `SCANNET_M3_MASK_XOR.csv`。本地目录保留了一个代表性场景的候选 trace，完整逐场景文件以服务器隔离目录为准。
