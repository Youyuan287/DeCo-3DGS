# ScanNet-10 M1/M2/M3/Adaptive 完整消融

## 评测组别

- **DeCo off**：原始官方 ScanNet `both_lse` + C1 renorm=0 主路径。
- **M1 + Fixed**：M1 debiased semantic bank，辅助 mode 使用查询无关 prior 权重。
- **M1 + M2 + Fixed**：在上一组加入多视角语义一致性 reliability。
- **M1 + M3 Adaptive**：M1 bank，使用查询自适应 Softmax mode 权重，M2 关闭。
- **M1 + M2 + M3 Adaptive**：同时启用 M2 reliability 和 M3 query-adaptive 权重。

所有组固定同一 Gaussian、NAG、observation、M1 bank、19 类 NYU40 子集、C1/C2、temperature=0.08 和 correction coefficient=0.25。没有重新训练、没有修改 NAG、没有混用旧 `clip_cache`。

## 每场 19-class 3D mIoU

| 场景 | DeCo off | M1 Fixed | M1+M2 Fixed | M1+M3 Adaptive | M1+M2+M3 Adaptive |
|---|---:|---:|---:|---:|---:|
| scene0000_00 | 0.508481 | 0.508382 | 0.508607 | 0.496413 | **0.509905** |
| scene0062_00 | 0.823382 | 0.822478 | 0.822478 | **0.823583** | 0.822478 |
| scene0070_00 | 0.360460 | 0.359337 | 0.359469 | 0.349726 | 0.349858 |
| scene0097_00 | 0.436054 | 0.436082 | 0.436082 | 0.435313 | 0.435313 |
| scene0140_00 | 0.525714 | 0.511766 | 0.512414 | 0.512662 | 0.511886 |
| scene0200_00 | 0.498026 | 0.497978 | 0.497978 | 0.499230 | 0.499230 |
| scene0347_00 | 0.387115 | **0.390418** | 0.390418 | 0.369512 | 0.392228 |
| scene0400_00 | 0.445153 | 0.445153 | 0.445153 | 0.445153 | 0.445153 |
| scene0590_00 | 0.519395 | **0.521686** | 0.520393 | 0.521359 | 0.521643 |
| scene0645_00 | 0.485317 | 0.486781 | 0.486905 | **0.489710** | 0.489705 |
| **十场等权宏平均** | **0.498910** | **0.498006** | **0.497990** | **0.494266** | **0.497740** |

## 宏平均差分

| 差分 | mIoU |
|---|---:|
| M1 Fixed − DeCo | −0.000904 |
| M1+M2 Fixed − M1 Fixed | −0.000016 |
| M1+M3 Adaptive − DeCo | −0.004644 |
| M1+M2+M3 Adaptive − DeCo | −0.001170 |
| M1+M2+M3 Adaptive − M1+M3 Adaptive | **+0.003474** |

## 结论

1. 在 ScanNet-10 上，M1 单独没有带来整体提升，宏平均从 0.498910 降至 0.498006。
2. M2 在 Fixed 分支几乎没有独立贡献：宏平均差分为 −0.000016。
3. M3 Adaptive 在关闭 M2 时整体退化最明显：−0.004644。
4. 加入 M2 后，Adaptive 分支恢复了约 +0.003474，但仍比原始 DeCo 低 −0.001170。
5. 因此 ScanNet-10 不支持“Adaptive M3 独立带来总体收益”的结论；M2 主要表现为对 M3 的抑制/稳定化，而不是独立提升。
6. 局部场景存在正向变化：scene0000_00 的 M1+M2+M3 达到 0.509905，scene0347_00 的 M1+M2+M3 达到 0.392228，scene0062_00 的 M1+M3 达到 0.823583；但不足以抵消其它场景退化。

## 结果位置

- 远程隔离目录：`/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/`
- 每场 summary：`sceneXXXX_00/SUMMARY.json`
- 每场逐类/根竞争明细：`sceneXXXX_00/SCANNET_M3_PER_CLASS.csv`、`SCANNET_M3_CANDIDATE_CHANGE.csv`
- 本地 summary 已复制到本目录的 `ablation/`
