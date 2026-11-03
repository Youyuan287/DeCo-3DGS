# ScanNet-10 当前模型全面评测

## 范围说明

本轮先用当前十个 Gaussian PLY、各自 `sai_nag_3.0.pt` 和现有 `scannet.zip` GT，按 scene0062 已认证的官方 ScanNet scorer 重新建立输入并评测原始 DeCo 主路径。没有使用旧的 805,980 点 `clip_cache` 作为本轮基线。

最新 M1/M2/M3 辅助消融目前只有 `scene0062_00` 具备经过审计的 observation + M1 bank；因此该场完成 off/fixed/adaptive 三组，另外九场只报告严格一致的原始 DeCo off。没有用 scene0062 bank 冒充其它场景，也没有从旧缓存伪造 M1/M2/M3 结果。

## ScanNet-10 原始 DeCo（官方 both_lse，C1 renorm=0）

| 场景 | 19-class mIoU | mAcc | OA | 有 GT 类数 |
|---|---:|---:|---:|---:|
| scene0000_00 | 0.508481 | 0.722603 | 0.711906 | 14 |
| scene0062_00 | 0.823382 | 0.934633 | 0.920759 | 6 |
| scene0070_00 | 0.360460 | 0.430974 | 0.442675 | 9 |
| scene0097_00 | 0.436054 | 0.573624 | 0.556803 | 7 |
| scene0140_00 | 0.525714 | 0.624399 | 0.666878 | 8 |
| scene0200_00 | 0.498026 | 0.602409 | 0.512115 | 8 |
| scene0347_00 | 0.387115 | 0.469745 | 0.527215 | 7 |
| scene0400_00 | 0.445153 | 0.694789 | 0.639436 | 7 |
| scene0590_00 | 0.519395 | 0.608526 | 0.720895 | 14 |
| scene0645_00 | 0.485317 | 0.680119 | 0.690690 | 16 |
| 十场等权平均 | **0.498910** | **0.634182** | **0.638937** | — |

协议是：Gaussian 到 GT mesh 最近点赋标签；19 类 NYU40 子集；root 级别 `both_lse` 文本评分；C1 为减去场景均值但不再 L2 renorm；逐场 scene mIoU 为有 GT 类均值，十场为等权平均。

## 已具备完整辅助消融的 scene0062_00

| 组别 | 3D mIoU | mAcc | OA |
|---|---:|---:|---:|
| DeCo off | 0.823382 | 0.934633 | 0.920759 |
| Fixed prior + M1 | 0.822478 | 0.934911 | 0.920713 |
| Query-adaptive M1 + M3 | **0.823583** | **0.936048** | **0.921460** |

相对 off：Fixed 为 −0.000904，Adaptive 为 +0.000200；Adaptive 相对 Fixed 为 +0.001104。该增益只在一个场景上成立，不能外推为 ScanNet-10 整体收益。

## 当前缺口

以下九个场景已有 Gaussian/NAG/二维特征，但没有同一冻结链上的 `observations_level3.npz`、semantic sidecar 和 M1 bank：

`scene0000_00, scene0070_00, scene0097_00, scene0140_00, scene0200_00, scene0347_00, scene0400_00, scene0590_00, scene0645_00`。

因此本轮不能给出这九场的 M1/M2/M3 消融数字。要完成真正的 ScanNet-10 最新消融，需要先为九场分别导出冻结-NAG observation/sidecar、构建独立 SAS bank，再以相同官方 scorer 跑 off/fixed/adaptive；这一步不能用现有旧 `clip_cache` 替代。

## 文件与复现输入

- 本地当前十场 JSON：`current_official_json/`
- 当前模型/GT 最近点输入清单：`CURRENT_OFFICIAL_INPUTS.json`
- 远程隔离目录：`/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/`
- scene0062 完整 M3 报告：`/root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/SCANNET_M3_QUERY_ADAPTIVE_ABLATION.md`

本轮没有修改生产 Gaussian/NAG、没有重新训练、没有调用特征生成或 NAG 聚类。
