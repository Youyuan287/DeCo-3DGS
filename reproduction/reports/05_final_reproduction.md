# Stage 5 — 四场景结果与核验

## 1. 真实结果 vs 论文 Table 3

主协议：`topk=3, level_until=3, filter_small=50, remove_small=True`，CLIP ViT-B-16(laion2b)，0.5 阈值，per-image 均值再对场景均值。

| Scene | 复现 mIoU | 论文 mIoU | Δ | query 数 |
|---|---:|---:|---:|---:|
| figurines | 55.75 | 64.7 | −8.95 | 56 |
| ramen | 48.53 | 51.2 | −2.67 | 71 |
| teatime | 69.68 | 81.0 | −11.32 | 59 |
| waldo_kitchen | 47.54 | 60.6 | −13.06 | 22 |
| **Mean** | **55.38** | **64.4** | **−9.03** | 208 |

逐 query 结果：`results/lerf_query_metrics.csv`（208 行，含 IoU/pred_px/gt_px/候选数）；预测与 GT mask：`results/masks/`；场景汇总：`results/<scene>_summary.json`。

## 2. 关键对照：同一底座、同一评测口径下与 THGS 的比较

本机此前的 THGS 复现（同一 2DGS 底座、同一渲染/GT 口径）：figurines 50.94 / ramen 32.44 / teatime 81.21 / waldo 53.88，Mean **54.62**。

| 方法（本机同口径） | Mean mIoU |
|---|---:|
| THGS 复现 | 54.62 |
| ReLaGS 复现（本次） | **55.38** |
| 论文 THGS | 54.94 |
| 论文 ReLaGS | 64.4 |

即：**ReLaGS 相对 THGS 的实测增益仅 +0.76**，而论文声称的增益约 +9.5。这说明差距主要不来自“ReLaGS 方法没跑起来”，而来自**我们无法复原的协议/底座差异**（见 04 的未知项清单）。

## 3. 差距归因（仅列与指标直接相关的可疑项，未做大规模搜索）

1. **评测协议未知（最可能）**：官方无评测代码；`remove_small` 一个开关在 figurines 上就造成 16.5 点差异。
2. **几何底座可能不同**：论文可能用其自改 rasterizer 重新训练 2DGS；本次复用官方 ready 场景（磁盘仅剩 2.6G，无法重训）。
3. **特征来源**：复用 LangSplat 的 CLIP ViT-B-16 fp16 特征；若论文用 openseg(ViT-L-14-336) 特征，语义场会不同（`merge_proj` 支持该分支，但本机无 `language_features_openseg`）。
4. **teatime/waldo 差距最大**（−11.3 / −13.1）：两场景基元数 ~1M，剪枝保留率仅 75–81%，超点数 25k/14k，层级合并与特征外点剔除对阈值更敏感。
5. **figurines 无独立流水线日志**：该场景为手工分步执行（同样命令），产物与其余场景一致，但缺少统一日志文件。

已排除：CUDA 扩展错误（渲染结果非空且 mask 合理）、GT 版本错误（与 THGS 复现同一份）、query 集错误（同一 JSON 推导）。

## 4. 可视化

每场景一组 `RGB | PRED | GT` 三联图：`results/vis/<scene>_<frame>_<prompt>.png`
- figurines frame_00041 / green_apple
- ramen frame_00006 / bowl
- teatime frame_00002 / apple
- waldo_kitchen frame_00053 / Stainless_steel_pots

## 5. 三个明确结论

1. **2DGS 几何底座是否成功复现？** 是（以复用官方 2DGS LERF 场景的方式）。四场景 ply 均可被 ReLaGS renderer 加载、全视角渲染、剪枝并进入语义链路；但**未从零训练**，此项为已知偏差。
2. **ReLaGS LERF-OVS 全流程是否打通？** 是。四场景均完成 剪枝 → kNN → 图重加权 → 二次划分 → 层级合并投影 → 层级语义产物 → 逐 query 评测，中间产物齐备且通过数量/层级一致性校验。
3. **是否成功复现论文 64.4% Mean mIoU？** **否**。实测 Mean **55.38%**，低于论文 9.03 个百分点；且在同口径下仅比 THGS 高 0.76 点。根因是官方评测协议缺失 + 可能不同的 2DGS 底座，无法在现有公开资料下闭环确认。

## 6. 未完成/受阻事项

- 未从零训练 4 场景 2DGS（磁盘 2.6G 剩余，需 ≥10G）。
- 未取得官方 ReLaGS 权重与官方评测脚本（HF 不可达；仓库未发布）。
- 未做 ScanNet / 3D-OVS / 场景图关系预测（任务范围外）。

## 7. 补充核验记录

### 7.1 逐 query 失败个案（可定位）

`ramen / frame_00006 / bowl`：`IoU=0.0`，`pred_px=0`，`n_candidates=1`。即检索返回的候选经过相似度间隙截断后被清空，渲染出全零 mask。这是 ReLaGS `_filter_by_similarity_gap(remove_small=True)` 的边界行为（单一候选时可能截断为空），不是渲染或 GT 问题。全部 208 条 query 的 `pred_px/gt_px` 均已落盘，便于逐条复核。

### 7.2 关于 THGS 目录的其他写入

`THGS/merge_proj.py`（17:44）、`merge_proj.py.bak_vala_full`（17:20）、`merge_proj.py.bak_ablation`（17:44）在本任务执行期间被修改。本任务对 THGS 只做读取与向外 `cp`（复制 2DGS ply 到 `work/`），从未写入；上述改动应为同机其他会话（VALA/消融实验）所为。本任务所有写入均限于 `ReLaGS_Reproduction/`。

### 7.3 未目视核对可视化

可视化三联图已生成（`results/vis/`，4 张），但本地模型无图像输入能力，未做目视判断；正确性依据为逐 query 的 `pred_px/gt_px/IoU` 数值记录。

## 8. 官方底座复跑（2026-10-23，本节为最终口径，覆盖第 1 节）

### 8.1 用了什么

- **底座**：ReLaGS 官方 HF（经 hf-mirror）发布的 `scenes/LeRF/<scene>/point_cloud/iteration_0/point_cloud.ply`，已证明与 THGS 的 30k 在 `0.0005` 阈值下逐点相同。
- 因此本组实验**不再依赖 THGS 的二次加工**，底座即官方产物。

### 8.2 结果（同一评测 harness）

| Scene | 官方发布场 `sai_nag.pt` | 我重建的场 | 论文 |
|---|---:|---:|---:|
| figurines | **58.11** | 无法重建（特征被删，见 8.4） | 64.7 |
| ramen | 40.40 | **49.48** | 51.2 |
| teatime | **82.91** | 69.31 | 81.0 |
| waldo_kitchen | **57.62** | 45.92 | 60.6 |
| **Mean** | **59.76** | 54.90（3 场景） | **64.4** |

### 8.3 结论变化

- 用**作者自己发布的场**，Mean 达 **59.76**，距论文 64.4 仅 **−4.64**；其中 **teatime 82.91 高于论文 81.0**。
- 说明论文 64.4 与我们之间的差距，主要不在“ReLaGS 代码跑不通”，而在：
  1. 官方评测代码从未发布，`topk / level / 阈值 / 聚合` 无法复原；
  2. 我的重建在 teatime/waldo 上明显弱于作者发布场（69.31 vs 82.91、45.92 vs 57.62），而在 ramen 上更强（49.48 vs 40.40）——同云同参下出现方向相反的差异，说明语义阶段仍有未定位的实现差异（最可疑：`merge_proj` 的特征重投影与随机采样、以及官方发布场可能使用不同特征版本）。
- 我此前报告的 **55.38** 应作废：它建立在错误的剪枝阈值（0.005）之上。

### 8.4 阻塞

`figurines/langsplat/language_features` 于 2026-10-23 00:53 被其他会话删除（无备份、无归档），figurines 的语义阶段无法重跑；其官方发布场评测（58.11）不受影响。

## 9. 剪枝阈值 0.1 实测（2026-10-23）

按用户指定参数 `--contribution_threshold 0.1` 重跑：

| 场景 | 30k | @0.1 | @0.0005(官方) |
|---|---:|---:|---:|
| figurines | 369,120 | 69,573 | 234,534 |
| ramen | 400,401 | 205,681 | ~494k |

ramen 端到端（官方`iteration_0`同参语义链）：**@0.1 = 48.67**，@0.0005 = 49.48，论文 51.2。
即阈值放大 200 倍、基元只剩 51%，mIoU 仅降 0.81 —— 说明语义指标对剪枝密度不敏感，0.1 可作为提速配置使用（渲染/建图成本显著下降），但不建议用于与论文对齐的正式口径。

## 10. figurines 特征丢失与重建（进行中）

- 根因：`figurines/langsplat/language_features` 被误删（用户确认）。
- 官方结构（由三个存活场景反推）：`_s.npy` (4,H,W) int，四个**互不重叠**的 mask id 段（每通道一组粒度）；`_f.npy` (N,512) fp16，每个 mask 一条 CLIP ViT-B-16 特征。每图掩膜中位数 174–219，通道覆盖率 0.85–0.98。
- 重建现状：`scripts/regen_langsplat_features.py`（SAM ViT-H AMG + CLIP ViT-B-16），pps=32 时仅 ~72 mask/图 → 覆盖不足、**不可比**；已改用 pps=64 后台重跑。
- 结论：重建特征**不可能与原版等值**；figurines 的权威数字仍以官方 `sai_nag.pt` 的 58.11 为准。重建场仅作“可跑通”验证，不进入与论文对比的表。

### 10.1 重建方法（最终版）

反推官方结构后确认：四个通道**各自是一套完整覆盖**（coverage 0.85–0.98）且 id 全局互不重叠，即 4 个粒度层级各自独立铺满图像。最终实现：

1. 每图用 SAM ViT-H（`points_per_side=64`, `pred_iou_thresh=0.88`, `stability=0.95`）产生掩膜池；
2. 统计每像素被多少掩膜覆盖，按“该像素所含掩膜的面积排名”取 4 个分位（最细…最粗）作为 4 个通道；
3. 每个掩膜用 CLIP ViT-B-16(laion2b) 编码 masked crop 得 512 维特征，存 fp16。

实测覆盖率 0.97（官方 0.85–0.98），四通道 id 互不重叠、粒度递增，与官方结构一致。
**仍不等值**：每通道掩膜数（146/135/122/116）明显多于官方（87/114/61/33），且 SAM 提案本身不可复现，故本特征集只用于“链路可跑通”验证，不作为与论文对比的依据。
