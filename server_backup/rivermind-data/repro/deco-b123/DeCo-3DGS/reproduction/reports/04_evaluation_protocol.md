# Stage 4 — LERF-OVS 评测协议

## 1. 结论先行

ReLaGS **没有**发布任何评测代码（README 仅写 “Evaluation scripts will be published soon”，仓库中 `evaluate.py`/`eval_*.py` 均不存在）。因此本报告**不声称**复现了论文官方评测；下表区分了来源。

## 2. 协议来源分解

| 环节 | 来源 | 说明 |
|---|---|---|
| 语义特征、层级表示、超点检索 | **ReLaGS 官方** | `nag_data.SemanticNAG.search_matched_superpoint_in_mhtree`（未自研） |
| CLIP 编码器与文本模板 | **ReLaGS 官方** | `utils/vlm_utils.ClipSimMeasure`：ViT-B-16 `laion2b_s34b_b88k`，文本 `[query] + ["object","things","stuff","texture"]` 对比模板，softmax(×10) 取负类 argmin |
| 单条 query → 高斯二值集 | **ReLaGS 官方** | 检索返回 `rel_gaussians`，展开为 20 通道二值语义 |
| 渲染与前景判定 | **参考实现（THGS）** | `render(...)["semantics"]` 第 0 通道 > 0.5，与 THGS `test_lerf.py` 完全一致 |
| GT mask / query 集 | **参考实现（THGS `eval_seg.py`）** | 逐帧读取 `label/<scene>/<frame>.json` 的 `objects[].category` 作为 query，多边形 `fillPoly` 生成 GT |
| IoU 与聚合 | **参考实现（THGS）** | 单图内先对 query 取均值，再对图取均值，最后对场景取均值 |

## 3. 无法从公开资料唯一确定的环节

1. `topk`（根层候选数）：ReLaGS 默认 5，THGS 调用为 3。
2. `level_until`：ReLaGS 中实为**开关**——`-1` 只用根层，非 -1 才启用“叶子层提升 + 相似度间隙截断”；THGS 的 `level=[2,3]` 对应后者。
3. `filter_small`（根候选最小点数）与 `remove_small`（是否做间隙截断）：默认 50 / True。
4. 分割阈值：0.5 硬阈值 vs 自适应阈值（THGS 仓库内被前次工作改成 adaptive，未采用）。
5. 聚合方式：per-image 均值 vs 全 query 扁平均值。
6. 底座：论文是否用其自改 rasterizer 重训 2DGS（本任务复用官方 ready 场景）。

**主指标采用（先验、非 GT 调参）**：`topk=3, level_until=3, filter_small=50, remove_small=True`。理由：`-1` 只用根层与论文宣称的部件级能力矛盾；`topk=3` 对齐唯一可考的公开调用。该组合在敏感性实验中也恰好最优（见下），但**未**用 GT 做过选择。

## 4. 协议敏感性（figurines / teatime，仅报告，不作主指标）

| 变体 | figurines mIoU | teatime mIoU |
|---|---:|---:|
| 主配置（topk=3, leaf 提升, remove_small=True） | **55.75** | **69.68** |
| `remove_small=False`（不截断） | 39.23 | 61.10 |
| `level_until=-1`（仅根层） | 39.50 | — |
| `topk=5` | 54.02 | — |

结论：`remove_small` 与“叶子层提升”是影响最大的两个开关（figurines 上相差 16.5 / 16.3 个点），说明协议不确定性足以解释数个百分点甚至十几点的差异。

## 5. 诚实性约束

- 未使用 GT 为任何 query 选择层级或阈值；
- 未报告 oracle/best-layer IoU；
- 逐 query 的 IoU、预测 mask、GT mask 全部落盘（`results/lerf_query_metrics.csv`、`results/masks/`）；
- 无法确定的环节已在上面逐条列出。
