# M1-SAS 数据审计（停止版）

日期：2026-10-29  
状态：**BLOCKED — 未生成 SAS bank，未运行新的 GPU 评测**

## 已核验输入

四个官方冻结 observation 均可读取，特征有限且维度为 512：

| scene | observations | views | unique regions | NPZ keys |
|---|---:|---:|---:|---|
| ramen | 8,470 | 131 | 248 | features, weights, region_ids, view_ids, visible/member CSR |
| figurines | 16,800 | 298 | 177 | 同上 |
| teatime | 20,096 | 177 | 1,225 | 同上 |
| waldo_kitchen | 13,425 | 187 | 397 | 同上 |

四个 observation NPZ 的 key 集合完全没有 `mask_labels`、`trace_conf`、`beta_raw`、`beta_kept`、`seg_min`、`seg_max`、feature-level 或 random-encoding seed。

## 生产代码事实

冻结 `merge_proj.py` 的 `SAI3D.extract_gaussian_features()` 在每个视角内执行：

```python
enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
...
sim = torch.matmul(gau_sem, enc.T)
gau_sem = sim.argmax(dim=-1)
sim_filter = sim.max(dim=-1)[0] > 0.85
gau_sem[~sim_filter] = 0
```

该函数没有固定 seed、没有保存 `enc`、没有保存 `sim.max()`，也没有导出每视角的 `gau_sem`。已有 observation sink 只保存归一化的 `sp_feat`、visibility weight、region/view ID 和 member/visible Gaussian CSR。

## 为什么不能安全重建

原 observation 只保留了

\[
f_o = \sum_m \beta^{kept}_{om} f_{vm}
\]

以及 visibility 比例。归一化的混合向量无法唯一反推出 mask 行、trace label、置信度或 `beta`。如果现在重新调用 trace，随机编码未被历史记录，得到的是新的关联，不是对原 observation 的数值重建。用当前 visible CSR 代替语义支持会直接违反 M1-SAS 的定义。

因此目前不能通过“原始 CLIP 混合特征重建”门槛，也不能进行 semantic weighted Jaccard、visible-builder 等价性、SAS bank 或四场景评测。

## 已排除项

- 已搜索当前隔离实验目录内所有 `.npz/.pt`；没有包含 `trace_conf`、`mask_labels` 或 `beta_kept` 的 sidecar。
- 已核对 NAG、observation、B1 bank 哈希，未修改任何生产输入。
- 未调用 `merge_proj.py`、`assign_label()`、`everything_seg()`，未重建 NAG，未重建 bank，未运行 mIoU。

## 恢复所需的唯一数据

需要从同一 Gaussian 顺序、同一 camera/view 顺序、同一 feature level、同一二维 mask 输入和**已知/固定的原始 random seed 或已保存 enc 矩阵**重新导出：`trace_label`、`trace_conf`、`seg_min/seg_max`、`beta_raw`、`beta_kept`。导出必须与 observation 在 `(region_id, view_id, level_id)` 上逐行对齐，并先通过重建误差验收。

在这些数据出现前，按任务停止条件不继续实现或宣称 M1-SAS 结果。
