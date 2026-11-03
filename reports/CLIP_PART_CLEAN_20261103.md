# 干净裁剪下 CLIP 能不能分出部件：LERF-OVS 与 ScanNet 的结论相反

日期：2026-11-03
脚本：`analysis/clip_part_clean_probe.py`
产物：`results/clip_part_clean_20261103/`（143 条记录）

## 问题

两边管线里，每条特征都是 `CLIP(mask 覆盖的像素)`。所以"部件为什么一直识别不出来"
可以脱离管线单独问：

> 给 CLIP 一块**干净的**部件像素，它会把部件名排在父物体/邻居之前吗？

- 分得出 → 损失是管线造成的（没把部件像素交给 CLIP），可以修；
- 分不出 → 编码器/名字侧的极限，改分割、层级、聚合都无效。

## 方法

同一条代码路径，两个数据集各自的 GT 与名字表：

| | GT | 裁剪 | 名字表 |
|---|---|---|---|
| ScanNet | `_vh_clean_2.labels.ply` 投影（约定 `p_cam=(p−C)@R`，已核对）→ 最大连通域 | 外扩 25%，外部置 127 | NYU40 的 19 类 |
| LERF-OVS | `test_mask/<view>/<query>.png` | 同上 | `structure.json` 的场景 prompts |

CLIP = ViT-B-16 / laion2b（与管线同一个）。指标：真名在名字表里的 rank、以及相对其余名字的
margin（σ = (cos_true − mean_others) / std_others）。GT 只决定裁哪块，不参与打分。

## 结果

| 分组 | n | rank1 | 平均 rank | 中位 σ |
|---|---:|---:|---:|---:|
| **LERF 部件** | 18 | **0.611** | **1.67** | 2.47 |
| LERF 其他 | 41 | 0.854 | 1.15 | 3.32 |
| **ScanNet 部件** | 21 | **0.000** | 6.43 | 0.944 |
| ScanNet 其他 | 63 | 0.111 | 8.30 | 0.316 |

> 更正：第一版 LERF 用了排序取帧（`frame_00001.jpg`），正确的是 `images/test_<v>.jpg`。
> 用错帧时 LERF 部件 rank1 只有 0.222；修正后 0.611。上表是修正后的。

### LERF：CLIP 做得到

干净裁剪下排第一的：`porcelain hand`（σ 2.8–14.7）、`old camera`（5.3–10.9）、
`rubber duck with red hat`（2.0–7.9）、`green/red apple`（2.9–4.2）、`yellow bowl`（2.6）、
`egg`（3.3）、`tea in a glass`（2.6–3.3）、`green/red toy chair`、`sheep`、`wavy noodles in bowl`。

### ScanNet：CLIP 做不到

| 类 | rank / 19 | top1 |
|---|---|---|
| door | 2, 2, 4, 6, 8, 8 | **wall / picture** |
| window | 3, 4, 5 | **wall / picture** |
| table | **11, 12, 13, 14, 15** | wall / picture / floor |

part-like 21 条里 **0 条**排第一；table 的 σ 多数为负。

## 结论

1. **"没建父部件—子部件的语义链接"不是主因。** LERF 侧 CLIP 在干净像素上已经能挑出部件
   （porcelain hand σ 6.3），缺的是**候选**——已有记录显示 recall 族 2D mask 的 IoU 中位只有 0.092，
   即管线从来没把部件像素交给 CLIP。
2. **ScanNet 侧不该按"部件"来治。** door/window 的干净裁剪输给 wall/picture，
   说明这 19 个名字在外观上不可分；table 的 top1 是 wall/picture/floor 且 σ 为负，
   更像**词表映射**问题（counter/desk/table 混用），不是"部件太难"。
3. 因此"在已有观测上做加权 / 聚类 / 一致性"这条路，在两边都不可能有部件增益——
   这一条现在有直接测量支撑，不再只是推断。

## 限制

- ScanNet 图像是 648×484（`extract_meta.json`：src 1296×968 降采样 2×），干净裁剪本身模糊；
  服务器无原图，无法把"分辨率"与"外观不可分"分开。但 table 掉到 11–15 且 σ 为负不像模糊症状。
- LERF 的 test view → 图像帧的对应是推断的（`int(v)` 取第 v 帧）；多个 prompt 排第一说明对齐基本正确，
  但没有独立校验。
- 每类每场景只有 3 帧（ScanNet）/ 3 个视角（LERF），样本量小，只看方向。
- 裁剪外部填充用 127（与管线 `get_seg_img` 一致），未做多填充方式对比。