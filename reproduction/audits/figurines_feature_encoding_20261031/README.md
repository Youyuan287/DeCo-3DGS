# 审计：figurines 语义场能否从零重建？（2026-10-31）

> 结论先行：**能，而且几乎逐位一致。** 用官方 `scripts/image_encoding.py`（laion2b_s34b_b88k + SAM ViT-H）
> 重新编码 299 帧后重建的 `sai_nag.3.0.pt`，其逐簇特征与作者发布的 `sai_nag.pt` 余弦相似度为 **0.9999**；
> 在同一评测脚本下 mIoU **0.6610**，作者发布权重为 **0.6329**（差 +2.81，来源见 §5，集中在 3 条 query）。
>
> 本审计**更正** [`../../FINAL_REPORT.md`](../../FINAL_REPORT.md) §8.3 的判断
> （"自建特征与官方 `image_encoding.py` 不一致 → 产物不等值"）。当时的重建确实不等值，
> 但根因不是编码参数，而是 §3 的**符号链接路径穿透**；修掉之后重建是等值的。

---

## 1. 问题

`FINAL_REPORT.md` 的主结果（Mean 64.01）复用的是**作者发布的** `sai_nag.pt`。
这留下一个未验证环节：**这份语义场本身能否由发布的代码 + 发布的输入重建出来？**
如果不能，那么"复现"实际只覆盖了"评测"一半。

LREF-OVS figurines 是唯一同时具备以下条件的场景，故选它做端到端重建：

- 299 张训练视图、原图 986×728 可用；
- 官方 2DGS 底座（`iteration_0`）与作者 `sai_nag.pt` 均已发布，可直接作为对照；
- 56 条 query / 4 张标注图，评测成本低。

## 2. 结果

同一评测脚本、同一参数（`--topk 5 --level_until 3 --filter_small 50 --center none --renorm 0`）：

| 语义场 | 来源 | 层级簇数 L1/L2/L3 | mIoU | 与官方 `nag_feat` 余弦 |
|---|---|---|---:|---:|
| **官方发布** | `hf:dfki-av/ReLaGS/scenes/LeRF/figurines/sai_nag.pt` | 2103 / 463 / 177 | 63.29 | — |
| **本审计重建** | 官方编码器重编码 299 帧 + 官方五步管线 | **2113 / 468 / 178** | **66.10** | **0.9999 / 0.9999 / 0.9999** |
| 对照（错误特征源） | 路径穿透读到的 LangSplat OpenAI ViT-B/16 特征 | 2110 / 468 / 178 | 58.11 | 0.9642 / 0.9680 / 0.9721 |
| 对照（官方特征 + 本分区） | 把官方 `nag_feat` 按最大重叠映射到本分区 | 2110 / 468 / 178 | 64.68 | ≈1.0（同源） |

余弦按"L(k) 簇 → L(k+1) 子簇最大重叠"配对、按像素数加权，即 §6 的 `COS vs official` 行。

**读法**：重建的特征与作者的特征在同一层级上是**同一个语义场**（0.9999）。
58.11 那一行是同一个代码路径喂了另一套 CLIP 权重（OpenAI ViT-B/16）的结果，与作者无关。

### 2.1 全部 14 次重跑（`results/rerun_matrix.csv`）

同一评测脚本、同一 56 条 query / 4 图、同一参数，只改"喂什么语义场"和"随机种子"：

| # | 臂（`results/reruns/<arm>__*.{log,csv}`） | 语义场来源 | seed | mIoU | flat |
|---:|---|---|---:|---:|---:|
| 1 | `fig_relags2_s7` | 官方编码器重编码（正确路径） | 7 | **66.10** | 65.56 |
| 2 | `swapA` | 本分区 + 官方 `nag_feat`（最大重叠映射） | 7 | 64.68 | 64.07 |
| 3 | `official` | 作者发布 `sai_nag.pt` | — | 63.29 | 62.78 |
| 4 | `fig_up_s7` | LangSplat 特征 + **上游未打补丁** `merge_proj.py` | 7 | 58.11 | 57.55 |
| 5 | `fig_relags_s7` | LangSplat 特征 + 本补丁版 | 7 | 58.11 | 57.55 |
| 6–10 | `figurines_s0/s7/s42/s1234/s2024` | LangSplat 特征 + 播种扫描 | 0/7/42/1234/2024 | 58.11 ×5 | 57.55 ×5 |
| 11–12 | `fig_off_s0/s42` | 同上，DeCo 两算子关闭 | 0/42 | 58.11 ×2 | 57.55 ×2 |
| 13 | `mp_off` | 同上，`merge_proj` 去均值关闭 | 7 | 58.11 | 57.55 |
| 14 | `swapB` | 官方标签 + 本特征 | 7 | **OOM，未跑完** | — |

三条可直接读出的结论：

1. **换种子不动分**（#6–10）：5 个种子到小数点后 4 位相同 → `torch.normal` 的随机性不是偏差来源。
2. **换实现不动分**（#4 vs #5、#11–13）：上游版 / 本补丁版 / 关算子，全 58.11 → 补丁没有引入偏差。
3. **只换特征就动 5.2 分**（#3 vs #5 = 63.29 vs 58.11；#3 vs #2 = 同特征换分区的 1.4 分）。
   #1 比 #3 高 2.81，是"特征等价 + 分区微差"的结果，见 §5。

`swapB` 的日志保留了 OOM 现场（`swapB__eval.log`），作为"官方标签按最大重叠贴回本分区"这条路走不通的记录。

## 3. 根因：符号链接路径穿透（`dataset_readers.py:117`）

官方 `scripts/image_encoding.py` 把特征写成 `<scene>/language_features/<frame>_f.npy`，
而 `scene/dataset_readers.py:117` 用**字符串拼接**读取：

```python
feature_path = os.path.join(images_folder, f'../language_features/{image_name}_f.npy')
```

我最初把重建用的场景目录做成"薄壳"：`images/`、`sparse/`、`poses_bounds.npy` 全部
**符号链接**回原始数据。于是 `images_folder/..` 在文件系统层面解析成
**符号链接目标的父目录**，`language_features` 落回原始场景 —— 也就是 LangSplat 那份
`figurines/language_features`，而不是我新编码的目录。

判据（直接打印 `Scene` 里每台相机实际加载到的那一份）：

```
cam frame_00137  sem=(237,512)  cos(sem, new_dir)=0.91110  cos(sem, langsplat_dir)=1.00000
cam frame_00141  sem=(266,512)  cos(sem, new_dir)=0.91477  cos(sem, langsplat_dir)=1.00000
cam frame_00218  sem=(205,512)  cos(sem, new_dir)=0.91170  cos(sem, langsplat_dir)=1.00000
```

即：`-s` 指向新目录，读到的却是旧目录的文件，且是逐位相同（cos 1.00000）。

**修法**：`images/` 必须是**真目录**，图片逐个 `ln -s` 进去；`sparse/`、
`poses_bounds.npy`、`language_features` 仍可整体软链（不参与 `..` 解析）。

```bash
SR=$B/src_fig_relags2
mkdir -p $SR/images
for f in $(ls $ORIG/images); do ln -sf "$ORIG/images/$f" "$SR/images/$f"; done
ln -sfn $ORIG/sparse $SR/sparse
ln -sfn $ORIG/poses_bounds.npy $SR/poses_bounds.npy
ln -sfn $B/work_feat/figurines_relags $SR/language_features   # 新编码
# 自检：realpath 必须落在新目录
python -c "import os;print(os.path.realpath('$SR/images/../language_features/frame_00137_f.npy'))"
# -> /root/rivermind-data/ReLaGS_Reproduction/work_feat/figurines_relags/frame_00137_f.npy
```

这个 bug 在"重建 vs 评测"两套流程之间制造了一个假象：管线每一步都报成功、
层级结构也合理（2110/468/178），只有数字偏低 5 个点。它不会抛异常，只能靠
**对特征本身**做逐位比对发现。

## 4. 特征源差异（OpenAI ViT-B/16 vs laion2b_s34b_b88k）

`figurines/langsplat/language_features` 来自 LangSplat，用 **OpenAI CLIP ViT-B/16**；
ReLaGS 的 `scripts/image_encoding.py` 用 **`ViT-B-16 laion2b_s34b_b88k`**。两者同为 512 维、
掩码数量完全相同（同一 SAM、同一 `points_per_side=32`），但向量不同：

| 比较对象 | 逐掩码最佳匹配余弦 | 簇级加权余弦（L1/L2/L3） |
|---|---:|---|
| OpenAI ViT-B/16 vs laion2b（同一批掩码） | 0.9126 / 0.9179 / 0.9… | 0.9642 / 0.9680 / 0.9721 |
| 本重建 laion2b vs 作者发布 `nag_feat` | — | **0.9999 / 0.9999 / 0.9999** |

所以 0.96 的余弦差就值 **5.2 个 mIoU 点**（58.11 → 63.29）。这解释了为什么早期的重建
"结构全对、数字偏低"：喂错了 CLIP 权重，结构再对也没用。

## 5. 重构后的 +2.81：它来自 3 条 query，不是普遍提升

逐 query 对齐（56 条，`results/official_figurines.csv` vs `results/relags2_figurines.csv`）：

- **变好 5 条 / 不变 49 条 / 变差 2 条**；
- 主要增量：`frame_00105/pumpkin` 0.000 → **0.721**（该条在官方权重下整幅为空）、
  `frame_00105/jake` 0.423 → 0.850、`frame_00041/jake` 0.442 → 0.849；
- 负向：`frame_00152/green apple` 0.913 → 0.914（+0.0001）、`frame_00195` 视图 −0.0001。

因此正确表述是：**重建出的语义场与发布权重等价（cos 0.9999），评测分数在
0.633–0.661 的同一水平；差距集中在个位数 query 上，不构成"精度提升"的主张。**

## 6. 可复现步骤

```bash
B=/root/rivermind-data/ReLaGS_Reproduction      # 工作根
D=<DeCo-3DGS checkout>                          # 与仓库同版本
V=$B/venv                                       # torch 2.2.0+cu118 + pytorch3d/FRNN/PyG 2.3.0
ORIG=$B/.../lerf_ovs/figurines                  # 原始场景（images + sparse + language_features）
SAM=$B/weights/sam_vit_h_4b8939.pth

# 1) 编码 299 帧（官方脚本；本审计为提速把 seg 存成 int16 并分 3 片并行）
cd $D && $V/bin/python scripts/image_encoding.py \
    --source_path $ORIG --sam_ckpt_path $SAM       # -> $ORIG/language_features/<frame>_{f,s}.npy

# 2) 造一个"薄壳"源目录（关键：images/ 是真目录，逐文件软链）
SR=$B/src_fig_relags2; ...                        # 见 §3，务必用 realpath 自检

# 3) 官方五步管线，产出 sai_nag_3.0.pt
$V/bin/python $B/tools/run_seeded.py 7 merge_proj.py -s $SR -m $MD \
    --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0

# 4) 评测（与 FINAL_REPORT 主口径完全一致）
$V/bin/python reproduction/scripts/eval_lerf_relags.py --scene figurines \
    --source $SR --model $MD --nag $MD/sai_nag_3.0.pt \
    --gt_root $B/ref/hf_lerf/lerf_ovs/label --out $O --csv $O/q.csv \
    --topk 5 --level_until 3 --filter_small 50
```

前置：`max_weight_pruning --contribution_threshold 0.0005` → `sp_partition -a` →
`graph_weight --level 1` → `sp_partition -k neighbor_new.pt --pcp_regularization 0.1
--pcp_spatial_weight 0.1 -a` → `merge_proj`，与 `FINAL_REPORT.md` §7 相同。
`run_seeded.py` 只是 `torch.manual_seed(seed)` 后 `runpy` 目标脚本 —— 因为
`merge_proj.py` 在 207 行用 `torch.normal` 初始化簇特征，**未播种时结果不可复现**。

### 6.1 关于随机性（顺带核实）

`merge_proj.py:207` `torch.normal(mean=cluster_mean, std=...)` 采样每个簇的初始特征，
未播种。固定种子 0/7/42/1234/2024 各跑一遍（同为错误特征源）：

| seed | 0 | 7 | 42 | 1234 | 2024 |
|---|---|---|---|---|---|
| mIoU | 58.11 | 58.11 | 58.11 | 58.11 | 58.11 |

簇数会在 ±3 内浮动（177 vs 178），mIoU 到小数点后 4 位不变 —— 随机性**不是**偏差来源，
但它确实让"逐位复现"不可能，故本审计所有重建都显式播种。

## 7. 产物

| 文件 | 内容 |
|---|---|
| `results/rerun_matrix.csv` | 全部 14 次重跑的 mIoU / flat / 语义场来源 / 种子（§2.1） |
| `results/reruns/` | 14 次重跑的评测日志 + 逐 query CSV（`<arm>__eval.log`、`<arm>__q.csv`） |
| `results/official_figurines.csv` | 官方 `sai_nag.pt` 的 56 条逐 query IoU |
| `results/relags2_figurines.csv` | 本重建（laion2b + 正确路径）的 56 条逐 query IoU |
| `results/figurines_relags2_sai_nag_3.0.pt` | 重建产物（`nag` 4 层 + `nag_feat` 3×512） |
| `scripts/eval_lerf_ovs.py` | 每次审计重跑用的评测脚本（与 `../../scripts/eval_lerf_relags.py` 同协议） |
| `scripts/run_seeded.py` | 固定种子执行器 |
| `logs/official_eval.log`、`logs/relags2_eval.log` | 两次评测的 `SCENE_RESULT` 行 |
| `logs/relags2_merge.log` | 重建管线日志 |

## 8. 仍然不能主张的

- 只覆盖 **figurines 单场景**（ramen / teatime / waldo 未重建，各需约 4 h 编码）。
- 0.9999 的余弦是"同一批输入 + 同一编码器"的必然结果，它验证的是**管线正确性**，
  不构成对新方法的证据。
- 发布权重 0.6329 与重建 0.6610 的差值由 3 条 query 支配（§5），样本量 56，
  **不足以讨论显著性**；任何"重建优于发布"的表述都不成立。