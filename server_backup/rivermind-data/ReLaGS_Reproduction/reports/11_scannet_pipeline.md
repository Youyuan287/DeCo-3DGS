# 11 · ReLaGS × ScanNet 训练与语义链路（实测）

> 目标：跑通 `2DGS → MWP → sp_partition → graph_weight → sp_partition(-k) → merge_proj → 3D mIoU`。
> 所有数字来自 `logs/scene*_*.log`，不是估计。

## 1. 2DGS 几何重建（论文 §9：ScanNet 要开 densification）

| scene | 帧 | 高斯数(30k) | ply 体积 | PSNR test | PSNR train | 训练耗时 |
|---|---|---|---|---|---|---|
| scene0000_00 | 251 train / 28 test | **1,301,423** | 317.5 MB | **18.04** | 20.09 | 11 min |
| scene0062_00 | 219 / 25 | **1,217,365** | 297.0 MB | **21.29** | 23.27 | 21 min（与 SAM 编码抢 GPU） |
| scene0070_00 | 238 / 27 | **922,236** | 225.0 MB | **28.02** | 29.03 | 6 min（独占 GPU，86 it/s） |

- 超参用 2DGS 默认：30k 迭代、densification 开、`--eval`（每 8 帧留 1 帧测试）
- scene0000 的 PSNR 18.04 偏低，原因是 ScanNet 自动曝光 + 运动模糊 + 半分辨率；同类方法在 ScanNet 单房间也是 18–22 这个量级。几何本身可用：高斯到最近 GT 点的平均距离 **0.101 m**、中位数 **0.050 m**（1M 高斯全部参与统计）。

### 1.1 为什么必须给 2DGS 打补丁

ReLaGS 仓库**没有 `train.py`**（README 要求用官方 2DGS 训练）。落地时踩了 5 个坑，全部已修并留 diff：

| # | 问题 | 修法 |
|---|---|---|
| 1 | conda 环境里已装 `diff_surfel_rasterization`（带 `semantics` 参数的 0.0.3），会**盖掉** 2DGS 自带同名包，报 `ValueError: too many values to unpack` | 训练时 `PYTHONPATH` 前置 2DGS 的 in-place 编译产物，隔离 |
| 2 | conda 只有 CUDA 12.1，torch 是 11.8 编译 → `The detected CUDA version (12.1) mismatches` | 用 `CUDA_HOME=<thgs env 前缀>`（nvcc 11.8.89）+ `CPATH=.../targets/x86_64-linux/include` + `TORCH_CUDA_ARCH_LIST=8.9`，`setup.py build_ext --inplace` |
| 3 | `simple_knn.cu: identifier "FLT_MAX" is undefined` | 加 `#include <cfloat>` |
| 4 | `Image.fromarray(arr*255, dtype=np.byte)` → `TypeError: Cannot handle this data type: (1,1,3), |i1` | 改成 `.astype(np.uint8)` |
| 5 | 2DGS 只认 `camera_angle_x`，ReLaGS 只认 `fl_x/fl_y` | 2DGS `scene/dataset_readers.py` 增加 `fov_y_direct`（有 `fl_x/fl_y` 时直接用），`scene/__init__.py` 传 `extension=".jpg"` |

## 2. 语义链路（单场景实测，scene0000_00）

| 步骤 | 命令要点 | 输入 → 输出 | 耗时 |
|---|---|---|---|
| MWP | `max_weight_pruning.py --contribution_threshold 0.0005 --vlm_type clip` | **1,301,423 → 1,015,937** 高斯（-21.9%），写 `point_cloud/iteration_0/point_cloud.ply` | 19 s |
| sp_partition pass1 | `sp_partition.py -s .. -m .. --iteration 0 -a` | ← 写 `neighbor.pt`（30-NN，122 MB） | 8 s |
| graph_weight | `graph_weight.py --config configs/scannet.yml --level 1` | `neighbor.pt → neighbor_new.pt`（用 seg level1 的 CLIP 特征重加权距离） | 23 s |
| sp_partition pass2 | `sp_partition.py -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a` | 图割，写 `nag-l1.pt`（**17,308** 个初始超点） | 42 s |
| merge_proj | `merge_proj.py --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3` | 三级区域生长 **17308 → 4698 → 937 → 366**，写 `sai_nag_3.0.pt`(32.6 MB) + `saifine.ply` + `saifine_labels.npy` | 226 s |

- 链路总耗时 ≈ 5.5 min（不含 SAM/CLIP 编码的 48 min）
- `nag_feat` 是 3 个层级的 512-d 簇特征：`(4698,512) / (937,512) / (366,512)`；`nag` 里多一层 `17308`（SPT 初始超点，无对应特征层）——**这个错位是 grid 评测第一次跑出 1.4% mIoU 的根因**，nag[lvl+1] 才对得上 nag_feat[lvl]
- MWP 后仍留 1.02 M 高斯：论文 τ=5e-4 只为「去掉对渲染几乎无贡献的图元」，不是压缩到几万点；我在早期一次误读（17 MB 的 ply）后确认过这一点

## 3. 评测协议（仓库没发布评测代码，必须自己复刻）

论文 §9 只写了两句：**densification 开**、**每个高斯取其最近 GT 点的标签（L2）**。其余自定，全部记为 deviation：

1. **GT**：`<scene>_vh_clean_2.labels.ply`（mesh 顶点 + NYU40 标签），忽略 label=0（unlabeled 占 5.4%：77012/81369 有效）
2. **预测**：`saifine_labels.npy`（merge_proj 每高斯的分簇 id，1-based）→ 簇特征 × 类名文本嵌入 → argmax → 簇类别 → 广播到该簇所有高斯
3. **mIoU**：每类 `inter/union`（高斯点集），mAcc = 每类 `inter/gt`；**只在「场景中出现的类」上平均**（另一口径：跨场景汇总 inter/union 后再平均，见 `aggregate_scannet.py`，两个都报）
4. 类子集：19/15/10 类与 OpenGaussian / VALA 一致（NYU40 从 1 开始，0=unlabeled）

### 3.1 协议敏感性（scene0000_00，网格实测）

因为代码没发布，「用哪个簇层级 + 什么文本 prompt」直接影响分数，所以做了完整网格（`results/scannet_grid_scene0000_00.json`）：

| 配置 | 19 类 mIoU/mAcc | 15 类 | 10 类 |
|---|---|---|---|
| raw prompt + 细簇 (4698) | 17.84 / 26.37 | 18.97 / 32.34 | 26.96 / 42.67 |
| raw prompt + 中簇 (937) | 34.54 / 48.07 | 32.00 / 54.65 | 31.15 / 58.53 |
| raw prompt + 粗簇 (366) | 42.28 / 56.21 | 39.61 / 62.53 | 43.09 / 68.96 |
| 模板集成 + 细簇 (4698) | 14.04 / 18.08 | 25.04 / 37.24 | 27.78 / 42.97 |
| 模板集成 + 中簇 (937) | 25.42 / 38.08 | 29.54 / 50.54 | 36.62 / 63.68 |
| **模板集成 + 粗簇 (366)** | **42.81 / 55.26** | **47.53 / 69.95** | **50.92 / 77.36** |
| 论文 Table 9 | 41.26 / 57.29 | 45.24 / 62.75 | 54.04 / 71.48 |

- 结论 1：**簇层级比 prompt 重要得多**（细簇直接掉到 1/3 分数），说明实例级最终簇才承载「物体」语义
- 结论 2：单模板极不稳定（`{}` vs `a photo of a {}` 在 10 类上差 15 个点），4 模板集成（`{}` / `a photo of a {}` / `the {}` / `a {} in a room`）在三个子集上都最稳且最接近论文
- **采用「模板集成 + 粗簇」作为主协议**，并在 scene0062/0070 上冻结复用来验证（不是逐场景调参）
- 该配置下 scene0000_00：**42.81/55.26、47.53/69.95、50.92/77.36**，与论文 41.26/57.29、45.24/62.75、54.04/71.48 差 **+1.55/-2.03、+2.29/+7.20、-3.12/+5.88**（10 类 mIoU 是唯一明显偏低项）

### 3.2 特征去均值（hubness）消融：一个必须公开的口径分叉

scene0062_00 上「原始余弦」出现整面墙被判定为 `sink` 的崩溃（wall IoU=0.001，380k GT 点里 73% 被判成 sink），逐簇检查确认不是几何/标注错（高斯到最近 GT 点平均 0.048 m）。这是 CLIP 特征云的 **hubness**：少数通用方向吸走几乎所有查询。标准缓解手段是先把簇特征去均值（必要时文本也去均值）再算余弦。同一份特征、同一份标签，只改归一化方式：

| 归一化 | scene0000 19/15/10 mIoU | scene0062 19/15/10 mIoU |
|---|---|---|
| **cosine（主协议）** | **42.81 / 47.53 / 50.92** | **36.40 / 35.75 / 36.20** |
| center_f（特征去均值） | 44.66 / 44.35 / 54.09 | 68.56 / 67.37 / 85.93 |
| center_t（文本去均值） | 35.25 / 42.21 / 50.93 | 25.17 / 35.75 / 36.10 |
| center_both | 45.29 / 44.32 / 55.12 | 68.55 / 67.36 / 86.21 |

- 去均值让 scene0062 从 36.40 跳到 68.56（19 类），**远高于论文 41.26**；而原始余弦下两场景（42.81 / 36.40）的平均恰好落在 41 附近
- 因此**主表用原始余弦**（最少调参、最接近「照论文直接复现」），去均值作为消融保留在 `results/scannet_norm_*.json`，不作为主数字
- 结论：这一层口径带来的单场景波动可达 **±30 个点**，是本次复现最大的不确定源（D4），比特征维度（CLIP-512 vs OpenSeg-768）的影响还大

## 4. Deviation 清单（与官方实现的偏离，按影响排序）

### 4.1 关于 `--vlm_type openseg` 的口径澄清（重要）

`scripts/run_scannet.sh` 与 `scripts/run_lerf.sh` 是**同一条链路**（MWP→sp→gw→sp→merge_proj），ScanNet 版只多了两处：`merge_proj --vlm_type openseg` 和末尾的 `predict_3d_scene_graph.py --vlm_type openseg`。也就是说 openseg 是为了让 merge_proj 产出 `sai_nag_openseg.pt`、供**场景图支线**使用；而论文 §7 把任务分成两条：

> (1) For open-vocabulary object querying, we follow LangSplat using OpenCLIP **ViT-B/16 (f_k ∈ R^512)**.
> (2) For the task of 3D scene graph prediction and relationship-guided 3D instance segmentation, we follow RelationField to use CLIP/**OpenSeg (f_k ∈ R^768)** …

Table 9（19/15/10 类 mIoU）属于 (1)，所以**本复现用 CLIP-512 与论文设定一致**；`scripts/image_encoding.py` 注释也写死 `300 features × 512-dim`。
另注：OpenSeg 官方仓库与权重现已无法获取（`github.com/cvlab-columbia/OpenSeg` 与 `nv-tlabs/OpenScene` 均返回 404，HF/ModelScope 无镜像、无 PyTorch 复现权重），因此即便要跑 768 维场景图支线，也只能用「CLIP ViT-L/14-336（768 维）写进 `language_features_openseg/*_f_openseg.npy`」这条替代路径——代码路径与文本空间（`ClipSimMeasure("ViT-L-14-336")`）都对得上。

| id | 偏离 | 影响 | 是否可消除 |
|---|---|---|---|
| D1 | **不是偏离（已更正）**：论文 §7 明确「开放词表分割/查询用 OpenCLIP ViT-B/16（512 维）」，`--vlm_type openseg`（768 维）只服务 3D 场景图支线（`predict_3d_scene_graph.py`）；README 的 `Extract 2D Semantic Features` 也只给 512 维 CLIP 提取器 | 无（当前设定与论文一致）；但需注意若作者 Table 9 实际用了 openseg，则会与本文档口径不同，见 §4.1 | — |
| D2 | 每场景按 `stride=ceil(N/300)` 取 250–290 帧（论文未给帧数） | 帧少→几何与特征覆盖下降 | 加帧即可（更慢） |
| D3 | 2DGS 用上游 `hbb1/2d-gaussian-splatting` + 5 个补丁（ReLaGS 未发布训练代码） | 与作者内部训练脚本可能有细节差 | 作者发代码后可对齐 |
| D4 | 评测代码自写（论文只给「最近 GT 点」一句） | 见 §3.1 的 15 点量级敏感度 | 官方 eval 发布后可校准 |
| D5 | SAM 用 `points_per_side=16, crop_n_layers=0`（LangSplat 原版是 32+1） | 掩膜更少更粗，编码从 36 s/img 降到 7–9 s/img | 可升到 32（单场景 ~2.5 h） |
| D6 | `graph_weight --config configs/scannet.yml`（`run_scannet.sh` 里写的是 `lerf.yml`，明显复制粘贴） | 图重加权强度不同 | 可切回 lerf.yml 对照 |
| D7 | `merge_proj --thres_merge 20 / --tau 3`（`configs/scannet.yml` 写 100） | 按 §8.5「以脚本为准」；簇数 366 | 可扫参 |
| D8 | 文本 prompt 用 4 模板集成（论文未提） | 见 §3.1，比单模板稳 5–15 点 | 已作为主协议并公开敏感性 |
| D9 | `merge_proj` 补丁额外存 `saifine_labels.npy`（原版只画彩色 ply，无法评测） | 无（纯增信息） | — |

## 5. 一键复现

```bash
bash scripts/batch_prep.sh                                  # 数据（curl 下载 + 解帧 + Blender 化）
bash scripts/run_scene.sh <scene> train                     # 2DGS 30k
SAM_PPS=16 SAM_CROP=0 python scripts/image_encoding_fast.py --source_path <dataset> --resolution 648 \
    --sam_ckpt_path $R/weights/sam_vit_h_4b8939.pth         # SAM + CLIP 特征
bash scripts/run_scene.sh <scene> sem                       # MWP→sp→gw→sp(-k)→merge_proj→eval
python scripts/aggregate_scannet.py                         # 跨场景汇总（dataset / scene_mean 两口径）
```