# 3D-OVS × ReLaGS(DeCo-3DGS) 全流程复现

> **范围**：这是本仓库的**扩展实验**，不是论文表格的复现。
> DeCo 只发布 ScanNet-10(K19)、LERF-Mask、LERF-OVS 三个结果，**没有 3D-OVS 条目，
> 也没有 3D-OVS 评测代码**。这里做的是：在 ReLaGS 栈上补出 3D-OVS 读出，
> 把 DeCo 的两个算子接上去，并给出 5 场景 × 7 臂的完整数字与失败分析。
>
> 数据：官方 3D-OVS Drive 全部 5 个 eval 场景；训练、编码、层级、合并、评测**全部从原始数据跑起**，
> 未复用任何预训练语义场。

---

## 1. 结果速查

主口径：区域级（超点）读出，每场景 5 个标注视角，`topk=5`、`level_until=3`、`filter_small=50`、
`--cand-mode gap2`、掩膜阈值 0.5。数字为 mIoU(%)，来源 `results/v2/<scene>_<arm>_summary.json`。

| 场景 | base | +C1 | +C2 | +C1+C2 | C1+renorm | C1+renorm+C2 | base topk=10 | ReLaGS 论文 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bed   | 56.36 | 56.37 | 49.30 | **62.83** | 49.12 | 11.19 | 56.36 | 94.4 |
| bench | 75.83 | 75.61 | 57.16 | **84.59** | 60.31 | 39.42 | 61.81 | 91.2 |
| lawn  | 73.18 | 69.37 | 69.37 | 69.37 | 53.24 | 33.20 | 73.18 | 94.8 |
| room  | 78.12 | 78.12 | 64.17 | 64.17 | **86.19** | 16.56 | 78.12 | 86.9 |
| sofa  | 66.75 | 65.78 | 65.80 | 65.76 | 52.93 | 41.65 | 66.75 | 92.5 |
| **均值** | **70.05** | 69.05 | 61.16 | 69.34 | 60.36 | 28.40 | 67.24 | **91.96** |

臂定义：

| 臂 | 参数 | 含义 |
|---|---|---|
| `base` | `--center none --text canon --renorm 0` | 不干预 |
| `c1` | `--center image --renorm 0` | 图像侧去公共模态（上游默认写法） |
| `c1r` | `--center image --renorm 1` | DeCo 论文里"有效"的那版去均值（归一化回原范数） |
| `c2` | `--text template --tau 0.02` | 文本侧模板 τ-LSE（4 个模板取 log-mean-exp） |
| `c1c2` / `c1rc2` | 两者叠加 | |
| `base_t10` | `--topk 10` | 候选截断上限消融 |

**一句话结论**：在本协议下 DeCo 两个算子在 3D-OVS 上是**洗白**（wash）——
`+C1` 在 bed/bench/room 上到小数点后 2 位不变（与 DeCo `PROTOCOLS.md` 所述
"单 query 去均值 + 不 renorm 会互相抵消"一致），`+C1+C2` 在 bed/bench 上 +6.5/+8.8
但在 room 上 −14.0，均值 −0.7。**没有任何一臂在 5 场景均值上超过 base。**

---

## 2. 环境

| 项 | 值 |
|---|---|
| 机器 | `root@hn01-ssh.gpuhome.cc -p 30833`，128 核 / RTX 4090 24 GB / CUDA 12.1 |
| 代码根 | `/root/rivermind-data/deco`（2DGS 训练）+ `deco/DeCo-3DGS`（语义管线） |
| 管线环境 | `/root/rivermind-data/ReLaGS_Reproduction/venv`：torch 2.2.0+cu118、PyG 2.3.0、pytorch3d、FRNN、`segment_anything`、`open_clip` |
| 训练环境 | `/root/rivermind-data/deco/venv`：torch 2.2.1+cu121，干净的 `diff_surfel_rasterization` + `simple_knn` |
| CLIP | `ViT-B-16 laion2b_s34b_b88k`（512 维） |
| SAM | `sam_vit_h_4b8939.pth` |

```bash
B=/root/rivermind-data/deco
R=/root/rivermind-data/ReLaGS_Reproduction
V=$R/venv
export PYTHONPATH=$B/DeCo-3DGS/ext:$B/DeCo-3DGS:$B/DeCo-3DGS/deco:\
$R/ReLaGS/ext/spt/dependencies/grid_graph/python/bin:\
$R/ReLaGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin
export HF_HUB_OFFLINE=1
```

两个环境不能混用：训练环境的 `diff_surfel_rasterization` 必须返回 3 个值，
而 ReLaGS 环境里那个被改成返回 5 个（多出 `out_sem`/`max_contribution`），
混用会在 `train.py` 报 `too many values to unpack (expected 3)`。

## 3. 数据

官方 Drive 的 5 个 eval 场景，`database.db` 跳过，逐文件做了 JPEG EOI / PNG IEND 完整性校验（0 损坏）。

| 场景 | 图像 | 标注视角 | 类别数 |
|---|---:|---:|---:|
| bed | 37 | 5 | 6 |
| bench | 36 | 5 | 6 |
| lawn | 36 | 5 | 6 |
| room | 31 | 5 | 6 |
| sofa | 28 | 5 | 6 |

目录：`images/*.jpg`（4096×3072）、`segmentations/{classes.txt, <view>/<class>.png}`、
`sparse/0/*.bin`、`poses_bounds.npy`。

### 3.1 两处必须的数据修正

1. **相机模型**。3D-OVS 是 `SIMPLE_RADIAL`（例：sofa `f=2925.54, cx=2048, cy=1536, k1=-0.0175`），
   而 2DGS 的 `colmap_loader` 只接受 `PINHOLE/SIMPLE_PINHOLE`，直接跑会
   `AssertionError: Colmap camera model not handled`。
   `scripts/to_pinhole.py` 就地重写 `cameras.bin`（备份 `.orig`）——**只改相机模型，
   不动图像与掩膜**，因此 GT 掩膜与渲染仍在同一像素栅格上。代价：丢弃 k1 的径向畸变，
   4096 px 宽度下角落残差约 34 px。
2. **编码分辨率必须等于训练分辨率**。`scripts/image_encoding.py` 默认把图缩放到 1080p 再编码，
   而训练是 `-r 4`（1024×768），于是 `merge_proj` 的射线投射（`seg_map` 1080×1440）
   与相机内参（768×1024）错位，层级直接塌成 7 个根簇、98% 高斯挤在 1 个簇里，
   检索全打到 floaters，mIoU ≈ 0.9%（这就是本项目最早那个"0.0088"）。
   `scripts/prep_r4.py` 先造一份 1024×768 副本再编码，之后把 `language_features`
   搬回 `-s` 指向的训练目录。自检：`seg_map` 形状 `(4,768,1024)`、每图 130+ 掩膜、
   掩膜嵌入两两余弦均值 0.52。

## 4. 单场景五步管线

以 `SCENE=bench` 为例，每步可单独重跑，`set -e` 保证失败即停。

```bash
# 0) 相机 -> PINHOLE
$V/bin/python $B/to_pinhole.py $B/data/3DOVS/$SCENE/sparse/0/cameras.bin

# 1) 2DGS 重建（30k 迭代，训练环境）
cd $B/2dgs && PYTHONPATH=$B/2dgs/submodules/simple-knn $B/venv/bin/python train.py \
    -s $B/data/3DOVS/$SCENE -m $B/output/3dovs/$SCENE --iterations 30000 -r 4

# 2) SAM + CLIP 编码（最慢，约 3–7 分钟/图；必须在训练分辨率上）
$V/bin/python $B/prep_r4.py $SCENE
cd $B/DeCo-3DGS && $V/bin/python scripts/image_encoding.py \
    --source_path $B/data/3DOVS_r4/$SCENE --sam_ckpt_path $R/weights/sam_vit_h_4b8939.pth
rm -rf $B/data/3DOVS/$SCENE/language_features
mv $B/data/3DOVS_r4/$SCENE/language_features $B/data/3DOVS/$SCENE/language_features

# 3) 语义层级（剪枝 -> 超点 -> 图权重 -> 二次划分）
cd $B/DeCo-3DGS
$V/bin/python max_weight_pruning.py -s $S -m $M --contribution_threshold 0.0005
$V/bin/python sp_partition.py      -s $S -m $M --iteration 0 -a
$V/bin/python graph_weight.py      --iteration 0 -s $S -m $M --config ./configs/3dovs.yml --level 1
$V/bin/python sp_partition.py      -s $S -m $M --iteration 0 -k neighbor_new.pt \
    --pcp_regularization 0.3 --pcp_spatial_weight 0.2 -a

# 4) 区域合并 + 特征投射
$V/bin/python merge_proj.py -s $S -m $M \
    --thres_connect 0.9,0.7,0.7 --thres_merge 100 --tau 3 --iteration 0
# -> $M/sai_nag_3.0.pt   (nag: 4 层标签；nag_feat: 3 层 512 维)
```

注意产出文件名带 τ：`sai_nag_3.0.pt`（**不是** `sai_nag.pt`）。`--thres_merge 100` 与
`configs/3dovs.yml` 一致；`graph_weight.tau: 0.5`、`pcp_regularization: 0.3`、
`pcp_spatial_weight: 2e-1` 均取仓库配置值。

### 4.1 三处上游兼容性修复（不改算法，见 `patches/`）

| 文件 | 问题 | 修复 |
|---|---|---|
| `ext/spt/data/data.py` | 以方法调用 `data.keys()`，PyG 2.3.0 里 `keys` 是属性 | 加 `_CallableKeys(list)` 垫片 |
| `merge_proj.py` | `view.semantic["sem"]` 是 fp16 | `.to(device).float()` |
| 仓库根 | pyrootutils 找不到项目根 | `touch .project-root` |

另外 `grid_graph` / `cp_d0_dist` 两个 C 扩展用 `ReLaGS/ext/spt/dependencies/*/python/bin`
里已编译好的版本，加进 `PYTHONPATH` 即可（否则 `spt/utils/instance.py:29` 报
`ModuleNotFoundError: No module named 'grid_graph'`）。

## 5. 评测协议

`scripts/3dovs_eval.py`（本仓库新增，上游没有）。与 THGS / LangSplat 的 3D-OVS 读出同构：

1. 遍历 `segmentations/` 下的标注视角目录，逐类读 PNG；
2. 每类 query → 根层超点候选 → `topk=5` → 层级下降（`level_until=3`）→ 最大相似度落差截断
   → 选出高斯集合；
3. 渲染这些高斯的语义通道（20 维硬编码通道，`(N,20)`），把预测**上采样到 GT 栅格**；
4. `pred > 0.5` 与 `gt > 0` 求 IoU；
5. 场景 mIoU = 先逐视角对类别平均、再对视角平均；总体 = 对场景平均。

```bash
cd $B/DeCo-3DGS && $V/bin/python deco/3dovs_eval.py \
  --scene bench --source $B/data/3DOVS/bench --model $B/output/3dovs/bench \
  --nag $B/output/3dovs/bench/sai_nag_3.0.pt --gt_root $B/data/3DOVS \
  --out $B/results/v2 --csv $B/results/v2/bench_base.csv \
  --center none --text canon --renorm 0 --topk 5 --cand-mode gap2 --tag base
```

`--cand-mode`：`gap`（上游原版，有 pad-0 空集 bug）/ `gap2`（默认，已修）/
`thresh` / `all` / `top1`；`--mode`：`prompt`（文献口径）/ `multi` / `oracle`。

## 6. 一个必须知道的协议缺陷（本项目最早的两个"假结果"）

**bug A：候选截断的 pad-0 空集。** 上游 `nag_data.py` 的 `_filter_by_similarity_gap`
构造 `sim_deltas = [0] + [...]` 后 `.index(max(...))`；当所有候选相似度相等时最大落差位
落在前导 0 上 → 返回空集 → 该类 IoU 恒为 0。修法是 `--cand-mode gap2`（去掉前导 0）。

**bug B：sofa 的 "0.0088" 是编码分辨率错位**（见 §3.1 第 2 条），不是模型能力问题。
定位过程：RGB 渲染对 GT 的 PSNR 33.27/31.87 dB（相机与训练都对），语义渲染全 1 张量
均值 0.999（渲染通路正常），但 `sai_nag` 的根层 1 个簇覆盖 98% 高斯、其余 6 个簇
在所有视角下渲染值**恰好为 0** —— 说明簇-高斯映射与像素对齐无关，是"特征错位"，
最终追到 1080p vs 768p。

## 7. 为什么和论文的 94.4/91.2 差 22 个点（重要，别误读）

**不是训练问题，也不是 DeCo 的问题，是读出协议的层级不同。**

- 我们的读出是**区域级**：查询 → 超点候选 → 二值选择 → 渲染 → 阈值化。
- 文献（LangSplat / ReLaGS / THGS）的 3D-OVS 是**像素级语言特征**读出：逐像素 CLIP 相似度。

在同一渲染器、同一 0.5 阈值、同一 GT 处理下做**oracle 选择**（把 GT 投影到超点上，
只在选择步骤作弊，其余不变）得到：

| | bed | bench | lawn | room | sofa | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| oracle 上限 | 95.94 | 95.61 | 97.19 | 93.04 | 94.20 | **95.20** |
| 我们的 base | 56.36 | 75.83 | 73.18 | 78.12 | 66.75 | 70.05 |

oracle 只改"选哪些簇"，就能到 95.20 —— 说明检索/渲染链路本身没问题，
22 个点全在**选择**这一步。机制上：每个场景恰有**一个大面积低对比类塌到 0**
（bed `white sheet` 0.000、lawn `green lawn` 0.000、room `wood wall` 0.000、
sofa `grey sofa` 0.001、bench `wood` 0.295），其余 5 类都在 0.83–0.98。
6 类里 1 个 0 就拉掉 ≈14 个场景点，正好是这个 gap。原因：这些类在根层只有
29 个超点中的少数小簇（lawn 实测 115/228/274/787/1291 个高斯，合计约 0.2% 高斯）
被 CLIP 排进 top-5，渲染后每像素混合权重 <0.5，二值化后整幅为空。

**因此**：

- 可以主张：本协议下 7 臂的**相对差异**可信（C1(renorm=0) 的恒等性被逐位复现）；
- 不可以主张：我们的绝对 mIoU 与论文数字可比 —— 两者不同层。
- 文献列取的是 5 个场景值的均值 **91.96**；任务书里说的 "ReLaGS 95.4" 与这 5 个值
  不自洽（94.4/91.2/94.8/86.9/92.5 平均是 91.96）。
- `room` 的修正标注上游未发布，任何 room 数字（我们的或他们的）都不严格可比。

## 8. 复现这条结果的最短路径

```bash
bash $B/run_3dovs_all.sh              # 5 场景全跑；已有结果自动跳过
bash $B/run_3dovs_all.sh sofa lawn    # 只跑指定场景
DRY=1 bash $B/run_3dovs_all.sh        # 只打印计划
FORCE=1 bash $B/run_3dovs_all.sh sofa # 强制重做
```

单场景从零（约 40 min 训练 + 2–3 h 编码 + 15 min 管线）：

```bash
SCENE=bed
bash $B/run_train.sh $SCENE && bash $B/run_enc.sh $SCENE \
  && bash $B/run_stage14.sh $SCENE && bash $B/run_stage5.sh $SCENE \
  && bash $B/run_all_arms_v2.sh $SCENE
```

## 9. 如何核验本目录的数字

```bash
python - <<'PY'
import json, glob
rows={}
for p in glob.glob('results/v2/*_summary.json'):
    d=json.load(open(p)); sc,arm=p.split('/')[-1][:-len('_summary.json')].split('_',1)
    rows.setdefault(sc,{})[arm]=round(d['miou']*100,2)
for sc in sorted(rows): print(sc, rows[sc])
PY
```

`results/v2/*_summary.json` 的 `miou`/`flat_miou` 都是 0–1 小数（表里乘了 100）。
每个数字都能对上 `logs/armsv2_<scene>.log` 里的 `SCENE_RESULT <scene> mIoU=... queries=30 views=5`
（每场景 7 条）；本次核验 35 个 summary 与 35 条日志行**逐一相符**。

## 10. 目录

```
3dovs/
├── README.md            本文件
├── scripts/             to_pinhole.py, prep_r4.py, aggregate.py, fetch_3dovs.py,
│                        run_{train,enc,stage14,stage24,stage5,all_arms_v2,eval_v2,scene}.sh,
│                        master.sh, sched5.sh, build_2dgs.sh, 3dovs_eval.py
├── configs/3dovs.yml    上游 3D-OVS 配置（未改）
├── results/v2/          35 个 <scene>_<arm>_summary.json + 同名逐 query CSV
├── results/v3/          oracle 上限对照（*_oracle_*）
└── logs/                armsv2_<scene>.log（各 7 条 SCENE_RESULT）、diag2.log（oracle 定位）
```

## 11. 已知局限

1. **区域级读出 ≠ 文献的像素级读出**，绝对数字不可比（§7）。
2. 相机模型改写丢弃了 k1 径向畸变，角落残差 ≈34 px。
3. 未做多 seed：几何来自 30k 训练，单 seed；层级管线的随机性来自 `merge_proj` 的
   `torch.normal`，3D-OVS 侧未做种子扫描（LERF-OVS 侧做过，见
   [`../audits/figurines_feature_encoding_20261031/`](../audits/figurines_feature_encoding_20261031/)）。
4. DeCo 论文没有 3D-OVS 数字，因此本表**无法与作者结果对照**，只能与文献基线对照。
5. `room` 标注未修正，`lawn` 的 94.8 论文标注为 "*"（修正标注未发布），均不可逐位对齐。