# 运行命令与复现等级

每个数字对应的命令、所需产物与耗时。复现等级定义见 `README.md` §7。

---

## 0. 统一环境（所有服务器命令的前缀）

```bash
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
REL=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
SAS=/root/rivermind-data/repro/deco-b123/m1_sas_20261029
export RELAGS_REPO=$REL HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$REL:$REL/ext:$REL/ext/spt/dependencies/grid_graph/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
```

三台机器的分工：

| 位置 | 用途 |
|---|---|
| 本机（Windows） | 离线重聚合、生成表格、检查口径 |
| `fj01`（4090） | LERF-OVS / LERF-MASK / ScanNet 的主实验 |
| `hn01` | 3D-OVS、LERF-MASK 证词 dump（CPU） |

SSH 用 `ssh fj01 '...'` / `ssh hn01 '...'`，不要开 ControlMaster（Windows Git Bash 下会失败）。
`fj01` 的配置带 RemoteForward，遇到端口冲突用 `ssh -o ClearAllForwardings=yes fj01`。

---

## 1. R0：本机可重算（分钟级，无需服务器）

```bash
python reports/PAPER_EXPERIMENTS_20261104/verify.py
python reports/PAPER_EXPERIMENTS_20261104/build_tables.py
```

`verify.py` 覆盖的 11 行 R0：

| 行 | 数字 | 输入 |
|---|---|---|
| O-01 | 0.6401 | `analysis/results/20261031/baselines/relags/*.csv` 与 `20261101/cnone_gate_tf/*_cmaskgate_none.csv` |
| O-02 | 0.6572 | 同上的 `c_iou` 列 |
| O-03 | 0.6582 | 同上的 `mask_iou` 列 |
| O-04/O-10/O-11 | 0.6711 / 0.6515 / 0.8275 | 重跑 `analysis/lerf_witness_certificate.py` |
| O-16 | 0.6751 | 三列的逐查询最大值再嵌套平均 |
| O-28/O-29/O-30 | 0.925 / 0.947 / 0.727 | `results/grounded_sam_20261104/*.json` |
| S-09…S-12 | 41.05/38.87 → 56.05/49.89 | `results/scannet_c1c2_ablation_20261103/c1c2_ablation.json` |
| S-33 | 0.5797 / 0.5568 / 0.5797 | `results/scannet_singlequery_20261104/scannet_singlequery_ap.json` |

单跑证书（它就是 O-04 的来源）：

```bash
$PY analysis/lerf_witness_certificate.py \
  analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_gate_tf \
  analysis/results/20261101/cnone_ev    analysis/results/20261101/witness \
  none 0.5 sc
```

预期最后几行：

```
MACRO main   0.6401
MACRO c      0.6572
MACRO mask   0.6582
MACRO cert   0.6711
cert fired on 19/208 (found 3/129, deep 6/25); small-view skips 2
cert-main mean +0.0313 CI95 [+0.0114, +0.0548]
```

---

## 2. R1：已有缓存，重跑评测

### 2.1 LERF-OVS 锚点与 C1 臂（P1）

```bash
for sc in figurines ramen teatime waldo_kitchen; do
  $PY $REPO/deco/lerf_ovs/eval_lerf_ovs.py --scene $sc \
    --source $SRC/$sc --model $MODEL/$sc --nag $MODEL/$sc/sai_nag.pt --gt_root $GT \
    --out <OUT>/$sc --csv <OUT>/$sc/per_query.csv \
    --center none --renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 \
    --sel gap --norm none --text-center none --tau 0.02 --text-agg lse
done
```

- `--center none` → **0.6401**（O-01）
- `--center image` → **0.6393**（O-21，173/208 逐位相同）
- `--center image --renorm 1` → **0.4686**（O-22）

注意：`analysis/lerf_calib/eval_lerf_ovs_orig.py` 是同一评测器的本地副本，参数一致。

### 2.2 LERF-MASK（P3）

```bash
$PY $REPO/deco/lerf_mask/eval_lerf_mask.py --scene <sc> \
  --source <lerf_mask>/<sc> --model <model>/<sc> --nag <model>/<sc>/sai_nag.pt \
  --gt_root <lerf_mask_gt> --out <OUT> --center none --renorm 0
```

`--center none` → 0.6800 / BIoU 0.6444；`--center image` → 0.7112 / 0.6655（M-02/M-03）。

### 2.3 ScanNet 统一算子（P5/P6）

```bash
export SCANNET_OUT=/tmp/deco_reg
$PY deco/scannet/score_scannet.py --renorm 0   # 58.55 pooled / 53.30 macro
$PY deco/scannet/score_scannet.py --renorm 1   # 57.85 pooled / 53.62 macro
$PY deco/scannet/aggregate.py --results /tmp/deco_reg
```

这条命令需要 shipped `clip_cache`（366 簇场）。若盘上是 386 簇，先看 `§5 已知缺口`。

### 2.4 renorm 机制探针

```bash
$PY deco/scannet/probe_renorm.py scene0000_00
```

预期 `V=1 argmax differs 0/366`、`V=4 differs 109/366 (29.78%)`、中位 margin `0.0034`。
这是 0.70 分差距的机制。

### 2.5 ScanNet 单查询 AP（S-33）

```bash
$PY analysis/scannet_singlequery_ap.py \
  --scenes scene0000_00,scene0347_00,scene0070_00 \
  --model_root /root/verify58_20261029/full10 \
  --gt <scannet_gt> --out results/scannet_singlequery_20261104 --alpha 1.0
```

---

## 3. R2：需要重新 dump / 编码（小时级）

### 3.1 LERF-OVS 训练-free 三选项（O-02/O-03 的输入）

```bash
bash analysis/run_cnone_dumps.sh      # center=none 的 cand / ev / segid 三份 dump
bash analysis/run_gate_cnone_tf.sh    # gateE-TF，pred 用 NAG root score
```

产物：`analysis/results/20261101/cnone_gate_tf/`、`cnone_ev/`。耗时约 1–2 h/场景。

### 3.2 证书稳健性（O-06/O-07/O-08/O-09）

```bash
# 5 帧 / 10 帧视角簇重采样
$PY experiments/cvpr27_8h_discovery/lerf_certificate_cluster.py \
  analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_gate_tf \
  analysis/results/20261101/cnone_ev analysis/results/20261101/witness none 0.5 sc "" _witness 5

# 跨场（center=image）
bash analysis/run_crossfield_image.sh
```

跨场产物：`analysis/results/20261101/{witness_image,gate_image,ev_image,segid_image}/`。

### 3.3 `a_g` 消融（O-12…O-15）

```bash
$PY experiments/cvpr27_8h_discovery/dump_ag.py          # 导出 a_g
$PY experiments/cvpr27_8h_discovery/build_n1_shifted.py \
  analysis/results/20261101 analysis/results/20261103_axis/n1 \
  analysis/results/20261103_axis/ag 1.0 a1z
$PY analysis/lerf_witness_certificate.py \
  analysis/results/20261101/cnone_gate_tf analysis/results/20261101/cnone_gate_tf \
  analysis/results/20261101/cnone_ev analysis/results/20261103_axis/n1/witness_ag none 0.5 sc
```

### 3.4 LERF-MASK 集合与证词（P4、M-07/M-08/M-09/M-10）

证词 dump 在 hn01（CPU，不需要 GPU）：

```bash
python3 experiments/cvpr27_8h_discovery/lm_witness_dump.py \
  --scene <sc> --model <model>/<sc> --nag <model>/<sc>/sai_nag.pt \
  --featdir <lerf_mask_features> --cands <cands.npz> --out <OUT> --views 30 --mincover 0.35
```

前缀曲线与增长判定：

```bash
python3 experiments/cvpr27_8h_discovery/lm_certificate_eval.py <witness.csv> <curve.log> 0.5
python3 experiments/cvpr27_8h_discovery/lm_grow_eval.py        <witness.csv> <curve.log> 0.5
```

前提：`observations_level3.npz` 必须存在；没有它证书跑不了（`16_LERF_MASK_SET_HEADROOM.md` §4）。

### 3.5 ScanNet C1/C2 在离线管线与 OpenSeg 上（S-09…S-12、S-29）

```bash
$PY analysis/scannet_c1c2_probe.py \
  --obs <obs.npz> --clip-dir <clip_dir> --os-dir <openseg_dir> --ply <gt.ply> \
  --gt <gt_dir> --nag <sai_nag_3.0.pt> --scene <sc> --out <OUT>

$PY analysis/scannet_c1c2_ablation.py --cache <cache.npz> --models <models> --out <OUT>
```

### 3.6 ScanNet 候选与分区上限（S-30/S-31）

```bash
$PY analysis/scannet_candidate_coverage.py \
  --scene <sc> --model <model> --gt <gt> --nag <nag> --out <OUT> --voxel 0.05 --min-points 200
```

### 3.7 Grounded-SAM 2D 探针（O-28/O-29/O-30）

```bash
python analysis/lerf_grounding_complementarity.py --out <OUT> --boxes <boxes.npz> --clip_csv <clip.csv>
python analysis/lerf_grounding_select.py           --out <OUT> --boxes <boxes.npz> --clip_csv <clip.csv>
```

模型：`/root/rivermind-data/part_grounding_20261101/model/grounding-dino-tiny`。
必须只取**检测器分数最高的单框**；多框再按 SAM 分数选会从 0.925 掉到 0.176。

---

## 4. R3：产物缺失，只能引用

| 行 | 缺口 | 影响 |
|---|---|---|
| S-02…S-08 | shipped `clip_cache` 对应 366 簇场，盘上是 386 簇 | 官方 58.55 锚点无法逐位复现；重建场给 56.05 |
| O-25 | 路线 B 在 hn01，未复核 | 0.6471 不可引用 |
| D-01…D-12 | 3D-OVS 全部产物只在 hn01；本地 `results/` 为空 | 所有 3D-OVS 数字只能引用报告 |
| M-12 | 检测桥（GroundedDINO + SAM）不在发布包里 | 0.9299/0.6606/0.5448 只能引用 |
| S-32 | 旧 OpenSeg 55.77 无对应产物 | 用 61.64 |

3D-OVS 的命令（在 hn01）：

```bash
python3 deco/3dovs_eval.py --mode oracle --scene bed        # 0.9594
python3 analysis/eval_3dovs_l0.py --scene bed --source <src> --model <model> \
  --nag <nag> --gt_root <gt> --out <OUT>                     # level-0 检索 0.5217
python3 analysis/audit_3dovs_decision.py --scene <sc> --source <src> --model <model> \
  --nag <nag> --gt_root <gt> --out <OUT> --resolution 4 --level_until 3 --tau 0.4 --pred_thresh 0.5
```

---

## 5. 已知缺口与坑

1. **`srem_tail.py` 被截断**：文件从 `gt_dir = ...` 开始，没有 import 与 argparse，不可运行。
2. **`REPRODUCE.md` 引用的 `run_axis.sh` 不存在**。真正的入口是 `analysis/run_*.sh`（38 个），
   它们调用的是 `deco/lerf_ovs/<script>.py`，不是 `analysis/<script>.py`——脚本要先部署到
   `$REPO/deco/lerf_ovs/`。
3. **20 个离线评测脚本硬编码 Windows 路径**
   `C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro`。结果不拷到那里，
   脚本会静默跳过输入而不报错。
4. **所有 `scannet_*.py` 硬编码服务器路径**（`REPO`、`REL`、`certified`、`verify58`）。
5. **`scannet_sam_pilot.py:36` / `scannet_proj_debug.py:36` 的相机约定是错的**（用了 `@ R.T`）；
   依赖像素级 GT 对齐的历史结论需复核。正确约定：`p_cam = (p - C) @ R`。
6. **`scannet_baseline_20261029/*.json` 报的是 scene-mean 19 类 49.89**，不是 40.36/40.79 锚点；
   用之前先确认读端口径。
7. **`lerf_pruneinclude` 的 +0.0240（CSV）与 +0.0194（PREREG 记录）不一致**；该模块未进主表，
   若要用需先复核。

---

## 6. 一键复现顺序（推荐）

```bash
# 1) 本机：先确认口径没坏
python reports/PAPER_EXPERIMENTS_20261104/verify.py

# 2) 本机：生成论文表
python reports/PAPER_EXPERIMENTS_20261104/build_tables.py

# 3) 服务器：若要重跑，按 §2.1 → §2.2 → §2.3 的顺序，每步先对锚点
#    锚点不中就不要读 Δ（PROTOCOLS.md §5）
```