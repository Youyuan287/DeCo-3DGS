# ScanNet-10：C1/C2 算子消融与官方锚点复现记录

日期：2026-11-03　　机器：fj01（RTX 4090）
脚本：`analysis/build_scannet_cache.py`、`analysis/scannet_c1c2_ablation.py`
产物：`results/scannet_c1c2_ablation_20261103/`、`results/scannet_repro_20261103/`

---

## 0. 一句话结论

用官方 `score_scannet.py` 的 bank 代码在**当前盘上的场**（386 簇 / 1,012,438 高斯）上跑通了七种配置；
**C2 单独的增益与仓库锚点一致（+7.6 vs +7.8），C1 单独的增益差得很远（+2.3 vs +14.7），交互项符号相反**。
仓库锚点（58.55 / 57.85）**无法从现有产物复现**，缺的是产生 366 簇那次 `merge_proj` 的场，不是命令。

---

## 1. 复现目标（锚点，来自仓库 `deco/RESULTS.md` + `deco/REGRESSION.md`）

ScanNet-10，K19，dataset-pooled，**除数 18**（bookshelf 在十个场景里没有 GT 点）。

| 变体 | pooled mIoU |
|---|---:|
| ReLaGS 论文 Table 9 | 41.26 |
| 本仓库复现 | 40.79 |
| + C1（image side only） | 53.67 |
| + C2（templates only） | 56.32 |
| + C1 + C2，`--renorm 1` | 57.85 |
| **+ C1 + C2，`--renorm 0`（默认）** | **58.55** |

`RESULTS.md` §3 的 C1×C2 2×2（scene-mean，renorm 1）：

| | bare | lexicon + templates |
|---|---:|---:|
| image: raw | 39.02 | 46.82 |
| image: centred | 53.67 | 57.85 |

**算子定义**（`deco/deco_ops.py`）：

- **C1** = `center_np(feats)`：对每个层级的特征矩阵**沿区域轴**减去均值（`f - f.mean(axis=0)`），
  作用在最终 `nag_feat` 上。`renorm=1` 时再补一次 `l2n`。
  ⚠️ 不是"逐视角 mask 中心化"——那是另一个算子（VCLS 一类），实测结论完全不同。
- **C2** = 文本侧：`one_word`（裸类名）vs `both_lse`（4 模板 + 手写词表，τ-LSE 边缘化）。

---

## 2. 环境与前置产物

```bash
export RIVERMIND_ROOT=/root/deco_reg2        # 本次临时 root，见 §3.1
export HF_HUB_OFFLINE=1
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python
CODE=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
```

| 角色 | 路径 | 状态 |
|---|---|---|
| 场景模型（`sai_nag_3.0.pt`、`point_cloud/iteration_0`） | `/root/rivermind-data/output/scannet/<scene>/` | ✅ 存在，**386 簇 / 1,012,438 高斯** |
| GT | `repro/deco-b123/scannet_v2_certified_20261029/<scene>/export/<scene>_vh_clean_2.labels.ply` | ✅ |
| 官方 scorer / aggregate | `$CODE/deco/scannet/{score_scannet,aggregate}.py` | ✅ |
| 官方 `clip_cache` | `repro/deco-3dgs-38620a7d/{recovery,projection_ablation}/clip_cache/` | ⚠️ 存在，但对应 **1,015,937 高斯 / 366 簇**，与现有 nag 不配对 |

---

## 3. 运行指令

### 3.1 搭临时 root（不改动 `rivermind-data`）

```bash
mkdir -p /root/deco_reg2 /tmp/deco_reg2
ln -sfn /root/deco_reg_root/c2        /root/deco_reg2/clip_cache   # 见 §3.2 重建的缓存
ln -sfn /root/rivermind-data/output   /root/deco_reg2/output
ln -sfn /root/rivermind-data/scripts  /root/deco_reg2/scripts
```

### 3.2 用当前场重建缓存（官方缓存的 N 对不上，必须重建）

```bash
cd /root/rivermind-data
for s in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 \
         scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  $PY scripts/build_scannet_cache.py --scene $s --model output/scannet/$s \
     --gt repro/deco-b123/scannet_v2_certified_20261029/$s/export/${s}_vh_clean_2.labels.ply \
     --out /root/deco_reg_root/c2
done
```

缓存约定（与 shipped 缓存一致）：`gt` = 每高斯最近 GT 点标签，`keep` = 全 True，`lab` 仅占位。

### 3.3 官方 scorer（得到 §4 表 B）

```bash
cd $CODE
for r in 0 1; do
  RIVERMIND_ROOT=/root/deco_reg2 SCANNET_OUT=/tmp/deco_reg2 HF_HUB_OFFLINE=1 \
    $PY deco/scannet/score_scannet.py --renorm $r
done
RIVERMIND_ROOT=/root/deco_reg2 HF_HUB_OFFLINE=1 \
  $PY deco/scannet/aggregate.py --results /tmp/deco_reg2
```

### 3.4 C1×C2 消融（得到 §4 表 A）

```bash
cd /root/rivermind-data
HF_HUB_OFFLINE=1 $PY scripts/scannet_c1c2_ablation.py \
  --cache /root/deco_reg_root/c2 --models output/scannet \
  --out results/scannet_c1c2_ablation_20261103
```

该脚本 `exec` 出 `score_scannet.py` 在 `def main():` **之前**的源码取常量（`NAMES/PROMPTS/TAU/LEX/classes`），
所以模板、词表、τ 与官方逐字一致；只把 C1 从"必然施加"改成开关。

---

## 4. 结果

### 表 A · C1×C2 消融（本次，pooled / scene-mean）

| | C2 = off（`one_word`） | C2 = on（`both_lse`） |
|---|---:|---:|
| **C1 = off** | **41.05 / 38.87** | 48.62 / 47.33 |
| **C1 = on, renorm 0** | 47.60 / 41.13 | **56.05 / 49.89** |
| **C1 = on, renorm 1** | 47.60 / 41.13 | 56.59 / 51.10 |

### 表 B · 官方 scorer 在重建缓存上的结果（both_lse）

| renorm | pooled mIoU | pooled mAcc | macro mIoU |
|---|---:|---:|---:|
| 0 | 56.05 | 68.24 | 49.89 |
| 1 | 56.59 | 68.72 | 51.10 |

子集（scene-mean）：renorm 0 → K19 49.89 / K15 49.15 / K10 51.70；renorm 1 → 51.10 / 50.06 / 53.47。

### 表 C · 文本阶梯（`Xc = center_np`，scene-mean）

| text bank | renorm 0 | renorm 1 | 锚点（r0） |
|---|---:|---:|---:|
| one_word | 41.13 | 41.13 | 42.53 |
| templates_mean | 44.06 | 44.06 | 48.55 |
| templates_lse | 48.44 | 47.38 | 50.76 |
| generic_lse | 48.79 | 49.23 | 49.63 |
| lexicon_lse | **52.79** | 51.86 | 53.17 |
| both_lse | 49.89 | 51.10 | 53.62 |

### 表 D · 与锚点的差

| | 本次 | 锚点 | Δ |
|---|---:|---:|---:|
| pooled r0 / r1 | 56.05 / 56.59 | 58.55 / 57.85 | −2.50 / −1.26 |
| macro r0 / r1 | 49.89 / 51.10 | 53.30 / 53.62 | −3.41 / −2.52 |
| C1 单独（scene-mean） | +2.26 | +14.65 | **−12.4** |
| C2 单独（scene-mean） | +7.57 / +8.46 | +7.80 | ≈0 |
| 交互项 | +0.88 / +1.51 | −3.62 | **符号相反** |

---

## 5. 复现注意事项（下次必读）

1. **缓存必须与 nag 成对。** shipped `clip_cache` 是 **1,015,937 高斯 / 366 簇**，盘上 nag 是 **1,012,438 / 386**。
   直接跑 `score_scannet.py` 会 `IndexError: boolean index did not match`。必须按 §3.2 重建，或找回那次重建的 ply+nag。
2. **锚点 58.55 不可由现有产物复现。** 两个独立信号说明场不同：① pooled 的 renorm 顺序翻转
   （锚点 r0>r1，实测 r0<r1）；② `lexicon_lse` 与 `both_lse` 的前后翻转（锚点 both>lex，实测 lex>both）。
   **数值差（−2.5）不如顺序翻转有诊断力。**
3. **`deco/REGRESSION.md` 只给了评分命令，没给场的重建参数。** 在干净机器上要复现 58.55，
   还需要那次产生 366 簇的 `merge_proj` 配置与对应 `point_cloud.ply` 的哈希——目前没有记录。
4. **C1 不要实现错。** 它是 `center_np`（沿**区域轴**减均值），不是逐视角 mask 中心化。
   用错算子会把 CLIP 上的 C1 增益从 +13.9 压到 +2.1（本项目实际踩过）。
5. **`--renorm` 不是装饰。** `renorm=0` 只做 `f - mean`；`renorm=1` 额外 `l2n`。
   单词语（`one_word`）下两者对 argmax 不可见（实测 47.60 / 41.13 完全相同），
   四模板 τ-LSE 下不同（锚点记录：109/366 = 29.8% 超点改变）。**这条可作自洽性检查。**
6. **两种聚合不要混用**：`pooled`（按类跨场景并集）与 `scene-mean`（先场景均值再平均）相差约 4–6 分，
   且会改变算子顺序。表格里两个都报，别只报一个。
7. **mAcc 也有两种**：`aggregate.py` 的 pooled mAcc（68.24）与逐场景 mAcc 的均值（63.42）不是一回事。
8. **除数 18 不是 19**：bookshelf 无 GT 点。子集 K15→14、K10→9。
9. **OpenSeg 场上该算子有害**：`deco/RESULTS.md` 记录 61.64 → 47.05，
   门控为 CMR（common-mode ratio）> 1.6；CLIP 2.14、OpenSeg 1.13。不要无条件搬用。
10. **不要引用本地 `reports/CVPR_DRAFT_DECISION_AUDIT_20261102.md` 里的 40.36 / 50.18 / OpenSeg 55.77**——
    那些数字已被仓库 `deco/RESULTS.md`（40.79 / 58.55 / 61.64）取代。

---

## 6. 本次会话的其它实验（同批产物，供索引）

| 脚本 | 问题 | 结论 |
|---|---|---|
| `scannet_candidate_coverage.py` | root 分区的完美语义上限 | 76.9 vs 当前 51.7，25.2 分在读端；细化候选 ≤ +4.7 |
| `scannet_mask_mixture_probe.py` | 一个簇是否被多张 mask 混合 | 平均 1.12 张 mask（>0.3），混合不是主因 |
| `scannet_feature_dispersion.py` | 纯簇/不纯簇的逐视角一致性 | 完全相同（0.715 vs 0.704）→ 跨视角一致性无信号 |
| `scannet_parent_child.py` | 子簇能否揭示父簇里的少数类 | 低于随机（top-3 命中 0.09 vs 随机 0.158） |
| `scannet_audit_offline.py` | 判决审计在两个场上的效果 | 十场景两场皆无效；CLIP 见证只在 1/10 场景优于区域，OpenSeg 7/10 |

---

## 7. 复现检查清单

- [ ] `sai_nag_3.0.pt` 的 `nag` 长度与 `clip_cache/*.npz` 的 `gt` 长度相等
- [ ] `nag` 层级数 = 4，`nag_feat` 长度 = 3（对应 level 1/2/3）
- [ ] 缓存 `gt>0` 比例与 shipped 缓存逐场景吻合（scene0000 0.982、scene0590 0.916）
- [ ] `aggregate.py` 打印 `divisor=18`
- [ ] 单词语下 `renorm 0` 与 `renorm 1` 完全一致（47.60 / 41.13）
- [ ] 与锚点比较时同时看 pooled 与 scene-mean，并检查 renorm 顺序是否一致