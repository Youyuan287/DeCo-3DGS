# LERF-OVS 67.60 (DeCo B0) — 可复现交付包

本目录记录并复现 **LERF-OVS 四场景 macro mIoU = 0.6760（67.60）** 的最好结果。
它不是单一“新网络”，而是在 **released ReLaGS/DeCo 读权 + 固定逐视角 mask token 场** 上，
用 **late-interaction 读出 + 同尺度证据 gating** 得到的训练-free readout。

## 0. 结果

B0 协议（C1 开启，`--center image`，text=canon，topk=5，level_until=3，filter_small=50，gap）：

| 场景 | main | gateE (ours) | Δ |
|---|---:|---:|---:|
| figurines | 0.5859 | 0.6129 | +0.0270 |
| ramen | 0.5181 | 0.6092 | +0.0911 |
| teatime | 0.7939 | 0.8223 | +0.0284 |
| waldo_kitchen | 0.6592 | 0.6597 | +0.0005 |
| **macro** | **0.6393** | **0.6760** | **+0.0367** |

配对场景分层 bootstrap（按图重采样，B=2000，seed=20261030）：

- gateE − main：**+0.0367，95% CI [+0.0247, +0.0482]**
- gateE − module C：+0.0158，CI [+0.0068, +0.0248]
- gateE − mask rerank：+0.0141，CI [+0.0056, +0.0240]

分族（found 124 / rank2 36 / deep 19 / recall 29）：

| 族 | main | module C | mask rerank | **gateE** |
|---|---:|---:|---:|---:|
| found | 0.8160 | 0.8312 | 0.8160 | **0.8312** |
| rank2 | 0.6635 | 0.6402 | 0.6631 | 0.6402 |
| deep | 0.2263 | 0.4446 | 0.5281 | **0.6607** |
| recall | 0.0266 | 0.0198 | 0.0261 | 0.0198 |

收益几乎全部来自 **deep（+0.434）** 与 **ramen（+0.091）**；found 不降，recall 未解决。

## 1. 原理

### 1.1 固定 token 场 + late interaction 读出（不是“文本进聚合”）

对每个候选，场里存的是**逐视角 SAM mask 特征集合** `{f_o}`（query-agnostic，建一次可复用）。
查询时有两种读出：

| 读出 | 公式 | 压缩？ |
|---|---|---|
| 单向量 | `f_c = Σ a_o f_o`, `s = <f_c, t>` | 是 |
| **late interaction** | `s = mean(top-k_o <f_o, t>)` | **否，保留 k 个视角 token** |

`id_mean` 是单向量；`id_top3` 是 late interaction。**两者的场相同，差别只在读出。**
同一 token 场下（B0）：

| 读出 | macro | Δ vs main |
|---|---:|---:|
| id_mean（单向量） | 0.6448 | +0.0056 |
| id_mean_c（单向量） | 0.6421 | +0.0028 |
| **id_top3（late interaction）** | **0.6619** | **+0.0226** |

所以增益来自**不把多视角压成一个向量**：平均会把背景/共享语义混进物体特征，top-k 让最像物体的视角说话。

### 1.2 同尺度证据竞争 gating（gateE）

两套选择：

- `S_C`：module C（guarded prefix pruning + scale-gated challenger inclusion），基于 NAG 分数；
- `S_M`：按 `S = z(NAG) + 1.0·z(id_top3)` 在 released k 内重排。

gateE 不比较跨尺度的 margin（实测会退化为 C），而是比较**同一 mask 证据空间**：

```
e_C = max_{i in S_C} id_top3(i)
e_M = max_{i in S_M} id_top3(i)
S_gateE = S_M   if set(S_M) != set(S0) and e_M > e_C
          S_C   otherwise
```

含义：只有当 mask 证据确实改变了选择、且其选中的集合物体证据更强时，才替换；
否则回到 module C 的保守修正。这样 found 不被牺牲，deep 被修复。

### 1.3 为什么不用一致性

一致性（ROFA/agree）是 query-agnostic 的：它优化“视角之间像不像”，在区域/掩码混合特征上
更容易选中共享/背景语义，把 found 命中从 77.4% 打到 13.7–15.3%。一致性只能用于**关联/分组**，
不能用于**评分聚合**。详见 `docs/WHY_CONSISTENCY_HURTS.md`。

## 2. 实现

### 2.1 组件

1. **逐视角 mask token 场**（Phase B）：`lerf_segment_identity_dump.py`
   - 对每个候选，取覆盖它的逐视角 SAM mask 特征；
   - 计算 `id_top3 = mean(top-3 per-view <f_o,t>)`、`id_mean`、`agree` 等；
   - 场本身与文本无关；`id_top3` 是查询时读出。
2. **module C**：guarded prefix pruning + scale-gated challenger inclusion（preds/evidence 输入）。
3. **gateE**：`lerf_c_mask_gate.py`，按 1.2 的规则选择并渲染。
4. **评估**：官方 GT 多边形，渲染 semantics>0.5；scene mIoU 先按图平均，再跨图/跨场景。

### 2.2 精确配置

- Base：DeCo B0，`--center image --renorm 0 --text canon --topk 5 --level_until 3 --filter_small 50 --sel gap`
- `S = z(NAG) + 1.0 * z(id_top3)`（λ=1.0 固定）
- k = released gap 长度（不重新截断）
- 无 GT 参与选择；GT 只用于 IoU 与分族

### 2.3 伪代码

```
for each query (image, prompt):
    p5 = released top-k pool (NAG scores)
    g  = gap_len(NAG scores); S0 = p5[:g]
    S_C = module_C(p5, preds, evidence)          # prune + include
    S_M = top-g by z(NAG) + z(id_top3)           # late-interaction rerank
    if set(S_M) != set(S0) and max_id_top3(S_M) > max_id_top3(S_C):
        S = S_M
    else:
        S = S_C
    render(union of S); IoU vs GT
```

## 3. 复现

### 3.1 依赖

- 官方读权：`ref/relags_hf/<scene>/sai_nag.pt`（ReLaGS/DeCo released）
- DeCo 评测仓库：`REPO/deco/lerf_ovs/eval_lerf_ovs.py` 及其依赖（`scene`, `nag_data`, `utils.vlm_utils`）
- 本包输入：`inputs/segidentity/<scene>_segidentity.csv`、`inputs/preds.csv`、`inputs/evidence/<scene>_evidence.csv`

### 3.2 一条命令（每个场景）

```bash
PY=/path/to/venv/bin/python3
REPO=/path/to/DeCo-3DGS
REL=/path/to/ReLaGS
SRC=/path/to/VALA/dataset/3dgs/lerf_ovs
MODEL=/path/to/ref/relags_hf
GT=/path/to/ref/hf_lerf/lerf_ovs/label
PKG=/path/to/this/package

for sc in figurines ramen teatime waldo_kitchen; do
  $PY $REPO/deco/lerf_ovs/lerf_c_mask_gate.py \
    --scene $sc --source $SRC/$sc --model $MODEL/$sc \
    --nag $MODEL/$sc/sai_nag.pt --gt_root $GT \
    --out $PKG/results/$sc --id_csv $PKG/inputs/segidentity/${sc}_segidentity.csv \
    --preds $PKG/inputs/preds.csv --evidence $PKG/inputs/evidence \
    --center image
done
```

输出：`results/<scene>/<scene>_cmaskgate_image.csv`，含 `main_iou/module C_iou/mask_iou/gatee_iou`。

### 3.3 汇总

```bash
$PY summarize_6760.py --results results
```

输出 macro、per-scene、per-family、bootstrap CI（按图、seed=20261030、B=2000）。

### 3.4 从头重建 token 场（可选）

若没有 `inputs/segidentity/*.csv`，运行：

```bash
$PY $REPO/deco/lerf_ovs/lerf_segment_identity_dump.py \
  --scene <scene> --source $SRC/<scene> --model $MODEL/<scene> \
  --nag $MODEL/<scene>/sai_nag.pt --gt_root $GT \
  --sas /path/to/m1_sas_20261029/<scene>_new --out $PKG/inputs/segidentity
```

## 4. 消融与协议

| 实验 | 结果 |
|---|---|
| 单向量读出（id_mean） | 0.6448（+0.56） |
| 单向量（id_mean_c） | 0.6421（+0.28） |
| late interaction（id_top3） | 0.6619（+2.26） |
| module C | 0.6603（+2.10） |
| gateB（启发式） | 0.6757 |
| **gateE（同尺度证据）** | **0.6760** |
| leave-query-view-out（剔除查询视角） | 0.6753（+3.60） |

**协议说明**：场是固定的、query-agnostic；文本只在读出端使用。
baseline 是**单向量读出**；我们是**多 token late interaction + gating**。
比较时必须声明读出方式/token 预算，或给 baseline 同样的 token 与读出。
不能直接声称“同读出下超过 baseline”。

## 5. 局限

1. 只在 LERF-OVS 4 场景、208 查询、单次运行；CI 见上。
2. 收益集中在 deep（19 条）与 ramen；teatime 完全不改（mask 重排 0 条），waldo 略动。
3. recall 29 条仍未解决（池内最佳 0.027）；需要候选生成/扩池。
4. `S` 融合的 λ=1 固定；mask rerank 的 λ 敏感性（0.5→+0.37、2.0→−2.00）仍存在，
   gateE 依赖 module C 回退降低风险。
5. 逐视角 mask 特征来自数据集的 `language_features`；与官方 `sai_nag.pt` 的生成版本可能不同。

## 6. 文件

```
README.md                     本说明
scripts/lerf_c_mask_gate.py   gateE 主程序
scripts/lerf_maskagg_rerank.py  mask rerank（对照）
scripts/lerf_segment_identity_dump.py  token 场导出
scripts/lerf_segment_identity_dump_lo.py leave-query-view-out 对照
scripts/summarize_6760.py     汇总/CI
inputs/segidentity/*.csv      逐候选 token 证据
inputs/preds.csv              NAG 候选分数
inputs/evidence/*_evidence.csv  area/visibility
results/gatee/*.csv           逐查询结果
results/baselines/b0/*.csv    B0 main 逐查询
manifest.json                 输入 SHA 与版本
docs/PREREG*.md               预注册
docs/WHY_CONSISTENCY_HURTS.md 诊断
```

## 7. 参考方法（投稿前请核对引用）

- ColBERT (SIGIR 2020)：late interaction / MaxSim。
- OpenMask3D (NeurIPS 2023)：多视角 mask 特征描述 3D instance。
- Open3DIS (CVPR 2024)：2D mask 引导 3D instance。
- MaskClustering (CVPR 2024)：视角共识 mask 图做 3D instance。
- PartSLIP / PartSLIP++ (CVPR 2023/2024)：多视角多原型。
- Prototypical Networks (NeurIPS 2017)、Deep Sets (NeurIPS 2017)、Set Transformer (ICML 2019)：多原型/集合表示。
- ReLaGS / SAI3D / Gaussian Grouping / LangSplat / LEGaussians / Dr.Splat：3D 场与 instance 建场对照。

—— 结果日期：2026-10-31；本包对应本地提交（见 manifest）。