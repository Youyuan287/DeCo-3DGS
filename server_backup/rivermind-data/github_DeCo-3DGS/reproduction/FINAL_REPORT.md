# ReLaGS · LERF-OVS 四场景复现 — 最终交付稿

合并自 `01`–`08`，数字以本稿为准（2026-10-23 最终口径）。
仓库：`github.com/dfki-av/ReLaGS` @ `52cc0c9`（"initial release of the code"）。

---

## 1. 三句话结论

1. **2DGS 几何底座：是。** 使用 ReLaGS 官方 HF 发布的 2DGS 场景（`iteration_0`，即 `max_weight_pruning` @ `0.0005` 后的产物）。已用顶点数逐点核验：THGS 的 30k 在该阈值下与官方 `iteration_0` **完全相同**（figurines 234,534；teatime 958,840），且 ply 含 `nx/ny/nz`、由 `diff-surfel-rasterization` 渲染 —— 两者是同一套 2DGS 底座，不是 3DGS。
2. **全链路：是。** 剪枝 → kNN 建图 → SAM 引导图重加权 → cut-pursuit 二次划分 → 层级合并与 CLIP 特征重投影 → 层级语义产物 → 逐 query 评测，四场景全部打通，中间产物与层级结构校验通过。
3. **64.4% 是否复现：基本复现，差 0.39 个点。** 在纯 HF 输入 + 作者默认检索参数下，四场景 Mean mIoU = **64.01**（论文 64.40）。单场景 ramen/teatime/waldo 与论文差距 ≤0.05，figurines 为 −1.41。

---

## 2. 最终结果

主口径：`search_matched_superpoint_in_mhtree(topk=5, level_until=3, filter_small=50, remove_small=True)`
—— 前三个值即仓库该函数的**默认值**，`level_until != -1` 对应论文 Algorithm 1 的层级下降。
指标：逐图对 query 求 IoU 平均，再对图平均；渲染掩膜取 `semantics[0] > 0.5`。

| Scene | 本次复现 | 论文 Table 3 | Δ | queries | images |
|---|---:|---:|---:|---:|---:|
| figurines | **63.29** | 64.70 | −1.41 | 56 | 4 |
| ramen | **51.15** | 51.20 | −0.05 | 71 | 7 |
| teatime | **80.97** | 81.00 | −0.03 | 59 | 6 |
| waldo_kitchen | **60.62** | 60.60 | +0.02 | 22 | 5 |
| **Mean** | **64.01** | **64.40** | **−0.39** | 208 | 22 |

明细：`results/lerf_query_metrics.csv`、`results/lerf_scene_metrics.csv`、预测掩膜 `results/masks/`。

---

## 3. 输入溯源（全部来自 HuggingFace）

| 角色 | 文件 | 来源 |
|---|---|---|
| 2DGS 底座 | `point_cloud/iteration_0/point_cloud.ply` | `hf:dfki-av/ReLaGS/scenes/LeRF/<scene>/` |
| 作者语义场 | `sai_nag.pt` | 同上 |
| 相机 / 场景参数 | `cameras.json`、`input.ply`、`cfg_args` | 同上 |
| 图像 | `<scene>/images/*.jpg` | `hf:MinghanQin/lerf_ovs/lerf_ovs.zip` |
| GT 掩膜与 query | `label/<scene>/*.json` | 同上 |

逐文件 sha256：`results/hf_only_manifest.csv`。

可追溯性证据：
- HF `lerf_ovs.zip` 内 22 个 GT json 与本地副本 **22/22 md5 一致**；本地副本与 THGS 内另一份独立副本亦逐场景一致。
- 抽检 3 张图像，HF 与本地 **md5 一致**。
- 评测脚本不读任何 `langsplat/language_features`（`grep` 确认）。

---

## 4. 评测协议的确定过程

1. 论文 §15："渲染语言特征图 → 逐像素 CLIP 相似度 → 阈值二值化 → IoU"。
2. 论文 §12 Algorithm 1：top-K root 候选 → 子簇相似度更高则下降 → 过滤小于父簇 1% 的簇 → 取最大相似度落差 → 聚合高斯。
3. 仓库无 mIoU 实现（`metrics.py` 仅 PSNR/SSIM/LPIPS）。
4. 用户确认：THGS 用其自身协议可复现论文基线值 → 簇级协议成立。

据此采用**簇级**协议（与 THGS `test_lerf.py` 同构），而非逐像素。

---

## 5. 协议敏感性与上限（全部真值驱动，仅作分析）

| 口径 | Mean mIoU | 说明 |
|---|---:|---|
| **本稿主口径**（topk=5, 层级下降，逐图平均） | **64.01** | 作者默认 + Alg.1 |
| 同一配置的 pooled 口径（对 208 query 直接平均） | 63.71 | 与上行的差异仅来自聚合方式：逐图平均给图数少的场景更大权重 |
| topk=3（此前误用 THGS 超参） | 59.76 | 偏差来源，已作废 |
| 仅 root 层（`level_until=-1`） | 43.40 | 关闭层级下降，与 Alg.1 矛盾 |
| 逐像素相似度阈值（§15 字面实现） | 42.83（waldo） | 严重过选，见 `07` |
| ORACLE：逐 query 在 20 组配置取最优 | 65.03 | GT 驱动上限，**不可作为结果** |

两种聚合都远高于 `topk=3` 的 59.76，结论不受聚合方式影响。`topk` 是最敏感参数；`filter_small` 场景相关且非单调（teatime 偏好 50、waldo 偏好 500），故保持默认 50，不按场景调整。

---

## 6. 底座核验

- `iteration_0` 与 THGS 30k 在 `max_weight_pruning --contribution_threshold 0.0005` 下顶点数**逐点相同**；该剪枝用 ReLaGS 的 2DGS 光栅化器渲染全视角计算 max-contribution。
- ply 字段含 `nx/ny/nz`（3DGS 不存法向），`ReLaGS` 的 `load_ply` 按 `['x','y','z','nx','ny','nz']` 读取。
- `THGS/submodules/` 为 `diff-surfel-rasterization`（2DGS 光栅化器）；THGS README 明写 "builds 3D semantic fields on top of **2DGS** scenes"。
- 结论：**THGS 与 ReLaGS 共用同一套 2DGS 底座**，此前"THGS 是 3DGS"的理解有误。

---

## 7. 流水线执行与兼容性修复

真实顺序（`scripts/run_lerf.sh`）：`max_weight_pruning(0.0005)` → `sp_partition -a` → `graph_weight --level 1` → `sp_partition -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a` → `merge_proj --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3`。

为跑通所做的 3 处兼容性修改（**未改算法**，diff 见 `patches/`）：

| 文件 | 问题 | 修复 |
|---|---|---|
| `submodules/diff-surfel-rasterization-trace-maxcontrib/setup.py` | conda 布局 CUDA 缺 `targets/x86_64-linux/include` | 追加该 include，并把包目录改名为与 `packages=` 一致（`__init__.py` 逐字节未变） |
| `ext/spt/data/data.py` | 以方法形式调用 `data.keys()`，PyG 2.3.0 中是属性 | 增加可调用 list 的 `keys` 属性垫片 |
| `merge_proj.py` | LangSplat 特征 fp16 与 fp32 矩阵相乘报 dtype 错 | `.float()` 转换（无损） |

另替换 2 个 py38 二进制为 py310（`grid_graph`、`cp_d0_dist_cp`），源自本机 THGS 环境。

---

## 8. 事件与更正记录（透明性）

1. **剪枝阈值用错**：`configs/lerf.yml` 写 `0.005`，而官方 `run_lerf.sh` 是 `0.0005`（仓库自身不一致）。我一度按 yaml 执行导致过度剪枝，该批结果（Mean 55.38）**已作废**。
2. **figurines 特征被删**：`figurines/langsplat/language_features` 于 2026-10-23 00:53 被删除（用户确认为误删）。
3. **重建尝试**：我据官方数据结构反推并实现了 SAM+CLIP 重建（`scripts/regen_langsplat_features.py`），但参数与官方 `scripts/image_encoding.py` 不一致，产物**不等值**，已删除，不进入任何结果。
   > **2026-10-31 更正**：该判断已过时。用官方 `scripts/image_encoding.py`（laion2b_s34b_b88k + SAM ViT-H）
   > 在训练分辨率上重编码 299 帧后重建的 `sai_nag_3.0.pt`，与作者发布权重的逐簇余弦为 **0.9999**，
   > mIoU **66.10**（发布权重 63.29）。当年"不等值"的真实根因是
   > `scene/dataset_readers.py:117` 用字符串拼 `images/../language_features/...`，
   > 而我把场景目录做成了符号链接薄壳 → `..` 穿透到原始目录，读到的其实是 LangSplat 的
   > OpenAI ViT-B/16 特征（逐位 cos 1.00000）。详见
   > [`audits/figurines_feature_encoding_20261031/`](audits/figurines_feature_encoding_20261031/)。
4. **HF 特征发现**：`JUNHAKBAE/THGS-lerf-ovs-language-features` 中 figurines 的 `_s` 形状与原生分辨率一致，但其余目录组织有误；本稿最终**不使用任何语言特征**（评测只用作者发布的 `sai_nag.pt`），故不受影响。
5. **选择合适的检索参数**：`topk=5` 是仓库默认值，属"修正回默认"，非按 GT 择优；但本人在同一评测集上观察过该值更好，存在轻微测试集选择效应，如实记录。ORACLE 行未作为结果。

---

## 9. 局限

- 官方**评测代码从未发布**，`level_until` 语义（数值本身不敏感，仅 `-1` 与"非 -1"有别）、多实例合并方式、逐图/逐 query 聚合权重均无法与作者逐位对齐；剩余 −0.39 的差距可能来自这些细节。
- 未从零训练 2DGS，直接复用官方发布场景（磁盘与本机算力限制）。
- 仅覆盖 LERF-OVS 四场景；未做 ScanNet / 3D-OVS / 场景图关系。
- 我此前的 55.38、58.11/40.40/82.91/57.62（topk=3）两组数字均已被本稿取代，保留在 `02`/`05`/`06` 作为过程记录。

---

## 10. 复现步骤

```bash
# 1. 取 HuggingFace 资产（模型 + 图像 + GT）
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("dfki-av/ReLaGS", repo_type="dataset", local_dir="ref/relags_hf",
                  allow_patterns=["scenes/LeRF/*"])
PY
curl -L -o ref/hf_lerf/lerf_ovs.zip \
  https://hf-mirror.com/datasets/MinghanQin/lerf_ovs/resolve/main/lerf_ovs.zip

# 2. 评测（四场景）
bash scripts/run_hf_only.sh            # 内部调用 scripts/eval_lerf_relags.py
```

关键参数：`--topk 5 --level_until 3 --filter_small 50`（`remove_small` 默认 True）。

---

## 11. 产物索引

| 路径 | 内容 |
|---|---|
| `reports/00_FINAL_DELIVERY.md` | 本稿 |
| `reports/01`–`08` | 分阶段记录（含被取代的中间数字） |
| `results/lerf_scene_metrics.csv` | 最终逐场景指标 |
| `results/lerf_query_metrics.csv` | 208 条逐 query IoU |
| `results/hf_only_manifest.csv` | 输入 sha256 溯源 |
| `results/final_protocol_metrics.csv` | 各协议对照 |
| `results/masks/`、`results/vis/` | 预测掩膜与三联图 |
| `patches/*.diff` | 3 处兼容性补丁 |
