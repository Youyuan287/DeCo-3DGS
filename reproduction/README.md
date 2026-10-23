# ReLaGS · LERF-OVS 四场景复现（独立复现实验）

本目录是 **ReLaGS**(CVPR 2025) 在 **LERF-OVS** 数据集上的第三方独立复现，
上游仓库未发布评测代码（README 中 "Evaluation scripts will be published soon"），
本目录补齐了缺失的评测链路，并给出可复现的四场景结果。

> 上游代码保持**逐字节不动**；复现所需的三处兼容性修复以 `patches/` 形式提供。

---

## 1. 结果

纯 HuggingFace 输入（官方 2DGS 场景 + 作者发布的语义层级 `sai_nag.pt` + 官方 `lerf_ovs` 图像/标注）。

| Scene | 本复现 | 论文 Table 3 | Δ | queries | images |
|---|---:|---:|---:|---:|---:|
| figurines | 63.29 | 64.70 | −1.41 | 56 | 4 |
| ramen | 51.15 | 51.20 | −0.05 | 71 | 7 |
| teatime | 80.97 | 81.00 | −0.03 | 59 | 6 |
| waldo_kitchen | 60.62 | 60.60 | +0.02 | 22 | 5 |
| **Mean** | **64.01** | **64.40** | **−0.39** | 208 | 22 |

同一配置换成"对 208 条 query 直接平均"的 pooled 口径为 **63.71**；差异仅来自聚合方式。

完整分析见 **[FINAL_REPORT.md](FINAL_REPORT.md)**，分阶段记录见 [`reports/`](reports/)。

---

## 2. 快速开始

```bash
# 0) 依赖：见 env/SETUP.md（Python 3.10 + torch 2.1 + CUDA 11.8 + PyG 2.3.0）
#    需要先编译 submodules/ 下的 CUDA 扩展（diff-surfel-rasterization 等）

# 1) 应用 3 处兼容性补丁（仅本次运行需要，不改变论文算法）
bash reproduction/patches/apply_patches.sh

# 2) 准备数据（全部来自 HuggingFace）
#    模型/场景：dfki-av/ReLaGS  →  scenes/LeRF/<scene>/{point_cloud,cameras.json,input.ply,cfg_args,sai_nag.pt}
#    图像+标注：MinghanQin/lerf_ovs → lerf_ovs.zip（<scene>/images + label/）

# 3) 四场景评测
cd ReLaGS
export PYTHONPATH="$PWD/ext:$PWD"
python ../reproduction/scripts/eval_lerf_relags.py \
  --scene figurines \
  --source   <data>/lerf_ovs/figurines \
  --model    <hf>/scenes/LeRF/figurines \
  --nag      <hf>/scenes/LeRF/figurines/sai_nag.pt \
  --gt_root  <data>/lerf_ovs/label \
  --out results --csv results/lerf_query_metrics.csv \
  --topk 5 --level_until 3 --filter_small 50 --save_masks
```

批量：`bash reproduction/scripts/run_hf_only.sh`（四场景，约 3 分钟/场景）。

---

## 3. 评测协议

上游**没有** mIoU 实现（`metrics.py` 只有 PSNR/SSIM/LPIPS），协议由论文正文 + 代码默认值确定：

| 环节 | 取值 | 依据 |
|---|---|---|
| 候选层级检索 | `search_matched_superpoint_in_mhtree` | `nag_data.py` |
| `topk` | **5** | 该函数**默认值**（非按 GT 择优） |
| `level_until` | **3**（`!= -1` 即启用层级下降） | 论文 Alg.1 第 4 步 |
| `filter_small` | **50** | 该函数默认值 |
| `remove_small` | **True** | 该函数默认值（最大相似度落差截断） |
| 掩膜二值化 | `semantics[0] > 0.5` | 渲染出的二值簇掩膜 |
| 聚合 | 先逐图对 query 平均，再对图平均 | 与 THGS `test_lerf.py` 一致 |

**协议敏感性**（详见 `reports/08`）：

| 口径 | Mean |
|---|---:|
| 本目录主口径（topk=5 + 层级下降） | **64.01** |
| topk=3（最初误用基线仓库超参） | 59.76 |
| 只查 root 层（`level_until=-1`） | 43.40 |
| 论文 §15 字面"逐像素阈值"（`eval_lerf_pixel.py`） | 42.83（waldo） |
| ORACLE：逐 query 在 20 组配置取最优 | 65.03（GT 驱动，**不可引用**） |

---

## 4. 输入溯源

| 角色 | 文件 | 来源 |
|---|---|---|
| 2DGS 底座 | `point_cloud/iteration_0/point_cloud.ply` | `hf:dfki-av/ReLaGS/scenes/LeRF/<scene>/` |
| 作者语义场 | `sai_nag.pt` | 同上 |
| 相机/参数 | `cameras.json`、`input.ply`、`cfg_args` | 同上 |
| 图像 + GT | `<scene>/images/*.jpg`、`label/*.json` | `hf:MinghanQin/lerf_ovs/lerf_ovs.zip` |

逐文件 sha256 见 `results/hf_only_manifest.csv`。已核验：22 个 GT json 与两份独立本地副本 md5 一致；抽样图像 md5 一致。

底座同一性：官方 `iteration_0` 与 THGS 的 30k 在 `max_weight_pruning --contribution_threshold 0.0005`
下**顶点数逐点相同**（figurines 234,534 / teatime 958,840）；ply 含 `nx,ny,nz`，由
`diff-surfel-rasterization` 渲染 → 两者同为 **2DGS**。

---

## 5. 目录

```
reproduction/
├── FINAL_REPORT.md      最终交付稿（11 节，含结论/溯源/协议/局限）
├── reports/01..08       分阶段记录（环境审计→2DGS→流水线→协议→结果→HF口径→逐像素→参数上限）
├── scripts/
│   ├── eval_lerf_relags.py     主评测（簇级协议）
│   ├── eval_lerf_pixel.py      逐像素 relevancy 口径（并列/消融）
│   ├── sweep_lerf.py           检索参数扫描 + ORACLE 上限
│   ├── run_hf_only.sh          四场景一键评测
│   ├── run_scene.sh / run_all.sh / run_official_scene.sh / run_at_dir.sh
│   ├── summarize.py            汇总 CSV → 表格
│   └── regen_langsplat_features.py  失败的重建实验（保留存档，不参与指标）
├── patches/             3 处兼容性 diff + apply_patches.sh
├── results/             逐 query(208)/逐场景指标、协议对照、参数扫描、manifest
└── env/                 pip freeze + 环境搭建说明
```

---

## 6. 兼容性修复（不改变算法）

| 文件 | 问题 | 修复 |
|---|---|---|
| `submodules/diff-surfel-rasterization-*/setup.py` | conda 布局 CUDA 缺少 `targets/x86_64-linux/include`，编译报 `cub/cub.cuh` 找不到 | 追加 include 路径，并把包目录名改成与 `packages=` 一致 |
| `ext/spt/data/data.py` | 以方法调用 `data.keys()`，PyG 2.3.0 中 `keys` 是属性 → `'list' object is not callable` | 增加返回可调用 list 的 `keys` 属性垫片 |
| `merge_proj.py` | LangSplat 特征为 fp16，与 fp32 矩阵相乘报 dtype 错 | `.float()`（无损） |

剪枝阈值、图权重、cut-pursuit 参数、区域生长阈值、CLIP 编码均保持官方值。

---

## 7. 已知局限

- 论文未给出 `topk`/聚合细节，评测代码未公开；剩余 −0.39 可能来自这些未公开细节（多实例合并、聚合权重）。
- 未从零训练 2DGS，复用官方发布场景。
- 仅覆盖 LERF-OVS 四场景，未做 ScanNet / 3D-OVS / 场景图预测。
- 本目录早期数字 55.38（剪枝阈值误用 0.005）与 59.76（`topk=3`）**均已作废**，仅作为过程记录保留在 `reports/05`、`reports/06`。

## 8. 引用

```bibtex
@inproceedings{xie2025relags,
  title     = {ReLaGS: Relational Language Gaussian Splatting},
  author    = {Xie, Yaxu and Arafa, Abdalla and Javanmardi, Alireza and Millerdurai, Christen
               and Hu, Jia Cheng and Wang, Shaoxiang and Pagani, Alain and Stricker, Didier},
  booktitle = {CVPR},
  year      = {2025}
}
```
