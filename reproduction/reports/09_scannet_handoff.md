# ReLaGS · ScanNet 复现交接（记忆 + 提示词）

> 用途：在新窗口继续 ReLaGS 的 ScanNet 实验。本文件上半部分是**事实记忆**（已核实），
> 下半部分第 9 节是**可直接粘贴的提示词**。
> LERF-OVS 四场景已完结：Mean **64.01** vs 论文 64.40（详见 `reproduction/FINAL_REPORT.md`）。

---

## 1. 机器与环境（已就绪，不要重装）

```
ssh root@hn01-ssh.gpuhome.cc -p 30946
```

| 项 | 值 |
|---|---|
| 项目根 | `/root/rivermind-data/ReLaGS_Reproduction/` |
| 上游代码（已打 3 个补丁、已编译 .so） | `ReLaGS_Reproduction/ReLaGS/` @ `52cc0c9` |
| Python | `ReLaGS_Reproduction/venv/bin/python`（3.10 + torch 2.1 + cu118 + PyG 2.3.0） |
| 备份/发布仓库 | `/root/rivermind-data/ReLaGS-1023/`（remote 已配好 **SSH**：`git@github.com:Youyuan287/ReLaGS-1023.git`，可直接 `git push`） |
| 磁盘 | `/root/rivermind-data` 50G，**剩余约 7.9G**（紧张，见 §4） |

运行任何 ReLaGS 脚本必须：

```bash
cd $R/ReLaGS
export PYTHONPATH=$R/ReLaGS/ext:$R/ReLaGS
export HF_HUB_OFFLINE=1
```

**只读约束（硬性）**：`/root/rivermind-data/VALA/`、`/root/rivermind-data/THGS/` 一律只读，
不得修改/删除/覆盖；所有写入限制在 `ReLaGS_Reproduction/` 与 `ReLaGS-1023/` 内。
VALA 里有一份 `dataset/scannet.zip` 可以**读**，但解压目标必须放在自己的目录。

---

## 2. LERF 阶段沉淀的方法论（ScanNet 沿用）

1. **自动推进**，不要每步确认；遇到兼容性 bug 直接修，但必须记录原因 + diff。
2. **不伪造论文数字**。任何论文数值必须先从 `ref/paper.txt` 找到原文再引用。
3. **不按 GT 调参**。协议参数优先取"代码默认值"或"论文原文"；若做扫描，必须标 `ORACLE` 且不得作为结果。
4. **输入全部可溯源**：只从 HuggingFace / 官方发布取，落 `sha256` 清单。
5. **优先复用**已有 harness、已有 venv、已编译扩展。
6. 每个结论都要有**可复核的产物**（CSV/JSON/日志），不能只有口头结论。

---

## 3. 已核实的 ScanNet 事实（务必先读）

### 3.1 论文里 ScanNet 做的是「3D 语义分割」，且有**修订版评测协议**

- `ref/paper.txt:1223` 「Additional Analysis on ScanNet」— 明确提到"用**固定数量/致密化(densified)**高斯训练 ScanNet 场景"。
- `ref/paper.txt:1239` **Table 9. ScanNet Results with Revised Evaluation Protocol** ← **指标口径以这张表为准，先读它**。
- `ref/paper.txt:682 / 694 / 714` 正文声明 ScanNet 上做语义分割且取得最高 mean IoU。
- `ref/paper.txt:1469` ScanNet 数据集描述（1513 个室内 RGB-D 场景）。
- `ref/paper.txt:1500–1512` 3D 基准（ScanNet / ScanNet++）的评测方式说明（预测如何映射到 GT）。实测数值与协议要点见 **§11.1**（已抄录，不要再猜）。

> 注意：LERF 用的是 CLIP ViT-B-16 + LERF 多边形 GT；**ScanNet 不是同一套**，不要照搬。

### 3.2 上游 ScanNet 脚本有三处**明显的不一致**（必须先解决再跑）

`ReLaGS/scripts/run_scannet.sh`：

| 现象 | 内容 | 风险 |
|---|---|---|
| 1 | `-m ./output/lerf/${scan}`，但收尾 `predict_3d_scene_graph.py --root_pred ./output/scannet/${scan}/` | 中间产物写进 `output/lerf/`，最后却去 `output/scannet/` 找 → 路径断裂，疑似从 `run_lerf.sh` 复制未改 |
| 2 | `--config ./configs/lerf.yml`，但仓库另有 `configs/scannet.yml` | 极可能用错配置 |
| 3 | `--vlm_type openseg`（LERF 是默认 CLIP） | **需要 OpenSeg 特征，不是 CLIP ViT-B-16** |

`configs/scannet.yml` 与 `configs/lerf.yml` 的关键差异（说明 ScanNet 参数确实不同）：

| 键 | lerf.yml | scannet.yml |
|---|---|---|
| `dataset.name` | LERF-OVS | 3DOVS（写错的痕迹） |
| `dataset.save_folder` | `lerf` | `output/scannet/without_densification` |
| `merge_proj.thres_merge` | 20 | **100** |
| `spt.pcp_regularization` | 0.1 | **0.3** |
| `pruning.contrib_thres` | 0.005 | 未写 |

处理原则：**以脚本实际传参为准**判"作者真实跑法"，但与 yml 的冲突要写进报告；
LERF 阶段就踩过同类坑（`run_lerf.sh` 用 `0.0005`、`lerf.yml` 写 `0.005`，按 yml 跑导致过度剪枝、结果作废）。

### 3.3 场景列表（10 个）

```
scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00
scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00
```

### 3.4 数据现状

| 资产 | 位置 | 状态 |
|---|---|---|
| ScanNet v2 原始包 | `/root/rivermind-data/VALA/dataset/scannet.zip` | **11.5 GB**，sha256 在同目录 `scannet.sha256`（`7931fd64…f66`），内含 `scannet/{scans,scans_test,tasks,scannetv2-labels.combined.tsv}`，7784 entries |
| 2DGS ScanNet 场景 | **未知**，需确认 HF `dfki-av/ReLaGS` 是否发布 `scenes/ScanNet/<scan>/` | HF API 在镜像上返回空，需换方式查（见 §5） |
| OpenSeg 特征 | **本机没有**（LERF 用的是 `langsplat/language_features` 的 CLIP fp16） | 需从 HF 取或自己跑 OpenSeg |

**磁盘风险**：11.5G 的 zip 只有 7.9G 剩余空间 → **禁止整包解压**。
建议用 Python `zipfile` 做**按需抽取**（只取 10 个场景目录），或先确认是否根本不需要原始 ScanNet
（若 HF 已发布训练好的 2DGS ScanNet 场景，则只需场景 + 语义场）。

---

## 4. 磁盘预算（先算再动手）

- 每个 ScanNet 场景原始数据约 0.1–0.3 GB（含 `.sens`、`label`、`ply`）。
- 2DGS 训练产物（若需自己训）：每场景数百 MB。
- 语义链路中间产物（`neighbor*.pt`、`sai_nag.pt`）：每场景约 0.3–1 GB。
- **先做一次 `df -h` + 估算，再决定是否只做 2–3 个场景**；宁可缩小场景数，也不要磁盘写满（写满会让 LERF 现有产物一起不可用）。

---

## 5. 需要优先确认的三件事（决定后续全部工作量）

> ⚠️ **2026-10-23 二次核实：本节问题已有答案，见 §11。先读 §11 再动手。**

1. **HF 上有没有 ScanNet 的 2DGS 场景和 `sai_nag.pt`？**
   查法（镜像 API 不可靠，用网页或 resolve 探测）：
   ```bash
   curl -sI https://hf-mirror.com/datasets/dfki-av/ReLaGS/resolve/main/scenes/ScanNet/scene0000_00/cameras.json | head -3
   ```
   或直接看 `https://huggingface.co/datasets/dfki-av/ReLaGS/tree/main/scenes`。
   > 有 → 直接复用，工作量与 LERF 相当；没有 → 需要自己训 2DGS（工作量大增，先评估磁盘/时间再决定）。
2. **OpenSeg 特征从哪来？** `--vlm_type openseg` 需要 `language_features_openseg`。
   查 `merge_proj.py` / `dataset_readers.py` 里 openseg 分支的路径约定，再定位 HF 上对应目录。
3. **ScanNet 的 GT 与评测口径**：`scannetv2-labels.combined.tsv`（类别映射）+ 论文 Table 9 的
   "revised protocol"。**先把 Table 9 读懂再写评测代码**，不要用 LERF 的 IoU harness 直接套。

---

## 6. 复用清单（不要重写）

| 已有资产 | 用法 |
|---|---|
| `scripts/eval_lerf_relags.py` | LERF 主评测；ScanNet 的聚合/掩膜约定可参考，但**指标定义需按 Table 9 重写** |
| `scripts/summarize.py` | CSV → 表格 |
| `patches/*.diff` | 3 个兼容性补丁（CUDA include、PyG `keys` 垫片、fp16 dtype），ScanNet 同样需要 |
| `reproduction/reports/01–08` | 环境审计、流水线顺序、协议推导的完整推理过程，可照抄结构 |
| `reproduction/env/SETUP.md` | 环境踩坑清单 |

---

## 7. 已知技术坑（已在 LERF 阶段踩过，别再踩）

1. **语义通道数编译期固定为 20**：给光栅化器 512 维会非法访存/静默截断。需要 512 维场时按 20 通道分块渲染再拼接。
2. **`unzip` 命令不存在**：用 Python `zipfile`。
3. **`huggingface_hub` 直连不通**：用 `curl` 打 `hf-mirror.com`，并 `env -u http_proxy -u https_proxy -u all_proxy`。
4. **PowerShell 里不要把 ssh 输出 pipe 给 `head/tail/Select-String`**（会 CommandNotFoundException）；把逻辑写进远端脚本，用 base64 heredoc 下发。
5. **`configs/*.yml` 与 `scripts/*.sh` 参数冲突**是这仓库的固有毛病，以脚本为准并记录差异。
6. 剪枝阈值 LERF 用 `0.0005`（`run_scannet.sh` 同样写 `0.0005`，**不要**用 yml 的 `0.005`）。
7. `figurines` 的语言特征曾被误删、且重建产物**不等值**——**不要试图自己生成特征**，一律用官方发布物。

---

## 8. 交付物定义

沿用 LERF 阶段的结构，全部放在 `ReLaGS_Reproduction/`：

```
reports/09_scannet_setup.md       环境/数据/资产确认结论
reports/10_scannet_pipeline.md    流水线实跑记录 + 参数与上游不一致处
reports/11_scannet_results.md     结果 vs 论文 Table 9，逐场景 + Mean
results/scannet_query_metrics.csv 逐 query（或逐类）明细
results/scannet_scene_metrics.csv 逐场景汇总
results/scannet_manifest.csv      输入 sha256 溯源
```

完成后把上述内容复制进 `ReLaGS-1023/reproduction/` 并 commit（remote 已配 SSH，直接 `git push`）。

---

## 9. 提示词（粘贴到新窗口）

```
我要复现 ReLaGS 在 ScanNet 上的结果。工作目录在远程服务器，请全程自动推进、不要每步征求确认。

【连接】
ssh root@hn01-ssh.gpuhome.cc -p 30946
项目根 R=/root/rivermind-data/ReLaGS_Reproduction
上游代码 $R/ReLaGS（已打 3 个兼容补丁、已编译，直接用，不要重装）
跑任何脚本前：cd $R/ReLaGS && export PYTHONPATH=$R/ReLaGS/ext:$R/ReLaGS && export HF_HUB_OFFLINE=1

【硬性约束】
1. /root/rivermind-data/VALA/ 与 /root/rivermind-data/THGS/ 只读，绝不修改/删除；写入只允许在 $R 和 /root/rivermind-data/ReLaGS-1023/。
2. 不得伪造任何论文数字。所有论文数值必须先 grep $R/ref/paper.txt 找到原文再引用，引用时给出文件行号。
3. 不得按 GT 调参。协议参数取"代码默认值"或"论文原文"；如做参数扫描，必须标注为 ORACLE 且不得当作结果。
4. 遇到兼容性 bug 直接修，但必须记录：现象、根因、diff、影响面。
5. 磁盘只剩约 7.9G：/root/rivermind-data 50G 已用 43G。动手前先 df -h 并估算占用，宁可减少场景数也不要写满磁盘。

【第一步：先读，不要先跑】
1. 读 $R/ref/paper.txt 的 ScanNet 部分：约 1223-1245 行（含 Table 9「ScanNet Results with Revised Evaluation Protocol」）、
   682/694/714 行（正文声明）、1469 行（数据集描述）、1500-1512 行（评测方式）。
   先告诉我：ScanNet 的指标定义到底是什么、GT 从哪来、有几个场景、论文报告的数值是多少（给行号）。
2. 读上游 $R/ReLaGS/scripts/run_scannet.sh 与 configs/scannet.yml，列出两者的冲突点。
   已知至少三处：(a) -m 写 ./output/lerf/${scan} 但结尾 predict_3d_scene_graph 读 ./output/scannet/；
   (b) --config 用 lerf.yml 而仓库另有 scannet.yml；(c) --vlm_type openseg 而 LERF 是 CLIP。
3. 确认资产是否存在：HF 上 dfki-av/ReLaGS 是否发布 scenes/ScanNet/<scan>/（含 point_cloud、sai_nag.pt）；
   OpenSeg 特征在哪；ScanNet 原始包在 /root/rivermind-data/VALA/dataset/scannet.zip（11.5G，只读，禁止整包解压，
   用 python zipfile 按需抽取到 $R 下的目录）。

【先读】/root/rivermind-data/ReLaGS-1023/reproduction/reports/09_scannet_handoff.md 的 §11
「已核实的 ScanNet 事实」——HF 有没有场景、本地 zip 能不能训练、OpenGaussian 的数据布局
都已经查清，不要重复调研，直接从 §11.4 的唯一阻塞点开始。

【已有可复用资产】
- 3 个补丁 $R/patches/*.diff（CUDA include、PyG 2.3.0 的 keys 垫片、fp16 dtype）
- 评测 harness $R/scripts/eval_lerf_relags.py（LERF 用；ScanNet 指标别直接套，但聚合与渲染约定可参考）
- 完整的 LERF 复盘 $R/reports/01-08，以及已发布的 https://github.com/Youyuan287/ReLaGS-1023 的 reproduction/ 目录

【已知技术坑，别再踩】
- 光栅化器语义通道编译期固定 20，给 512 维会非法访存/静默截断；需要高维场时按 20 通道分块渲染再拼接。
- 服务器没有 unzip，用 python zipfile。
- huggingface_hub 直连不通，用 curl 打 hf-mirror.com 并 env -u http_proxy -u https_proxy -u all_proxy。
- 在本机 PowerShell 里不要把 ssh 输出管道给 head/tail/Select-String，把逻辑写进远端脚本再执行。
- configs/*.yml 与 scripts/*.sh 参数冲突是本仓库固有毛病，以脚本实际传参为准，并把冲突写进报告。
- 剪枝阈值用 0.0005（脚本口径），不要用 lerf.yml 里的 0.005。

【交付物】
reports/09_scannet_setup.md、reports/10_scannet_pipeline.md、reports/11_scannet_results.md
results/scannet_{query,scene}_metrics.csv、results/scannet_manifest.csv（sha256 溯源）
最后复制进 /root/rivermind-data/ReLaGS-1023/reproduction/ 并 commit（remote 已配 SSH，可直接 git push）。

【汇报方式】
每完成一个阶段给我：做了什么、真实数字、与论文的差、下一步。结论要有产物支撑，不要只给口头判断。
如果某步被阻塞（资产缺失/磁盘不足/协议无法确定），立刻说明具体阻塞点和你建议的取舍，不要静默绕过去。
```

---

## 10. LERF 阶段的最终数字（对照基线，便于判断 ScanNet 是否跑偏）

| Scene | 复现 | 论文 |
|---|---:|---:|
| figurines | 63.29 | 64.70 |
| ramen | 51.15 | 51.20 |
| teatime | 80.97 | 81.00 |
| waldo_kitchen | 60.62 | 60.60 |
| **Mean** | **64.01** | **64.40** |

协议：簇级检索 `topk=5 / level_until=3 / filter_small=50 / remove_small=True`，
掩膜 `semantics[0] > 0.5`，先逐图对 query 平均再对图平均。

---

## 11. 二次核实结果（2026-10-23 晚，覆盖 §3.1 / §5）

### 11.1 论文 Table 9 实测数值（原文抄录，`ref/paper.txt:1239`）

| Method | 19 类 mIoU / mAcc | 15 类 mIoU / mAcc | 10 类 mIoU / mAcc |
|---|---|---|---|
| THGS* | 39.33 / 54.23 | 43.61 / 61.55 | 52.52 / 70.50 |
| **Ours** | **41.26 / 57.29** | **45.24 / 62.75** | **54.04 / 71.48** |

协议（附录 §9 原文要点）：

1. ScanNet 场景训练时**开启 densification**（固定基元数重建质量差，Dr. Splat 已有同样观察）；
2. 评测时**把每个高斯赋为「最近 GT 点」的标签**（L2 距离；Dr. Splat 用 Mahalanobis）；
3. 指标是 **3D mIoU（点集 IoU，Eq.16）+ mAcc**，按 19/15/10 类子集报告；
4. `THGS*` = 作者在**同一协议下重跑**的 THGS；
5. 这是 **3D** 指标，不是 LERF 的 2D 多边形 IoU —— `eval_lerf_relags.py` 不能直接套用。

### 11.2 HF 资产核查（结论：没有 ScanNet）

`dfki-av/ReLaGS`（model repo，公开，1.59 GB）只有：

- `scenes/LeRF/<4 个场景>/{cameras.json,cfg_args,input.ply,nag-l1.pt,neighbor.pt,neighbor_new.pt,point_cloud/iteration_0/point_cloud.ply,sai_nag.pt}`
- `GNN_model/graph_transformer_512_80.pth`

**没有任何 ScanNet 场景、没有 ScanNet 的 `sai_nag.pt`** → 必须自己训练。

### 11.3 本地 ScanNet 数据核查（结论：不能用于训练）

`/root/rivermind-data/VALA/dataset/scannet.zip`，11.5 GB，7784 entries：

- `.sens` 文件数 = **0**；`/label/` 目录数 = 0；`/color/` 目录数 = 0
- 每个场景仅 5 个条目：`_vh_clean_2.ply`、`_vh_clean_2.labels.ply`、`.aggregation.json`、`.0.010000.segs.json`
- 未压缩体积构成：`scans/` 20.15 GB、`scans_test/` 0.74 GB、`tasks/` 0.64 GB

→ 它**只能提供 GT 标签**（正好对应 OpenGaussian 需要的 `*_vh_clean_2.labels.ply`），**无法训练 2DGS**（无 RGB 图像与位姿）。
→ 全盘搜索：无 `.sens`、无 ScanNet 图像目录；VALA 代码与其 ScanNet 工作目录已被删除，仅剩 `/tmp/vala_*_help.txt` 等残迹。

### 11.4 唯一阻塞点

**ScanNet 的 `color/`（RGB）+ `transforms_*.json`（位姿）+ `language_features/` 从哪来？**

| 选项 | 可行性 |
|---|---|
| (a) 手上还有 OpenGaussian 预处理包（OneDrive / Baidu） | 最快，直接铺开 |
| (b) 有 ScanNet 官方授权，下载 `.sens` 自行转换 | 体量大（数百 GB），需 `download-scannet.py` + `preprocess_2d_scannet.py` + `scannet2blender.py` |
| (c) 都没有 | ScanNet 无法开始 |

### 11.5 OpenGaussian 路线（用户指定参考）

仓库：**`github.com/yanmin-wu/OpenGaussian`**（用户名是 `yanmin-wu`，不是 `yanminwu`）

ScanNet 数据布局：

```
scannet/<scene>/color/                     # RGB 图像
              language_features/           # LangSplat 风格 SAM+CLIP 特征
              points3d.ply                 # 初始化点云
              transforms_train.json / transforms_test.json
              *_vh_clean_2.labels.ply      # GT（本地 zip 里已有）
```

预处理包：作者 OneDrive / Baidu（`color.zip` + `language_features.zip`），**无需 ScanNet 授权**；只有想扩场景才需要 `.sens`。

评测：`scripts/eval_scannet.py`（`target_id` 设 19/15/10），文本特征 `assets/text_features.zip`。

**关键差异**：OpenGaussian 走 3DGS（`diff-gaussian-rasterization`），ReLaGS 走 2DGS（`diff-surfel-rasterization`），**checkpoint 不可互换**；只能复用数据与评测思路，重建须用 ReLaGS 的 `train.py` 且**开启 densification**。

### 11.6 特征口径冲突（必须先定）

- ReLaGS `scripts/run_scannet.sh` 传 `--vlm_type openseg`（OpenSeg 特征）
- OpenGaussian 的 `language_features/` 是 LangSplat 的 **CLIP** 特征

两者不可混用。先确认 ReLaGS 论文 ScanNet 实验实际用哪个（查 `merge_proj.py` 两个分支、`dataset_readers.py` 读取路径、论文措辞），再决定下载哪套。

### 11.7 磁盘现状

`/root/rivermind-data` 50G，已用 33G，**剩余 18G**。本轮已删除并留档：

- `ReLaGS_Reproduction/work/`（2.8G，我自己的流水线中间产物，最终指标不依赖）
- `ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs.zip`（0.7G，已解压）
- `VALA/dataset/3dgs/lerf_ovs/*/langsplat`（5.6G）+ `*/distorted`（1.0G）——清单见 `results/vala_langsplat_inventory.txt`

`VALA/dataset/scannet.zip`（11.5G）**尚未删除**：它是 §11.4 选项 (b) 的唯一本地来源，确认走 (a) 后可释放。

### 11.8 服务器 2（`fj01-ssh.gpuhome.cc:30253`，代号 255）核查 — 2026-10-23

**结论：那里也没有 ScanNet 的 RGB/位姿数据，但找到了 VALA 的完整代码。**

1. **VALA 代码在 255 上**：`/root/rivermind-data/VALA`（30G），含 `run_scannet.sh`、`eval/scannet_utils.py`、
   `train.py`、`gaussian_feature_extractor.py`、`VALA_HANDOFF.md`、`VALA_REPRO_ISSUES.md`。
   （225 上只剩 `dataset/`，代码不在。）
2. **VALA 的 ScanNet 其实没跑成**：
   - `VALA_REPRO_PROGRESS.md` 第 26 行：`Stage 4 status: LERF complete; ScanNet pending`
   - `VALA_REPRO_ISSUES.md` R003 原文：*"VALA commit 48902a5 does not contain the dataset/scannet module …
     run_scannet.sh expects preprocessed scene directories. **ScanNet download is stopped pending an
     authoritative source/protocol.**"*
3. **数据同样缺失**：`VALA/dataset/3dgs` 只有 `lerf_ovs`（9.7G），没有 `scannet_langsplat`；
   `VALA/dataset/scannet.zip` 与 225 上那份**同一个 ply-only 包**（0 `.sens`、0 `color/`）；
   255 的 `/root/rivermind-data/scannet.zip`（747 MB）是**中断的下载**（无 EOCD，首条目仍是 `scannet/`，即同一归档）。
   全盘 `find / -name "*.sens"`、`-type d -name color` 均无结果。
4. **可复用的只有「跑法」和「类表」**，不是数据：

   VALA 的 ScanNet 流程（数据到手后可直接照抄）：
   ```bash
   python train.py -s dataset/3dgs/scannet_langsplat/$case \
       -m output/3dgs/scannet_langsplat/$case --iterations 30000 --eval
   python gaussian_feature_extractor.py -m output/3dgs/scannet_langsplat/$case \
       -s dataset/3dgs/scannet_langsplat/$case --iteration 30000 --eval \
       --feature_level 0 --use_efficient --tau_mass 0.9 --tau_abs 0.01
   ```

   `VALA/eval/scannet_utils.py` 里的 19/15/10 类子集索引（Table 9 用的就是这些）：
   ```python
   10: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 33]
   15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]
   19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
   ```
   （索引对应 `scannetv2-labels.combined.tsv` 的类别 id；0 为 ignore。）

5. **判断**：ScanNet 阶段的唯一缺口就是 **RGB 图像 + 位姿 + 语言特征**。
   在拿到 OpenGaussian 预处理包（或 ScanNet `.sens` 授权）之前，这项无法开始；
   `scannet.zip`（两个位置共约 12.3G）对本任务只剩「GT 标签」价值，可释放。
