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
- `ref/paper.txt:1500–1512` 3D 基准（ScanNet / ScanNet++）的评测方式说明（预测如何映射到 GT）。

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