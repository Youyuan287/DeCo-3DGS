> **Archived — superseded.** This was an internal handoff written before the ScanNet phase ran;
> the numbers and the "nothing downloaded yet" status in it are out of date. Kept for the record.
> Current results: [`deco/RESULTS.md`](../../../deco/RESULTS.md).

---

# ReLaGS · ScanNet 复现执行交接（服务器 TZJ）

> 新窗口直接照此执行。LERF-OVS 已完成（Mean **64.01** vs 论文 64.40），本阶段只剩 ScanNet。
> 上文探索记录见同目录 `09_scannet_handoff.md`（含证据链），本文件是**执行版**。

---

## 0. 一句话现状

ScanNet 数据授权已拿到、官方下载脚本已就位，但**一帧 RGB 都还没下**；SRC 里没有 ScanNet 场景，
HF 上也没有，所以必须自己走「下载 → 转格式 → 训 2DGS → 语义链路 → 3D 评测」全流程。

---

## 1. 机器与环境

```bash
ssh root@hn01-ssh.gpuhome.cc -p 30946        # 代号 TZJ / 225，有 GPU，本任务主战场
```

| 项 | 值 |
|---|---|
| 项目根 | `/root/rivermind-data/ReLaGS_Reproduction`（记作 `$R`） |
| 上游代码（已打补丁 + 已编译） | `$R/ReLaGS` @ `52cc0c9` |
| Python | `$R/venv/bin/python`（3.10 / torch 2.1 / cu118 / PyG 2.3.0） |
| 发布仓库 | `/root/rivermind-data/ReLaGS-1023`，remote 已配 SSH，可直接 `git push` |
| 磁盘 | `/root/rivermind-data` 50G，**剩余 18G**（紧张，见 §9） |
| 参考服务器（VALA 代码） | `ssh root@fj01-ssh.gpuhome.cc -p 30253`（代号 255，**无 GPU**） |

跑任何 ReLaGS 脚本前：

```bash
cd $R/ReLaGS
export PYTHONPATH=$R/ReLaGS/ext:$R/ReLaGS
export HF_HUB_OFFLINE=1
```

---

## 2. 目标指标（论文 Table 9，`ref/paper.txt:1239`）

| Method | 19 类 mIoU / mAcc | 15 类 mIoU / mAcc | 10 类 mIoU / mAcc |
|---|---|---|---|
| THGS* | 39.33 / 54.23 | 43.61 / 61.55 | 52.52 / 70.50 |
| **Ours** | **41.26 / 57.29** | **45.24 / 62.75** | **54.04 / 71.48** |

协议（论文附录 §9 原文要点，务必照做）：

1. 训练 ScanNet 场景时**开启 densification**（固定基元数重建质量差）；
2. 评测时**把每个高斯赋为「最近 GT 点」的标签**（**L2 距离**；Dr. Splat 用 Mahalanobis）；
3. 指标为 **3D mIoU（点集 IoU，Eq.16）+ mAcc**，分 19/15/10 类子集；
4. `THGS*` = 作者在同协议下重跑的 THGS；
5. 这是 **3D** 指标，**不能**用 LERF 的 `eval_lerf_relags.py`。

类子集索引（来自 VALA `eval/scannet_utils.py`，对应 `scannetv2-labels.combined.tsv` 类别 id，0 为 ignore）：

```python
10: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 33]
15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]
19: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
```

---

## 3. 已核实的事实（不要再重复调研）

1. **HF `dfki-av/ReLaGS` 没有 ScanNet**：只有 `scenes/LeRF/*`（4 个 LERF 场景）+ `GNN_model/graph_transformer_512_80.pth`，1.59 GB。
2. **本地 `scannet.zip` 只有标签**：11.5 GB / 7784 entries，`.sens` = 0、`/color/` = 0；每场景仅 5 个文件
   （`_vh_clean_2.ply`、`_vh_clean_2.labels.ply`、`.aggregation.json`、`.0.010000.segs.json`）。
   → 可作 GT，**不能训练**。第二个 `scannet.zip`（747 MB）是它的中断下载，可删。
3. **VALA 也没跑成 ScanNet**：`VALA_REPRO_PROGRESS.md:26` = "LERF complete; ScanNet pending"；
   `VALA_REPRO_ISSUES.md` R003 = "ScanNet download is stopped pending an authoritative source/protocol"。
4. **OpenGaussian 是正确参考**（`github.com/yanmin-wu/OpenGaussian`，注意是 `yanmin-wu`）。其 ScanNet 布局：
   ```
   scannet/<scene>/color/  language_features/  points3d.ply
                 transforms_train.json  transforms_test.json  *_vh_clean_2.labels.ply
   ```
   可用官方 `.sens` 转换：`preprocess_2d_scannet.py`（OpenScene）+ `scannet2blender.py`（OpenGaussian）。
5. **VALA 的 ScanNet 跑法**（数据到位后可照抄，3DGS 版；ReLaGS 要换 2DGS）：
   ```bash
   python train.py -s dataset/3dgs/scannet_langsplat/$case -m output/3dgs/scannet_langsplat/$case --iterations 30000 --eval
   python gaussian_feature_extractor.py -m output/3dgs/scannet_langsplat/$case -s dataset/3dgs/scannet_langsplat/$case \
       --iteration 30000 --eval --feature_level 0 --use_efficient --tau_mass 0.9 --tau_abs 0.01
   ```

---

## 4. 已就位的下载工具

```bash
/root/rivermind-data/downloads/download-scannet.py     # 244 行，官方脚本，-h 已验证可用
mkdir -p /root/rivermind-data/scannet_raw
```

10 个场景（ReLaGS `scripts/run_scannet.sh` 的列表）：

```
scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00
scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00
```

下载命令（**先测一个场景的体积和耗时**，再批量）：

```bash
cd /root/rivermind-data
$R/venv/bin/python downloads/download-scannet.py -o scannet_raw --id scene0000_00 --type .sens
du -sh scannet_raw/*; df -h /root/rivermind-data | tail -1
```

> `--id` 与 `--type .sens` 的组合**尚未实测**；若被忽略会下载该场景全部类型（体积翻几倍）。
> 若体积超预算：分批下载、或只下 `.sens` 后立刻转格式并删原始包。

---

## 5. 执行步骤（含验收标准）

**Step 1 · 下载**
- 先单场景验证脚本行为与体积，再批量 10 个；
- 验收：每场景 `<scene>.sens` 存在、大小合理，记录 `du` 与耗时。

**Step 2 · `.sens` → 2D 帧**
- 用 OpenScene 的 `preprocess_2d_scannet.py`（需 `.sens` reader：`imageio` + `scannetutils`），
  产出 `color/`、`pose/`、`intrinsic/`；
- 验收：图像数与 `.sens` 帧数一致，位姿矩阵数量一致，抽 1 帧目视/数值检查分辨率。

**Step 3 · → Blender 格式**
- 用 OpenGaussian 的 `scripts/scannet2blender.py`（有 `TODO`，需改路径），产出
  `points3d.ply` + `transforms_train.json` / `transforms_test.json`；
- 验收：`ReLaGS` 的 `dataset_readers` 能直接读该目录（用 `-s` 试跑一次不训练的加载）。

**Step 4 · 训练 2DGS（关键：开 densification）**
- 用 `$R/ReLaGS/train.py`，`--iterations 30000`，**开启 densification 相关参数**（与论文一致；
  不要用固定基元数），产出 `output/scannet/<scan>/point_cloud/iteration_30000/point_cloud.ply`；
- 再跑 `max_weight_pruning.py --contribution_threshold 0.0005`（脚本口径，**不要用 yml 的 0.005**）；
- 验收：ply 顶点数、渲染出一帧与 RGB 对齐。

**Step 5 · 语义链路**

```bash
python sp_partition.py     -s <data>/<scan> -m <model>/<scan> --iteration 30000 -a
python graph_weight.py     --iteration 30000 -s <data>/<scan> -m <model>/<scan> --config ./configs/scannet.yml --level 1
python sp_partition.py     -s <data>/<scan> -m <model>/<scan> --iteration 30000 -k neighbor_new.pt \
                           --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a
python merge_proj.py       -s <data>/<scan> -m <model>/<scan> --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 \
                           --vlm_type <openseg 或 clip> --iteration 30000
```

- ⚠️ 上游 `run_scannet.sh` 有三处不一致（`-m ./output/lerf/...`、`--config lerf.yml`、`--vlm_type openseg`），
  `configs/scannet.yml` 另有一套参数（`thres_merge: 100`、`pcp_regularization: 0.3`）。**以论文为准并记录取舍**。
- 验收：`sai_nag.pt` 生成、层级数/簇数与 LERF 同级结构一致。

**Step 6 · 评测（3D mIoU）**
- 读 `*_vh_clean_2.labels.ply`（本地 `scannet.zip` 里有）作为 GT；
- 每个高斯按 **L2 最近点**取 GT 标签；
- 按 19/15/10 类子集算 3D mIoU + mAcc，与 Table 9 对齐；
- 验收：产出逐场景 + 均值表，附预测/GT 点数与混淆矩阵。

---

## 6. 动手前必须先定的一件事：**openseg 还是 CLIP**

- ReLaGS `run_scannet.sh` 传 `--vlm_type openseg`（OpenSeg 特征，ViT-L-14-336 体系）；
- OpenGaussian 的 `language_features/` 是 LangSplat 的 **CLIP** 特征；
- 两者特征空间不同，**不能混用**。

查证顺序：`grep -n "openseg" merge_proj.py utils/* dataset_readers.py` → 看两条分支各自读什么目录/维度 →
对照论文 ScanNet 段落措辞 → 决定下载/生成哪套特征。**结论要写进报告并给证据**。

---

## 7. 硬性规则

1. `VALA/`（两台机器）、`THGS/` **只读**，禁止修改/删除；写入只在 `$R` 与 `ReLaGS-1023`。
2. **不伪造论文数字**，引用必须给 `paper.txt` 行号。
3. **不按 GT 调参**；参数取代码默认值或论文原文，扫描结果只能标 `ORACLE`。
4. 每个结论要有产物（CSV/JSON/日志 + sha256 manifest）。
5. 磁盘先算后动手；接近写满时先停并汇报，不要赌。
6. 自动推进，但**阻塞要说清具体阻塞点和建议取舍**，不要静默绕过。

---

## 8. 已知技术坑（LERF 阶段踩过）

1. 光栅化器语义通道**编译期固定 20**，给 512 维会非法访存/静默截断；需要高维场时按 20 通道分块渲染再拼接。
2. 服务器**没有 `unzip`**，用 Python `zipfile`。
3. `huggingface_hub` 直连不通；用 `curl` 打 `hf-mirror.com` 且 `env -u http_proxy -u https_proxy -u all_proxy`。
4. 本机 PowerShell **不要把 ssh 输出管道给 `head/tail/Select-String`**；逻辑写进远端脚本，用 base64 heredoc 下发。
5. `configs/*.yml` 与 `scripts/*.sh` 参数冲突是本仓库固有毛病，以脚本为准并记录差异。
6. 剪枝阈值用 **0.0005**（脚本口径），不是 yml 的 0.005。
7. 3 个兼容补丁已在 `$R/ReLaGS` 生效：CUDA include、PyG `keys` 垫片、fp16 dtype；`patches/` 有 diff。
8. 不要把 `figurines` 那类"自己生成特征"的弯路再走一遍——特征必须来自官方或作者发布的预处理包。

---

## 9. 磁盘预算（18G，必须先算）

- `.sens` 单场景量级未知（Step 1 测），10 个场景可能 5–15G；
- 转换产物（`color/`）每场景几百 MB；训练输出每场景数百 MB；语义中间产物每场景 0.3–1G；
- 建议：**先做 2–3 个场景跑通全链路**，再决定是否扩到 10 个；每个阶段做完即删中间原始包；
- 可立即释放：`/root/rivermind-data/scannet.zip`（747MB 中断下载）、`VALA/dataset/scannet.zip`（11.5G ply 包，
  仅在 255 上；若确认只用官方 `.sens`，可删）。

---

## 10. 交付物

```
$R/reports/10_scannet_data.md        下载与转换记录（体积/耗时/sha256/踩坑）
$R/reports/11_scannet_pipeline.md    训练与语义链路（参数、与上游不一致处、取舍理由）
$R/reports/12_scannet_results.md     19/15/10 类 3D mIoU + mAcc vs Table 9
$R/results/scannet_scene_metrics.csv
$R/results/scannet_manifest.csv
```

完成后复制进 `/root/rivermind-data/ReLaGS-1023/reproduction/`，commit 并 `git push`（remote 已配 SSH）。

---

## 11. 提示词（粘贴到新窗口）

```
在这个服务器上继续 ReLaGS 的 ScanNet 复现，全程自动推进，不要每步问我。

【连接与环境】
ssh root@hn01-ssh.gpuhome.cc -p 30946
R=/root/rivermind-data/ReLaGS_Reproduction
跑脚本前：cd $R/ReLaGS && export PYTHONPATH=$R/ReLaGS/ext:$R/ReLaGS && export HF_HUB_OFFLINE=1
有 GPU；磁盘只剩 18G，任何下载/解压前先 df -h 并估算。

【先读这份交接】
/root/rivermind-data/ReLaGS-1023/reproduction/reports/10_scannet_exec_handoff.md
里面有：Table 9 的目标数值与协议（densified 重建 + 最近 GT 点 L2 赋标签 + 3D mIoU/mAcc，19/15/10 类）、
已核实的事实（HF 没有 ScanNet 场景、本地 zip 只有标签、VALA 也没跑成）、
VALA 的跑法与类表、6 步执行计划、10 条已知坑。不要重复调研。

【硬性约束】
1. VALA/ 和 THGS/ 只读，禁止改删；写入只在 $R 和 /root/rivermind-data/ReLaGS-1023。
2. 不伪造论文数字；引用给 $R/ref/paper.txt 行号。
3. 不按 GT 调参；参数取代码默认或论文原文，扫描只能标 ORACLE。
4. 自动推进，但被阻塞时立刻说明具体阻塞点和建议取舍，不要静默绕过。

【立刻开始】
Step 1：用 /root/rivermind-data/downloads/download-scannet.py 先下 scene0000_00 的 .sens，
量出单场景体积与耗时，据此判断 10 个场景能否放进 18G，再决定批量策略。
之后按交接文档 §5 依次做：.sens → color/pose → blender 格式 → 2DGS 训练（开 densification）→
MWP 0.0005 → sp_partition/graph_weight/sp_partition -k/merge_proj → 最近点赋标签的 3D 评测。

【动手前必须先回答】
ReLaGS 的 ScanNet 用的是 openseg 特征还是 CLIP 特征？
依据：run_scannet.sh 传 --vlm_type openseg，但 OpenGaussian 的 language_features 是 CLIP。
去 grep merge_proj.py / dataset_readers.py 的两条分支 + 论文措辞，给出结论和证据，再决定下载/生成哪套特征。

【汇报】
每个阶段给我：做了什么、真实数字、与论文差多少、下一步。结论要有产物支撑。
```

---

## 12. LERF 基线（判断是否跑偏的参照）

| Scene | 复现 | 论文 |
|---|---:|---:|
| figurines | 63.29 | 64.70 |
| ramen | 51.15 | 51.20 |
| teatime | 80.97 | 81.00 |
| waldo_kitchen | 60.62 | 60.60 |
| **Mean** | **64.01** | **64.40** |

协议：簇级检索 `topk=5 / level_until=3 / filter_small=50 / remove_small=True`，掩膜 `semantics[0] > 0.5`。