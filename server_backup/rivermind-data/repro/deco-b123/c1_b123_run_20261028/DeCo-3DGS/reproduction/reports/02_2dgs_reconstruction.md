# Stage 2 — 2DGS 几何底座

## 1. 官方权重可行性核查

- `huggingface.co` 在服务器上不可达（`000`），`huggingface_hub` 下载失败；且 HF 仓库 `dfki-av/ReLaGS` 仅含 `point_cloud/iteration_0`（剪枝后产物）与 `nag-l1.pt / neighbor*.pt / sai_nag.pt`，**不含** `iteration_30000` 几何底座，也无 `language_features`。
- 因此官方权重无法直接作为「可继续跑完整流水线」的输入。

## 2. 决策：复用官方 2DGS 的 LERF 场景

THGS 复现时使用的是 **THGS 作者自己打包**的 ready-to-use 场景（不是 2DGS 官方发布物）（其 README 所述“ready-to-use scenes（含 2DGS + semantic field）”），即由 2DGS 官方代码训练到 iteration 30000 的产物，由 THGS 作者打包分发。本任务直接复用其**几何部分**（`iteration_30000/point_cloud.ply`），按场景复制到 `work/lerf/<scene>/`，并重写 `cfg_args` 指向本机数据路径。

**未从零训练 30000 迭代**，原因与影响：
- 磁盘仅剩 2.6G，4 场景训练（含中间快照）至少需 10G+；
- 复用对象本身即官方 2DGS 输出，属于“同一几何底座”，不是自行替换的近似；
- 残余风险：ReLaGS 论文可能使用其自改 rasterizer 重新训练的底座，此点无法从公开资料确认，已列入 05 的不确定性清单。

## 3. 底座校验（不是只看退出码）

| 场景 | Gaussians | 视图数 | ReLaGS renderer | 剪枝后 |
|---|---:|---:|---|---:|
| figurines | 369,120 | 299 | ✅ | 182,017 (49.3%) |
| ramen | 400,401 | 131 | ✅ | 350,655 (87.6%) |
| teatime | 1,093,781 | 177 | ✅ | 891,094 (81.5%) |
| waldo_kitchen | 1,062,260 | 187 | ✅ | 806,254 (75.9%) |

渲染校验方式：`max_weight_pruning.py` 用 ReLaGS 的 `trace_maxcontrib` 光栅化器对全部训练视角做前向渲染并累计 `max_contribution`，全部视角无 NaN/越界，输出 ply 可被后续 `Scene` 正常加载——这说明几何、相机、法向与 rasterizer 三者一致。

## 4. 产物

- 工作目录：`ReLaGS_Reproduction/work/lerf/<scene>/point_cloud/iteration_30000/point_cloud.ply`（只读来源，不修改 THGS）
- `cfg_args`：已重写为 `source_path=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/<scene>`
- PSNR/SSIM 未做（任务说明中非必需），底座正确性由“可渲染 + 可剪枝 + 后续语义链路一致”间接确认。

## 5. 来源核查（2026-10-23 补充，含对前文的更正）

**更正**：本报告前文写成「2DGS 官方为 LERF-OVS 提供的 ready-to-use 场景」并不准确。THGS README 第 62 行给出的是 **THGS 作者自己打包**的 ready-to-use 场景（Google Drive，含「2DGS + semantic field」），不是 2DGS 官方发布物。前次 THGS 复现下载解压了这四份包（`REPRODUCTION_SUMMARY_SOURCE.md` 第 90 行自述：*using available ready-to-use 2DGS scene folders rather than retraining 2DGS from scratch*），本任务只取其中的几何 ply。

证据：

| 证据 | 观察 |
|---|---|
| 目录构成 | `output/lerf/<scene>/` = `cameras.json` + `cfg_args` + `input.ply` + `point_cloud/iteration_30000/point_cloud.ply`（+THGS 自产 `sai_nag.pt`/`*saifine.ply`） |
| 无训练痕迹 | 全树无 `chkpnt*.pth`、无 `train/ours_*` 渲染、无 tensorboard `events.out*` |
| mtime | 四场景 ply 与 cameras.json 集中在 `2026-09-16 03:32:29–34`（解压式一次性写入，非逐小时训练） |
| `cfg_args` | `source_path='/data/dsh/datasets/lerf/figurines'`，是作者机器的路径，非本机 |
| `input.ply` | 存在（COLMAP 初始化点云），符合 2DGS 场景目录结构 |
| ply 字段 | 仅几何（含 `nx,ny,nz` 法向与 45 维 `f_rest`），**无任何语义字段** |

因此：**几何高斯场来自 THGS 作者发布的 ready-to-use 包**（其内部由 2DGS 官方代码训练），**语义场是我用 ReLaGS 在剪枝后的基元上重算的**（`sai_nag_3.0.pt`），没有复用 THGS 的 `sai_nag.pt`。

## 6. 更正与实测结论（2026-10-23，覆盖第 2 节的部分表述）

此前我根据“官方 `iteration_0` 点数比 THGS 的 30k 还大”推断底座是“另一套 2DGS 训练结果”，**该推断错误**：我拿官方**剪枝后**的云去比 THGS **未剪枝**的 30k。实测：

| 场景 | THGS 30k | MWP@0.0005 结果 | 官方 `iteration_0` |
|---|---:|---:|---:|
| figurines | 369,120 | **234,534** | **234,534** |
| teatime | 1,093,781 | **958,840** | **958,840** |

顶点数逐点相等 → **两者是同一套 2DGS 30k 底座**，官方发布的是其剪枝后的 `iteration_0`。

同时发现我此前的剪枝阈值用错：`scripts/run_lerf.sh` 里是 **`CONTRIB=0.0005`**，而 `configs/lerf.yml` 写的是 `0.005`（仓库自身不一致）。我此前信了 yaml 用 0.005，导致 figurines 我只留 182,017 点（官方 234,534）、teatime 891,094（官方 958,840）——**过度剪枝**。

获取官方底座的方式（已落地）：hf-mirror 可用，`https://hf-mirror.com/dfki-av/ReLaGS/resolve/main/scenes/LeRF/<scene>/...`，落盘于 `ref/relags_hf/<scene>/`。
