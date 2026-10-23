# Stage 1 — 环境与资产审计

主机 `jupyter-or4nv8tw46dfjtzj`（root），审计时间 2026-10-22 → 2026-10-23。所有命令经 `ssh -p 30946` 远程执行。

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | RTX 4090, 24564 MiB（单卡） |
| 驱动 | 580.119.02 |
| 系统盘 `/` | 30G（可用 8G） |
| 数据盘 `/root/rivermind-data` | 50G（**可用 2.6G / 95% 占用**） |

磁盘是本次最硬约束：无法从零训练 4 个场景的 2DGS，也无法新建完整 conda 环境（约需 10–13G）。

## 2. 软件栈

| 项 | 值 |
|---|---|
| Python | 3.10.13（`/root/rivermind-data/thgs_mamba_root/envs/thgs`） |
| PyTorch | 2.2.0+cu118，CUDA 可用 |
| nvcc | 11.8（conda env 内），系统仅有 CUDA 12.1（与 torch cu118 不匹配，不可用） |
| gcc | 11.4.0 |
| 网络 | GitHub 200 / PyPI 200 / **huggingface.co 不可达** / hf-mirror 200；git 全局代理 `127.0.0.1:17894` 失效，需 `-c http.proxy=` 绕过 |

## 3. 现有资产

| 资产 | 路径 | 状态 |
|---|---|---|
| THGS | `/root/rivermind-data/THGS`（12G） | 无 `.git`；已被前次实验修改（`test_lerf.py` 含 quality-aware 评分、`merge_proj.py` 59KB、多个 `.bak` 与 sweep 脚本） |
| LERF-OVS | `/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs` | 4 场景 + `label/` GT |
| language_features | `lerf_ovs/<scene>/language_features` → `langsplat/language_features` | LangSplat 格式：`_f.npy`(fp16 CLIP ViT-B-16) + `_s.npy`(SAM 多级 mask)；598/262/354/374 文件 |
| GT | `lerf_ovs/label/<scene>/frame_*.jpg+.json` | 每场景 4–7 帧，`objects[].category` + `segmentation` 多边形 |
| 2DGS 场景 | `THGS/output/lerf/<scene>/point_cloud/iteration_30000/point_cloud.ply` | 369,120 / 400,401 / 1,093,781 / 1,062,260 Gaussians |
| ReLaGS | `ReLaGS_Reproduction/ReLaGS` | commit `52cc0c98b0ccf9afebea31901bf79246650286b3` |

## 4. checkpoint 兼容性（逐项实测）

PLY 字段：`x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`，`sh_degree=3`，标准 2DGS surfel。

| 检查 | 结论 |
|---|---|
| PLY 字段与维度 | ✅ 与 ReLaGS `GaussianModel.load_ply` 一致 |
| rotation/scaling/opacity 存储 | ✅ 四元数 / 对数缩放 / logit opacity |
| renderer 能否加载 | ✅ 实测通过（`max_weight_pruning.py` 渲染 299 视角无异常） |
| 相机-图像-GT 对应 | ✅ `cameras.json` 与 GT 帧名逐帧匹配 |
| `cfg_args` | ⚠️ 原值为前机器路径，已按本机重写 |
| 预编译扩展复用 | ✅ THGS 的 `FRNN/_C`、`grid_graph`、`cp_d0_dist_cpy`（py3.10）在 sm_89 实测可用 |
| THGS 的 `diff_surfel_rasterization` | ❌ 无 `max_contribution` 通道，不能替代 ReLaGS `trace_maxcontrib` 变体 |

## 5. 复用结论

**直接复用（只读）**：LERF 原始数据、`language_features`、GT 标注、2DGS `iteration_30000`、THGS 编译好的 CPU/FRNN 扩展。

**必须新建**：ReLaGS 仓库、隔离 venv、`trace_maxcontrib` CUDA 扩展、每场景的 `iteration_0`/`neighbor*.pt`/`nag-l1.pt`/`sai_nag_3.0.pt`、评测脚本与结果。

**明确不复用**：THGS 的 `sai_nag.pt`、THGS 预测 mask、THGS 改过的 `test_lerf.py`（协议已被前次工作改动）。
