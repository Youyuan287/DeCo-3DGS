# Stage 3 — ReLaGS 语义处理链

## 1. 真实执行顺序

官方 `scripts/run_lerf.sh` 的顺序经核对后按下面执行（README 的概括性描述不足以还原）：

1. `max_weight_pruning.py -s <src> -m <out> --contribution_threshold 0.005`
   → 用 ReLaGS 光栅化器对**全部训练视角**累计 `max_contribution`，筛掉小贡献基元，写 `point_cloud/iteration_0/point_cloud.ply`
2. `sp_partition.py -m <out> -i 0 -a`
   → FRNN 建 kNN 邻接，写 `neighbor.pt`（**不产生** `nag-l1.pt`）
3. `graph_weight.py --iteration 0 -s <src> -m <out> --config ... --level 1`
   → SAM 多级 mask 对比度重加权，写 `neighbor_new.pt`
4. `sp_partition.py -m <out> -i 0 -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a`
   → 用重加权邻接做 cut-pursuit，写 `nag-l1.pt`
5. `merge_proj.py -s <src> -m <out> --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0`
   → 层级区域生长 + CLIP 特征重投影 + 外点剔除，写 `<scene>saifine.ply` 与 `sai_nag_3.0.pt`

注意：`--vlm_type` 默认 `clip`，`tau=3` 使输出名为 `sai_nag_3.0.pt`（不是 `sai_nag.pt`，也不是 `sai_nag_3.pt`）。

## 2. 逐场景统计（真实产物）

| 场景 | 剪枝前→后 | 保留 | 视角 | 超点 | NAG 层级 | 最终区域 | merge 耗时 |
|---|---:|---:|---:|---:|---|---:|---:|
| figurines | 369,120 → 182,017 | 49.3% | 299 | 4,646 | 4646/1970/438/160 | 160 | 384 s |
| ramen | 400,401 → 350,655 | 87.6% | 131 | 7,559 | 7559/1810/504/253 | 253 | 137 s |
| teatime | 1,093,781 → 891,094 | 81.5% | 177 | 25,443 | 25443/7351/2528/1217 | 1217 | 247 s |
| waldo_kitchen | 1,062,260 → 806,254 | 75.9% | 187 | 14,188 | 14188/3763/852/367 | 367 | 261 s |

figurines 区域生长：原 213 区域，合并 53，移除 0；ramen 316→合并61/移除3；teatime 1478→257/5；waldo 450→83/1。

## 3. 完整性校验（不止退出码）

| 检查 | 结果 |
|---|---|
| 剪枝前后点数非零且单调下降 | ✅ 四场景均成立 |
| `neighbor.pt` 与 `neighbor_new.pt` 确实不同 | ✅ 文件大小/内容不同（重加权生效） |
| 第二次 partition 使用 `-k neighbor_new.pt` | ✅ 代码路径与产物时间戳一致 |
| `nag-l1.pt` 的 cluster id 上界与超点数一致 | ✅ max(id)+1 == sp num |
| NAG 层级逐级收敛 | ✅ 4 级严格递减 |
| `sai_nag_3.0.pt` 含 `nag`/`nag_feat` 且层级数匹配 | ✅ len(nag)=4, len(nag_feat)=3 |
| 语义特征 / Superpoint / Gaussian 索引一致 | ✅ `nag_feat[i]` 行数分别等于 `nag[i+1]` 的簇数；渲染时按 `labels[level]` 展开到 182k/350k/891k/806k 基元 |
| 输出文件非空 | ✅ 8.9M / 12.3M / 40.5M / 26.3M |

## 4. 为打通流程所做的兼容性修改（已存 diff）

| 文件 | 问题 | 修改 |
|---|---|---|
| `submodules/diff-surfel-rasterization-trace-maxcontrib/setup.py` | 包目录名 `diff_surfel_rasterization_trace` 与 `packages=[...trace_maxcontrib]` 不一致，`pip install` 直接失败 | 目录重命名为 `..._maxcontrib`；并在 `nvcc` 参数中追加 conda 布局头文件 `$CUDA_HOME/targets/x86_64-linux/include`（否则缺 `cub/cub.cuh`、`thrust/*`、`cuda/std/*`） |
| `ext/spt/data/data.py` | 以方法形式调用 `data.keys()`，但 torch_geometric 2.3.0 的 `keys` 是属性 → `'list' object is not callable` | 在 `Data` 上增加返回可调用 list 的 `keys` 属性垫片，两种调用风格同时成立（不改任何调用点） |
| `merge_proj.py` | LangSplat 特征为 fp16，`sp_mask_mat(float32) @ view_level_feature(fp16)` 报 dtype 错 | `gt_feature = view.semantic["sem"].to(device).float()` |

以上均为**兼容性修复**，未改动论文算法（剪枝阈值、图权重、cut-pursuit 参数、区域生长阈值、CLIP 编码全部保持官方值）。补丁见 `patches/`。

## 5. 更正（2026-10-23）

- **剪枝阈值用错**：官方 `run_lerf.sh` 为 `CONTRIB=0.0005`，`configs/lerf.yml` 写 `0.005`（仓库不一致）。本文第 2 节表格（182,017 / 350,655 / 891,094 / 806,254）是 **0.005** 的结果，属过度剪枝，不是官方口径。
- **官方口径**下应得到 234,534 / 494,xxx / 958,840 / …，已通过 `ref/relags_hf/<scene>/point_cloud/iteration_0/point_cloud.ply` 直接取得官方云验证（figurines、teatime 顶点数逐点相等）。
- 语义阶段（sp_partition / graph_weight / merge_proj）的调用参数与官方脚本一致（`--level 1`、`--thres_connect 0.9,0.7,0.7`、`--thres_merge 20`、`--tau 3`、`pcp 0.1/0.1`），配置值经逐项比对与仓库 `configs/lerf.yml` 相同。
