# LERF 2D mask 提升探针报告（现有 SAM mask，不切换）

日期：2026-10-31
脚本：`analysis/lerf_lift_probe.py`（修正投影后重跑）；数据：`results/lerf_lift_20261031/`
方法：对每条查询，取查询视角中与 GT 最匹配的那张 2D mask（SAM 分割图 level-3），用可见高斯（render_point 权重 >1e-4）的投影上下文把该 mask 提升为 3D 高斯选择，渲染后与 GT 计算 IoU。
**重要修正**：此前的投影实现用错了旋转方向（`R^T`），导致 lift 结果全为 0、可表示性审计的绝对值偏小。正确形式为 `p_view = R @ p + T`，用 2D mask 自洽性验证（IoU(渲染, 2D mask)=0.902，IoU(渲染, GT)=0.907）。

## 1. 结果（208 查询）

| 族 | n | lift IoU 中位 | 2D 覆盖中位 | lift≥0.5 | lift≥0.7 |
|---|---:|---:|---:|---:|---:|
| found | 124 | 0.813 | 0.907 | 90 | 77 |
| rank2 | 36 | 0.519 | 0.504 | 19 | 12 |
| **deep** | **19** | **0.921** | 0.954 | **17** | 15 |
| recall_fail | 29 | 0.092 | 0.112 | 6 | 3 |

- lift 与 2D 覆盖相关系数 0.967：提升几乎无损地把 2D 覆盖转化为 3D 段。
- **deep 族**：现有 SAM mask 提升后 IoU 中位 0.921，而该族当前 main 仅 0.226——**候选是好的，缺的是把它选出来**。
- found 族 0.813（与 main 0.816 相当）：提升不会伤害已正确查询。
- rank2 0.519（低于 main 0.664）：单张查询视角 mask 不够，rank2 不应依赖该模块。
- recall 仅 6/29 ≥0.5：2D 源对这些部件/质量名词本身就不完整（与前一份 2D 覆盖审计一致）。

## 2. 含义

1. **SAM 不需要切换**：现有 mask 作为"masklet 候选"是可行且高质量的来源，与 VALA/ReLaGS 的"先有好候选/干净特征"思路一致，但用的是已经在数据集里的东西。
2. 该模块的目标族明确：**deep（19 条，上限 +0.67×19/208 ≈ +6 点）**，found 不伤害，rank2/recall 不依赖。
3. 下一步的关键问题不是"有没有好候选"，而是"**用哪个分数把 lifted 候选排出来**"：lifted 候选的特征可用其 2D mask 的 CLIP 特征（`view.semantic['sem'][mask_id]`），与 NAG 分数同属文本相似度，但特征源不同（蒸馏 vs 原始 CLIP），需要先做分数可比性探针，再做预注册的候选池扩展。

## 3. 建议的下一步（预注册）

**Masklet 候选模块（M1）**：
- 候选池 = released NAG top-5 ∪ 查询视角按文本相似度 top-3 的 lifted mask 候选；
- lifted 候选分数 = 其 2D mask 的 CLIP 特征与文本余弦；NAG 候选用原分数；合并前做层内标准化（先做可比性探针确认）；
- 选择：合并序上的 released gap（主），C 模块变体（次）；
- 门控：场景均值 Δ≥+0.01、CI 下界>0、无场景恶化、found 不退化；分族报告 deep/rank2/recall。
- 预期：主要来自 deep 族（候选 IoU 0.92），上限约 +3~6 场景均值；风险是跨源分数不可比（需先探针）。

## 4. 产物

- `results/lerf_lift_20261031/`（4 CSV）；`results/lerf_representability_20261031/`（修正后重跑）
- 脚本：`analysis/lerf_lift_probe.py`、`run_lerf_lift_probe.sh`；`analysis/lerf_2dmask_coverage.py`、`run_lerf_2dmask_coverage.sh`