# LERF 可表示性审计报告（修正版）

日期：2026-10-31（初版 2026-10-30；**初版使用的相机投影实现有误——旋转方向用成了 `R^T`，导致绝对值偏小；本版为修正投影 `p_view = R·p + T` 后重跑的结果**，并用 lift 自洽性验证：IoU(渲染, 2D mask)=0.902）
脚本：`analysis/lerf_representability_audit.py`；数据：`results/lerf_representability_20261031/`
方法：对每条查询，`render_point` 取逐高斯渲染权重，保留可见高斯，投影到查询视角，统计落入 GT 掩膜的支持。

## 1. 结果（按失败族）

| 族 | n | n_vis_gt 中位 | 不透明度质量中位 | 每像素支持 | 像素覆盖 |
|---|---:|---:|---:|---:|---:|
| found（已正确） | 124 | 2626 | 2006 | 0.280 | 0.216 |
| rank2 | 36 | 3697 | 2720 | 0.211 | 0.181 |
| deep（深层排序失败） | 19 | 2427 | 1626 | 0.228 | 0.172 |
| **recall_fail（池内无正确候选）** | **29** | **1624** | **1134** | **0.218** | **0.179** | 

**核心发现（修正后依然成立）：召回失败查询的物体支持并不低于成功查询**，所有族零支撑比例均为 0%。"物体没被重建"的假设不成立。

## 2. 结论

1. 可表示性不是主瓶颈；29 条召回失败的提示词全部是部件/复数/质量名词（onion segments×7、corn×5、porcelain hand×3、bear nose×3、door handle×2、hooves×2、knife×2、napkin×2 等）。
2. 瓶颈在特征的判别力与表示粒度，以及（对 18/29 召回查询）查询视角的 2D 掩膜覆盖不足——见 `LERF_2DMASK_COVERAGE_REPORT.md`。
3. 在正确投影下，现有 SAM mask 的提升质量很好（deep 族 lift IoU 中位 0.921），见 `LERF_LIFT_PROBE_REPORT.md`——候选来源是够的，缺的是把它选出来。

## 3. 产物

- `results/lerf_representability_20261031/`（修正后重跑）
- 脚本：`analysis/lerf_representability_audit.py`、`run_lerf_representability_audit.sh`