# Metadata 消融 fresh code review

2026-11-02。结论：**SAFE_TO_RENDER_SANITY / same-family / provisional**。未发现阻止按当前冻结方案部署的具体错误。先执行 figurines 首条 sanity；A/S/J 精确复现及五臂固定数量通过后，再执行四场 208 条 full。本审查没有启动远端作业或渲染，不能据此宣称消融有效。

审查范围：本轮 `PREREG.md`、`make_predictions.py`、`render_predictions.py`、`launch.sh`、`summarize.py`；原 `train_ranker.py`、保存的 S/J 模型、完整 features/predictions/CSV、原 renderer 与共享统计实现；已归档 NAG support 和 Scene 初始化实现。按工作区 ARIS experiment-bridge 的 fresh reviewer 范围检查，不做新颖性认证。

## BLOCKING

**无。** 代码与实际预测文件均符合这次两列直接贡献消融，未发现新训练、heldout GT 分支选择、输出数量变化或排序规则变化。

## NON-BLOCKING

- `summarize.py` 当前自动报告覆盖总体、场景、family、胜负平与集合变化计数，但没有逐 query 列出“原主要失误是否修复及新增损失”。PREREG 要求该诊断，最终结果解释应从真实 full CSV 补充四个变化 query（ramen/hand、teatime/hooves、waldo_kitchen/plastic ladle、waldo_kitchen/spoon）的得失。原始预测与 CSV 已足够支持补充，不阻止渲染，也不要求新增实验或改评分器。

## 正确性依据

- **干预位置正确。** `score_from_inputs` 先调用保存参数的 `apply_preprocessor`，随后仅将设计矩阵第 17、19 列（零基，`log1p_cover_views`、`fallback_seg`）置零。其他 scalar 列及 J 的完整语义块未变；系数、imputer、scale、active、block_divisor 均从原 NPZ 读取。新分数等于原分数减去两列直接贡献。
- **无新拟合或选择泄漏。** 导入原训练模块不会调用其 main；本轮只使用加载、预处理和矩阵乘法。原 loader 确实读取归档 GT 来验证 join 并计算未使用的 `y/mass`，但它们没有进入 `score_from_inputs`、排序、g_A 或干预选择。没有按 family、query 或结果选分支。
- **LOSO 与数量保持。** 独立核验全部 8 个所用 S/J 模型的 heldout/head/train_scenes；train_scenes 恰为其余三场。67 个 query 的候选列表及顺序、g_A 与原文件一致，A/S/J selections 逐列表原样保留。S0/J0 都按 `(-score, 原候选索引)` 取原 g_A，五臂均无重复候选且均位于原池。
- **原分数可复现。** 对实际 366 个候选向量及 67 个 query 独立重算，旧 S/J 分数最大绝对误差为 `4.440892098500626e-16`；保存的 S0/J0 分数与直接矩阵计算逐元素完全相同。仅四个 query 的选择集合改变，S0/J0 改变的是同四个 query；这只是预测差异，不是效果证据。
- **渲染等价与 GT 用途正确。** support 仍由既有 `_get_related_gaussians` 对选中 level/id 求二值并集，20 通道扩展与 `>.5` 阈值沿用原实现。新增 RAM cache 以 image 与排序后的选择集合为键；support 并集不依赖候选顺序或 prompt，因此该缓存不改变 mask。所有选择和渲染固定后才读取当前 GT，GT 只用于计算 IoU。A/S/J 每条均与原 pairwise CSV 比较，任一误差超过 1e-6 或非有限数即停止。
- **评测覆盖正确。** renderer 检查 67 个预测、各场 prompt 数 21/14/14/18，并依原完整 CSV 遍历 56/71/59/22 条 query-view；固定数量、候选池、旧选择及 score→selection 均在每条渲染前检查。full launcher 依次遍历四场；sanity 仅 figurines 首条，输出目录与 full 分离。
- **统计实现符合预注册。** 主指标为图内 query 均值→场景内图均值→四场等权。共享 `paired` 的独立 globals 确实使用 seed 20261102、B=2000，在每场内配对重抽 image cluster；不是对 208 条记录独立 bootstrap。唯一主比较为 J0−J，J0−A、S0−S、J0−S0 均保留。PASS 判据是主差值>0、CI 下界>0、各场下降不超过 .01，没有挑最好 head。缺少结果或数量不完整时不产生主结果。
- **新增写入范围符合约束。** launcher 中脚本输出、日志、临时目录和显式运行缓存均位于 `/root/rivermind-data/repro/deco-b123/metadata_ablation_20261102`，关闭 pyc，未包含安装或下载命令。renderer 以 `Scene(..., 0, load_sem=False, load_test=False)` 载入既有 checkpoint；归档 Scene 实现中写 input.ply/cameras.json 的分支只在 loaded_iter 为 None 时执行，本调用不会进入。远端实际导入与算子可用性仍由既定 sanity 验证。

## 独立只读验证

1. 实际原始数据验证通过：1097 candidate-view、366 features、67 scene/prompt、208 query-view；归档 score/label join 最大误差均为 0。
2. 上述所有模型/预测、两列干预、旧分数和五臂 fixed-count 复算通过。未调用任何 fit 函数，未重写预测或模型。
3. 四个本轮 Python 文件 AST 解析通过；launch.sh 为 LF，无 CR 字节。
4. 在内存中用归档 A/S/J 数据构造明确标记的统计 fixture（令 S0=S、J0=J，不是本轮结果）：主 A 为 `0.639262456425896`，零干预差值和 CI 为零，门槛 FAIL；常量差值 fixture 的差值和 CI 一致；缺输入返回 WAITING_FOR_INPUTS 且没有 overall 指标。确认 bootstrap 实际 globals 为 seed 20261102、B=2000。

审查仅写入本文件。宿主未暴露可独立核验的 native reviewer 实际模型和 effort，记录 `actual_model=null`、`actual_effort=null`；不声称 cross-family accepted。
