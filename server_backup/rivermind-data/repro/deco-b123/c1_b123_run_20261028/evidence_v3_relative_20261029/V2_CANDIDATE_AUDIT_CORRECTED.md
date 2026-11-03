# V2 真实候选与空间核验（修正版）

原脚本 `make_v2_attribution.py` 的 M2 行中 `candidate_set_same` 比较 `s.main_candidate_regions` 与 `s.new_candidate_regions`，对象是 **off/main vs semantic**，不是 semantic vs reliable。这是字段映射错误。原 CSV 的20条 False 包含顺序变化；忽略 score 并按 `(level,root_id,candidate_id)` 排序后，真实集合变化是 18 条。本次语义/可靠性分支重新渲染且与旧 CSV 逐查询 IoU 完全匹配。

语义 vs reliable 的真实候选集合变化 0 条，真实二值渲染mask变化 0 条。此结论来自直接对两分支结果比较，不再从错误字段推断。

off vs semantic：候选集合变化 18 条，Gaussian 二值集合变化 18 条，完整渲染 mask XOR 非零 16 条，IoU变化12条。Gaussian 通过同一冻结 NAG 的 `labels[level]==candidate_id` 并集直接比较；mask 通过逐像素 XOR 与字节 SHA256 比较；像素总数相同并不能证明 mask 相同。

196条 IoU 不变查询的精确核验计数：{'n': 196, 'candidate_changed': 6, 'gaussian_changed': 6, 'mask_changed': 4}。见 `V2_SPATIAL_CANDIDATE_AUDIT.csv`，每条保留独立布尔值/计数。旧报告中以像素数判断 mask 相同及以 IoU 相同断言候选相同的表述撤回。旧报告 M2 的179881行不是独立模式数，包含重复图像查询；`delta/correction` 原为区域量，被重复写在每个模式上，不能按行累加当成区域总量。

四条mask改变但IoU仍为0的查询均为ramen/corn：frame00024、00060、00065、00119，背景FP分别增加167693、106519、73624、171946，合计519782。因为交集保持0，IoU仍为0不能揭示这些新增背景响应；只分析12条IoU变化查询会遗漏这一风险。

## 三个机制

ramen 六个 nori：off候选包含 root92、root5、level2 root16/child40；V2提高 root5/92 的分数并改变相邻 gap，删除 child40（2881 Gaussians）。六视角背景 FP分别减少3498、20180、11242、2282、10854、4868；GT交集变化0、-119、-32、0、-24、-9。因此收益主要来自排除背景支持，而非新观察到目标。

ramen frame_00128/corn：V2新增 level3 root12（14024 Gaussians，score=0.544533），原有候选保持；GT交集未增加，背景FP增加57472。root分数越过root/child比较或gap边界可引入大量空间支持，不能由 correction 小就断言安全。

figurines tesla door handle：off的level2 root74/child132与root29/child38被root26/child194替代（21 Gaussians，score=0.638246）。frame00041恢复466 GT像素、增加1 FP；frame00105恢复3229 GT像素、减少5826 FP。该例是候选支持改变，不能单凭模式高分宣称语义互补。

上述每个图像查询的候选level/root/ID、分数、该root各模式advantage/权重/delta/correction，以及真实Gaussian坐标轴对齐包围盒、集合增删和mask XOR，完整保存在 `V2_KEY_MECHANISMS.csv`。包围盒是原始模型坐标，只描述空间范围，不将其当作语义标签。
