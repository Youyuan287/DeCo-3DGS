# 将冻结J/J0候选分支接入既有gateE：叠加检验

2026-11-02，用户询问是否能叠加到.6760，按其授权直接检验。既有分析已知J0单独整体较低、9条优于gateE，且plastic ladle在gateE中已正确，因此不将单例恢复收益相加。该已探索四场的后续检验不称独立确认。

原gateE（E）完全按归档代码重建：原A池、GBM pred、area/mean_vis证据、id_top3缺失处理和moduleC/SM规则保持，使用归档真实gatee_20261031字段gatee_iou作渲染复现。原A与J/J0也保持。J0为前一轮预定两列贡献取消，禁止重训或继续改列。

仅追加一条确定的第三分支规则。对每条query-view，沿用原gateE的同一mm_arr（非有限id_top3以该query候选均值填补，全部缺失则0）。定义e(S)=max_{i∈S} mm_arr[i]，空集−1e9。EJ0：若e(J0)>e(E)且选择集合不同则采用J0，否则保持E；EJ同理使用未修复J作机制控制。严格大于，不设margin或其他阈值，平分保留E。不读取GT、family、旧IoU或oracle决定分支。

分支：A原baseline、E原gateE、J/J0原冻结模型、EJ（控制）、EJ0（唯一主分支）。重构E按原规则允许其选择数量与g_A不同；EJ/EJ0必须精确等于所选父分支的集合和数量，不强行改变原E的prune/include。J/J0固定原g_A。

预测前先冻结本协议。候选/评分来自既有缓存，不重编码CLIP、不新生成mask。重构时验证每query原g_A、moduleC/mask/E长度、e_c/e_m与gateE归档一致。先figurines首1 sanity核验A/E/J/J0对各自归档IoU≤1e-6，然后四场56/71/59/22，共208条精确Gaussian渲染。结果仅使用真实GT评价。

唯一主比较EJ0−E；EJ0−EJ为修复的额外作用，EJ−E为未修复分支控制，必须报告J0与历史E单独结果。主指标图内query均值→scene内image均值→四scene等权，paired imagecluster bootstrap B2000 seed20261102。主门槛均值差>0、CI下界>0、无scene下降>.01。只有EJ0−EJ也得到正向证据时才能将额外收益归因本次metadata修复。不得挑EJ/EJ0较好者或按GT/family挑分支。

报告路由次数、集合变化、逐case得失、scene与family。若无变化也完整验证baseline，不假造增益。GT择优仅可另标oracle，不能作为方法输出。本次不将叠加/gating本身称CVPR新颖性。

服务器新写入全部/root/rivermind-data/repro/deco-b123/gatee_stack_20261102，复用既有环境；缓存/tmp同根、不安装依赖、不写系统盘。预计<0.5GPU小时。主框架和旧实验只读。
