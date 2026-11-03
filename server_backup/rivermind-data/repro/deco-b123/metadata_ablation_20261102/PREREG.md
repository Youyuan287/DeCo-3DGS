# 冻结模型的观测可用性直接贡献消融

2026-11-02，用户授权继续；在生成新预测、渲染和查看新结果之前固定。该假设来自已探索四场中的plastic ladle失误，不是独立确认，也不是新方法结果。

假设：原LOSO RankNet把cover_views/fallback信号用于偏好观测较完整候选，可能压过真正语义匹配。仅取消这两项直接贡献能否改善真实mask选择，需全体查询而非单例验证。

冻结原pairwise_semantic_20261101的12个模型；本轮仅用既有S/J。原20列scalar的imputer、scale、active、block divisor及全部系数不变；候选池、67个scene/query、g_A、CLIP、Gaussian、renderer均不变。对每个OOF候选，将已标准化设计矩阵中log1p_cover_views、fallback_seg两列置零，得到S0/J0。语义交互块不变。不重训、不改正则、不用GT选择分支、不扫其他列组合、不按query/family分流。

五分支：A原main/gap；S原scalar；J原scalar+q⊙f；S0/J0为对应冻结模型的上述消融。所有分支统一选择g_A个候选，平分按原A池顺序。旧A/S/J精确渲染与原记录逐条复现误差≤1e-6。先figurines首1 sanity，再四场56/71/59/22，共208条。

唯一主比较J0−J，S0−S为机制控制；J0−A必须同时报告，J0−S0为语义增量诊断。不得挑S0/J0更好的作为主结果。主指标scene内先图内query平均，再图平均，再四scene等权；图像cluster配对bootstrap B2000 seed20261102，报告场景与family、胜负平、真实集合变化、主要失误是否修复及新损失。family仅诊断，不作输入。

诊断收益门槛：J0−J>0、CI下界>0、无场景下降超过.01。即使通过也只说明取消这两项直接贡献的探索性效果；若只修复个案却伤及别处，不判通用修复。若仅恢复到原A，不算超过当前最好gateE=.6760107。本轮A不称gateE，不与官方重建外指标混比。

仍保留cover_frac、mwp_conf、id_top3/id_mean插补等信息，不宣称彻底移除所有missingness通路。不能从阴性结果推导CLIP无信息或必须新增mask。取消特征贡献是固定函数干预，不等于缺列重训因果结论。

本地只生成预测、验证分数与汇总；所有服务器新增脚本、结果、缓存、临时目录位于/root/rivermind-data/repro/deco-b123/metadata_ablation_20261102。复用现有环境，不安装依赖、不写系统盘。预计顺序渲染小于0.5GPU小时；无新训练。ARIS fresh code review→sanity→full→result-to-claim。
