# 结果到主张：固定候选池的 LOSO 语义交互读出

**判定：claim_supported = no。** 本次固定配置的线性 q⊙f 交互头没有改善最终 mask 选择，也没有显示超出标量头与错配 query 控制的输出增益。此结论限定于本次 A 候选池、固定 g_A、冻结 CLIP/NAG 与四场景 LOSO；不等于证明 CLIP 无信息或 mask 选择方向无效。

| 固定比较（图→场景 macro） | 差值 | 95% 场景内图像 cluster 配对 CI | 判定 |
|---|---:|---:|---|
| J−A | −0.006886338 | [−0.031524390, +0.007317295] | 未达 +0.01，CI 下界不大于 0 |
| J−S | 0 | [0, 0] | 无语义增益 |
| J−Q | +0.00000000536 | [0, +0.00000001608] | 无实质增益，CI 下界为 0 |

A=.6392624564；S=J=.6323761181；Q=.6323761127。waldo_kitchen 的 J−A=−.0420325203，亦违反“无场景下降超过 .01”。因此冻结的三个主门槛及两个语义控制门槛全部失败。J−A 的 CI 跨 0，**不能称为已证明统计显著有害**。原最佳 gateE=.6760107 仍是另一配置的 anchor，本轮 A 不是 gateE。

原始 JSONL/CSV 核对覆盖 67 个 scene/query 决策、366 个 candidate groups、208 个 query-view，四折共 12 次优化成功；A 逐条误差为 0，所有分支保持 g_A。源码的 pair/scaler 只用训练场景，heldout 标签用于拟合后的评价；Q 仅错配新增交互块，标量块仍用真实 query。此为有监督读出实验，不是 training-free 方法。

支持的机制描述是：J 与 S 的 67/67 个选中 mask 集合一致，208/208 个 IoU 相同。J 的交互系数范数非零（约 .473–.516），J−S 分数绝对差均值 .04225，说明交互分支确实运行；分数变化没有跨过最终 top-g_A 决策边界。DECISION_TRACE 中 61/67 个 query 的 J−S 总分扰动跨度小于 S 的截断间隙，是固定模型的事后描述；J−S 同时包含重新拟合的标量系数变化，不能把该差值全部归因于语义块，更不能据此另设 gate 或调参。J 的加权 pair 准确率 .935657 高于 S 的 .932145 只是次端诊断，未转化为 top1 或多 mask 输出改善。

不支持的主张包括：语义交互改善 mask 选择、真 query 优于同容量错配控制、线性 head 优于现有最佳 gateE、跨新场景的一般有效性。q⊙f 加线性权重只测试了对角双线性交互，并未覆盖所有语义读出模型。四个场景已被多轮探索，图像 cluster CI 仅描述这些固定场景；单 mask oracle=.687273 也不是固定 g_A 多 mask 输出上界。

下一步停止本次固定线性头的改进主张，不通过扫描正则、重加权、事后挑 head 或新阈值挽救。当前负结论无需补跑。若继续已保留的 A∪B union probe，应沿用其事前约定，分开报告候选保留、single 与 gap 输出；它是尚未执行的另一问题，不能由本次失败推断必须增加候选。任何更强语义读出的尝试都须作为新假设重新预定，未来正结果仍需要独立场景和对 gateE 的直接配对。

证据直接读取：PREREG.md、train_ranker.py、train_summary.json、predictions.jsonl、results/*_pairwise.csv、SUMMARY.json、REPORT.md、DECISION_TRACE.json、full_driver.log；.aris/evidence_precheck_pairwise.json 的三项数值存在性检查均 verified。未新增训练、渲染、调参或实验完整性审计。

评议：fresh native reviewer；requested model=gpt-6-astra、effort=ultra；actual model/effort 未由工具暴露，记为 null。review_independence=same-family；acceptance_status=provisional；confidence=high（限于冻结改善主张不获支持）。未发现现成 EXPERIMENT_AUDIT.json，integrity_status=unavailable；该状态不改变本次基于原始结果的负判定。
