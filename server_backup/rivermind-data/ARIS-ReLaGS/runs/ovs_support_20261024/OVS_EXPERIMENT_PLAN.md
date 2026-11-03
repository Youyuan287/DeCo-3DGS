# LERF-OVS figurines：固定候选的空间支持核验

用户最新指示覆盖旧LERF-Mask计划；LERF-Mask结果不进入本轮结果表。用户2026-10-24已批准四组配对并明确授权将研究摘要、路径和代码发至指定API供DeepSeek/GLM顾问审查。

## 算法与固定项

- 执行入口：独立scripts/run_ovs_support.py，原仓库不修改。
- 当前可核验OVS模型：ReLaGS_Reproduction/ref/relags_hf/figurines/sai_nag.pt。来源为作者PT，不冒称自建PT。暂未发现可完整追溯的自建ReLaGS OVS PT。首先验证推理改动，不能据此主张改进了自建聚合算法。
- 56 query-view、4图，主指标图内query平均→图像平均。原参考63.2871401173%只作重放检查，不作调参目标。
- topk5/level_until3/filter_small50/remove_smallTrue/render threshold .5，CLIP B16/laion2b_s34b_b88k不变。
- 根top5及parent-child提升使用原始语义分数；在analyze_leaf后、最大gap前固定候选池。四组池哈希必须相同。原单候选空输出行为保留，不借此修复基线。
- OVS已有VALA/langsplat目录的299帧proposal/features512缓存；按帧名排序，排除4个评测帧后均匀取8帧。只渲染固定节点，不新训练、提特征、SAM或建图。
- 候选语义mask先以>.5阈值二值化，再最近邻降至1/4尺寸。query支持在同一低分辨率上计算。

## 四组唯一差异

令r为候选原相关性，V_cv为候选在支持视角v投影mask非空；M_qv为从该帧实际有mask的proposal中，以原CLIP规则选择的最高分mask，B_cv为候选投影。

baseline：r。

visibility：r × mean_v V_cv。

spatial：r × mean_v[V_cv IoU(B_cv,M_qv)]。

permutation：同spatial，但feature→mask对应在同帧/同层/log2面积箱内置换，查询最佳feature不变、只替换对应mask。

只有整个query池的可见性都为0时四组回原分数；可见但IoU全0保留零分，不能空间组特例回退。

## 置换及可识别性

seed0为基础，帧名SHA256的前32bit作为确定性子种子；同层/面积箱内随机循环错排，非单例均移动，单例不动。保留mask/score整体分布，不声称单个query面积完全相同。记录全bank单例率，以及真正query top1 mask ID/像素变化率。按DeepSeek建议：单例比例>25%或query级像素变化率中位数<50%，不据此得出空间对应机制结论；仍保留完整结果，不依据GT补选视角。

DeepSeek回复里“spatial不优于permutation才弱一致”逻辑颠倒，不采纳：必须真实对应优于置换才支持空间关联假设。该项交GLM复核。

## 执行与终止

1. 已完成CPU单测12项；独立GLM实现审查通过后，先首prompt sanity，再完整四组。无参数搜索。
2. baseline每query与历史OVS CSV匹配：IoU容差0.51e-6（原CSV六位舍入），pred_px逐条相同，任何不匹配则停止方法结论。
3. 保存输入/代码哈希、固定视角、命令、各query的pre-gap候选/各组分数与选中节点、Gaussian并集hash、预测mask、逐queryCSV、最终JSON。
4. 数据盘预留10GiB/系统盘5GiB，单GPU作业，禁止删除旧产物。输出仅ARIS-ReLaGS/runs/ovs_support_20261024/{sanity,full}。
5. spatial未胜baseline、不胜visibility或置换，不能宣称空间机制有效；如明显退化或新增空输出，不扩大场景、不盲调权重。单场景正向也不声称跨场景稳定贡献。

## 证据限制

缓存512维和历史提取器默认权重相同不等于已完整溯源其生成权重；缺失生成日志时本轮标注限制性pilot。8视角、单场景、单次确定性推理不等于全视角/三seed验证。暂不主张新颖性或投稿级贡献。

## 实施审查记录

DeepSeek正文已保存planner_OVS_20261024.json。GLM首次因reasoning耗尽token被截断，不视为通过；根据[官方API文档](https://docs.z.ai/api-reference/llm/chat-completion)添加可配置reasoning_effort=low后重试，正文见reviewer_OVS_20261024_low.json。顾问意见不等于用户授权，也不等于实验收益。

GLM要求的评测帧名匹配/剔除断言及历史CSV覆盖前置验证，已补充2项先失败后通过的测试；全目录15项通过。第三项比较方向错误已在上文纠正。8个固定支持帧在VALA缓存源和ReLaGS source的RGB逐字节一致，非评测视角；该检查不等于编码权重完整溯源。

sanity仅首个prompt“old camera”覆盖3图，运行成功且baseline与历史逐query匹配；不能把该小样本值写为figurines场景成绩。随后按原预注册四组56query运行，不依据sanity指标改参数。
