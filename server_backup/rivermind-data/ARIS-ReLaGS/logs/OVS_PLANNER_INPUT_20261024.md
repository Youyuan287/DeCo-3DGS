# LERF-OVS figurines 空间支持核验：当前唯一有效任务

用户已明确纠正：LERF-OVS和LERF-Mask为两个数据集，必须在LERF-OVS取得实质性进展。先前LERF-Mask73.13%只作旁支重放记录，不进入本轮方法表。旧prompt涉及LERF-Mask的路径和数字已经作废。用户已授权发送本项目方法、非敏感路径和评分代码至本API供规划/审查。

历史约束：二维oracle/正确3D节点存在但没选中已做过；局部语义contrast全场景退化，跨层zscore未兑现为最终指标，扩大topK及统一NMS退化；Alpha全替换不支持整体/部件双专家；FCE/BSHRINK/HCONS全栈不优于匹配pure。只做一个受控推理假设，不追旧数字，不重新证明oracle，不直接二维补洞。

本轮真实资产：
- LERF-OVS evaluator=/root/rivermind-data/ReLaGS_Reproduction/scripts/eval_lerf_relags.py
- code=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS，SemanticNAG.search_matched_superpoint_in_mhtree
- baseline model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines/sai_nag.pt（作者PT来源，明确标注，不冒充自建PT），四层labels，234534 Gaussians，feat表2103/463/177×512。
- source=ref/hf_only/figurines，GT=ref/hf_lerf/lerf_ovs/label/figurines，56query-view/4图；历史mIoU=63.2871%，四场景64.0068%仅为该HF轨道参考。
- OVS原训练图对应缓存=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features，299帧，帧名frame_00001等，features[N,512]float16，seg[4,728,986]int32，-1背景、ID全局引用feature表。存在feature ID无mask的情况，query选proposal必须过滤实际有mask的ID。
- 未确认缓存具体编码器权重，不能只因512维宣称与baseline完全同源；执行端检查编码日志/源码及相机图像对齐。若未完全溯源明确限制，不新提特征。

已获准四组：baseline/visibility/spatial/permutation。保持topk5、level_until3、filter_small50、remove_smallTrue、语义渲染>.5；根top5及parent-child提升仍用原始CLIP sims。只在_analyze_leaf_candidates返回后与_filter_by_similarity_gap之前重打分，严格相同pre-gap候选集合。保持原最大gap规则（单候选可能全空，本轮不修复）。

拟固定公式（不扫描）：8个按相机名均匀抽取非评测视角；每候选用已有renderer投影，先>.5再最近邻降至1/4宽高。V_cv=投影mask非空。每帧从有效proposal中以原ClipSimMeasure规则选query top1，M_qv为对应已有mask；不读GT。
visibility证据=mean_v V_cv，spatial证据=mean_v[V_cv*IoU(B_cv,M_qv)]。新分数=原score×证据。若整个query池的可见性全0，所有组回原分数；若可见但空间IoU全0，不能只因全0就回原分数，要明确是否保持零分触发原gap空输出。请裁决这一点。
permutation在同帧/同SAM层/log2面积箱中固定seed0打乱feature→mask对应，保留同一个原始top1 feature、映射到打乱mask。非单例用随机循环错排，单例保留，记录真正top1被移动和实际像素变化率。保留mask bank、分数bank、面积总体分布，不声称每query精确面积不变。

请用不超过1200字裁决：是否允许开展此窄pilot、需修正的最多3项问题、具体fallback/公式、可识别性和停止规则。不要扩大路线，不输出长泛泛综述。当前没有证明创新或增益。
