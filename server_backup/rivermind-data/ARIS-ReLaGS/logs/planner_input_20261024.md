# 已授权的单场景配对实验：请给出实现裁决

用户已明确同意此前四组方案并要求“请你和ARIS进行实验”。你是规划顾问，当前Codex负责实现和执行。旧system里“等待通知/未授权”的状态已经过时；本次授权仅包含figurines单场景缓存复用的推理配对，不包含训练、SAM、特征提取、建图或扫描。

请结合随此prompt附带的研究审计，给出简洁、可实现、可证伪的方案。只推进一个假设：冻结已有NAG和候选，通过跨训练视角的查询空间支持改善范围选择。不是再次证明oracle高，不把2D掩膜直接当3D输出，不使用GT触发。不要扩大成完整研究流水线。

## 本次已核验资产

- 服务器FJ:30253，4090空闲，数据盘16GiB可用，最多1GPU作业，保留10GiB。
- 自建模型 `/root/rivermind-data/output/lerf_mask/figurines/sai_nag_3.0.pt`，几何 `point_cloud/iteration_0/point_cloud.ply`。
- 真实入口 `/root/rivermind-data/lerf_mask_repro/scripts/eval_lerf_mask.py`。
- 源码 `/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/nag_data.py` 中 SemanticNAG。
- 自建历史结果 figurines=0.7312689967392746，7个prompt×4视角=28queries，LERF-Mask/GaussianGrouping口径(按prompt先视角平均再类别平均)，不是56queries的LERF-OVS 63.29或四场景HF64.01。
- 参数topk5/level_until3/filter_small50/remove_small=True/render threshold .5。
- 训练视角299个，图像与缓存 `/root/rivermind-data/datasets/lerf_mask/build/figurines_train/{images,language_features}`，每帧 `_s.npy` 分层proposal ID map和 `_f.npy` CLIP特征，具体shape将由执行端检查。
- 完整评测相机 `/root/rivermind-data/datasets/lerf_mask/build/figurines_eval`，split.json记录训练frame与4个test图。只使用训练视角建立空间证据，测试GT仅评测。
- CLIP baseline `ClipSimMeasure()` = ViT-B16/laion2b_s34b_b88k，文本与canon object/things/stuff/texture对比softmax10，保持不变。

## 真实调用链与约束

SemanticNAG.search_matched_superpoint_in_mhtree:
1. `_get_root_candidates(sims,filter_small)` 根层相似度top10并大小过滤。
2. `_build_root_candidates_with_leaf_info` 后取根top5。
3. `_analyze_leaf_candidates` 将语义比分更高的child提升，产生overall_candidates。
4. `_filter_by_similarity_gap(overall_candidates,True)` 按相似度降序，用最大相邻gap决定保留前缀(第0个gap固定0，单候选可能全空，这是原行为，本轮不修复)。
5. `_get_related_gaussians` 组合选中节点，renderer输出semantics第1通道>.5。

为保持候选池严格相同，拟仅在第3与4之间重打分。根top5、parent/child提升逻辑在所有组保持原始sims。不能在method组给所有节点改sims导致搜索候选池隐性变化。每query记录pre-gap池与hash、选中节点、Gaussian并集hash。

## 预定四组

- baseline：原样分数与gap规则，重放并核验历史输出。
- visibility：相同池，只引入候选在固定训练视角的可见性证据。
- spatial：语义相关性乘候选投影与query最佳已有2D proposal的跨视角一致性。
- permutation：与spatial同算法，但固定seed0，在同帧、同SAM层及log2面积箱内打乱特征到mask对应；记录可置换率/单例，不能只有打乱标签名无实际变化。

只用按相机名排序均匀抽取8个训练视角的首轮受控pilot，不作全视角稳定性结论。缓存reuse不是特征重提取。允许固定NAG节点的低分辨率语义投影渲染，禁止重新训练/建图。

请明确：空间分数公式、无可见证据fallback、visibility控制与permutation的可识别性、同一候选池的边界，最危险的混淆因素，以及应当否定而不是继续调参的结果。不要预测会成功。不要用API审查意见代替实验结果。
