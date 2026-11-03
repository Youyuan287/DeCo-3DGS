# 开放词汇三维分割：证据锚定的问题裁决

用户本轮明确要求与ARIS协同，授权传递必要研究摘要用于问题发现/审查。只读研究，不启动任何GPU实验。领域简称“开放词汇三维分割”，3D-OVS仅是具体数据集。不要受旧system的figurines实验范围约束：旧实验已完成，本轮审查领域问题，验证将优先LERF-OVS。

你是无浏览/文件/执行工具的顾问；Codex负责全部实时检索与原文核验。因此只提供问题候选、竞争解释、需要查证的论文线索和检索词，不要说“我已查阅某文献/运行实验”。候选最多3个，不打分讨好、不造模块名；正文≤1800汉字，优先输出答案而非耗尽推理预算。

## 可追溯证据（摘要不是领域定律）

1. 十月177配置共享NAG/投票消融：pure四场景三seed=57.0868±.1916%，R56.1575±.6641，RB55.3462±.8803，RBFH55.5861±.7990。R同时改渲染贡献权重+trim20%；B开时从trim切到IRLS并加跨节点pooled残差方差收缩，缺IRLS-no-shrink对照。H按child Gaussian数加权prior改parent，alpha=.5(1-cos)。FCE历史是视觉端signed power，gamma.55在旧ramen PT正向，但同NAG新底座R→RF三seed都负。不能从名字或单场景概括有效。
2. B0/Alpha冻结Gaussian/NAG/proposal，但配套文本编码/预处理也改变；207样本正式宏平均B0-remap56.65→Alpha46.27，ramen/tea改善、fig/waldo下降。4个whole/part配对B0双正确3/4、Alpha2/4；10个错误8个unrelated、2个sibling；hubness null12组合无超p95。不支持整体/部件专家或平均余弦高=塌缩。
3. 正确3D节点存在未选中已在七月4query和后续冻结52query诊断中证明过；fixed3 oracle .8395、production .5458，可变数量oracle仅再多.0106。是协议内oracle不是任意形状上界。Top3→5→10全场景退化，统一NMS也退化；layer zscore只改善proxy约.127pp CI跨0；local-contrast正式开发集.5458→.1717。不能重复这些答案。
4. 六月parent/part面积先验及2D补空/SoM已做：scene-fill63.50、fine-opt64.13来源含2Dfallback；SoM67.48由GT IoU<.1筛触发，不是盲推理。继承/残差/软路由已有实现与汇总，缺同源有效对照，不能写没做过。
5. THGS作者PT58.855→ReLaGS作者PT64.0068，208配对mask主口径图内query均值→图像均值→四场景宏平均。净提升5.1519pp，空相关净贡献4.2511pp(82.52%)，双方非空175贡献.9008pp，其中kamaboko贡献2.1191pp。描述性拆账不是因果。ReLaGS多处变化，不能从PNG归因上游。
6. 最新OVS figurines固定pre-gap池四组56query：baseline63.2871%、visibility49.4952%、spatial52.7068%、permutation16.6072%。空间支持r×8视角meanIoU，对照同CLIP/NAG/相机/原gap；真实对应明显胜打乱却输baseline10.5803pp。spatial4改善/13退化/39不变，空mask与baseline均2但样本不同，双方非空贡献-12.0116pp，空转换+1.4313pp。top1实际像素置换中位100%、单例5.44%，两次小mask降采样消失。
7. 具体反例：red apple原正确节点仍是spatial第一，但score分布使原最大gap将保留数1→3，Gaussian1531→4603；pirate hat正确baseline节点(测试四图IoU .82-.95)在支持视角7/8可见却IoU支持全0，乘法改选错节点。前者已代码/分数证实，后者需投影可视化独立复核后才能排除坐标问题。最新raw scores诊断：空间组31query-view集合不变，10条“固定回原候选数可恢复原集合”贡献-3.9109pp，4条同数量改成员贡献-6.5953pp，11条混合改变贡献-.0741pp。不是固定数量方法的新mIoU。

## 当前代码接口

THGS get_related_gaussian按L2/L3 relevance全局Top3，Gaussian并集二值渲染>.5，CLIP分数无需>.5。
ReLaGS search_matched_superpoint_in_mhtree: 根top10/大小过滤→根top5→child高于parent时提升→按最大相邻score gap保留前缀→Gaussian并集→渲染>.5。root/child检索与候选范围已有query依赖；固定NAG节点选择不能重生边界。
当前64.01是作者PT轨道，不等于自生成聚合方法。已有2D masks/features、相机、labels，旧生产逐视角随机映射缺失。重提状态不能冒充原状态。

## 请完成

四入口（问题/方法/数据/既有假设）各最多2候选，可空，重复根因合并。最终最多3个问题，唯一推荐或拒绝推荐；每个给普遍性根据、原因竞争解释、真正最小干预、最近先例检索词及放弃条件。不要默认推荐新多向量/attention/区域重排，更不能把单场景gap工程改进直接包装领域贡献。对“语义识别分数被用来决定最终范围/停止”是否有更普遍证据，请最严厉反驳。指出何时应改候选构建而非匹配。
