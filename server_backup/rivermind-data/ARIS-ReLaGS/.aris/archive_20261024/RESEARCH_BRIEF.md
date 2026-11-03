# ReLaGS 上的方法凝练与指标改进

用户目标：从半年THGS/ReLaGS/VALA/B0-Alpha等实验中凝练一个真正可验证的创新点，在跑通的ReLaGS上提高最终分割指标。

当前前提：FJ服务器 `root@fj01-ssh.gpuhome.cc:30253`。本机`id_ed25519`可连接；`id_rsa`认证失败。2026-10-23只读资源检查为RTX4090 24GB、系统盘约9.3GB可用、数据盘约19GB可用。这不是将来运行前的资源保证。

**等待用户通知当前ReLaGS跑通。** 服务器有`ReLaGS-1023`（只读检查HEAD为65dfba925abbd45cb68ddc8cc6959669e821f961）与`ReLaGS_Reproduction/ReLaGS`（52cc0c98b0ccf9afebea31901bf79246650286b3，已有兼容补丁和工作区修改）。不能因历史目录存在64.01结果就宣布用户当前仓库已跑通。前者是否用户当前工作目录仍须通知后确认。

保留的发现：

1. 正确3D候选存在但未选中在指定query上已有重复证据，不是新发现；二维proposal与NAG/受限组合oracle分开。
2. mask-conditioned观察有作用，但Alpha全量替换退化，不支持B0整体/Alpha部件分工。
3. 208对历史输出中THGS58.8550→ReLaGS64.0068的增益高度集中：THGS单方空贡献4.4685pp；双方非空贡献.9008pp，而kamaboko贡献2.1191pp。不是当前服务器新跑的结果。

方法凝练要求：说明新增的是什么信息/状态/目标；哪项旧实验已经否定简单解释；对哪个真实函数和数据接口做最小改动；为什么能改变最终范围而非只改善余弦/NDCG；哪项结果会使我们放弃。先保留“冻结候选、空间支持核验及对应置换控制”作为候选假设，允许证据驱动地否定它。

新颖性核对须覆盖直接相关THGS/ReLaGS/VALA/LaGa/EviSplat/DIGS/FLAIR/query-adaptive方法。保留多视角、多个向量、query注意力或层级路由本身不能直接声称创新。先基于原始证据凝练，不运行整套默认ARIS idea discovery/pilot。

历史ReLaGS HF-PT参考协议topk5/level_until3/filter_small50/remove_small=True、语义渲染阈值.5仅为追溯线索。当前基线跑通后以实际入口重新确认，不能从THGS的Top3默认值迁入，也不能为追历史数值改协议。

当前可以做：检查ARIS配置、测试API、整理已有证据。当前不做：自动修复ReLaGS、调用模型开始完整研究循环、GPU实验、重建基线。输出语言中文；后续每次只推进一条假设，保留原始失败结果。
