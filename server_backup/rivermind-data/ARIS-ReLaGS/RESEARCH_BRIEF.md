# ReLaGS 上的方法凝练与指标改进

最新覆盖指示（2026-10-24）：用户要求LERF-OVS实质进展；已批准figurines固定候选空间支持四组推理实验及发送方法/代码给指定API审查。只做缓存复用，不训练/提特征/建图/扫描。LERF-Mask73.13%不得计入OVS结果。当前具体方案见review-stage/spatial_support_20261024/OVS_EXPERIMENT_PLAN.md。下文停点只保留历史追溯，不能作为本次未授权的理由。

用户目标：从半年THGS/ReLaGS/VALA/B0-Alpha等实验中凝练一个真正可验证的创新点，在跑通的ReLaGS上提高最终分割指标。

当前前提：FJ服务器 `root@fj01-ssh.gpuhome.cc:30253`。本机`id_ed25519`可连接；`id_rsa`认证失败。2026-10-23只读资源检查为RTX4090 24GB、系统盘约9.3GB可用、数据盘约19GB可用。这不是将来运行前的资源保证。

**2026-10-24用户已通知HF-only复现完成，本轮CPU复核208条IoU与历史逐项相同，四场景宏平均64.0068369583%。** 结果目录ReLaGS_run_20261023。服务器有ReLaGS-1023与ReLaGS_Reproduction/ReLaGS两份源码，其evaluator不同，实验前须固定入口；原launch命令不在新结果目录，当前文件哈希不能倒推原运行身份。自建lerf_mask仅迁移、未跑；ScanNet不扩展。共享conda环境不改，绝对链接保持，补丁不重复应用。

保留的发现：

1. 正确3D候选存在但未选中在指定query上已有重复证据，不是新发现；二维proposal与NAG/受限组合oracle分开。
2. mask-conditioned观察有作用，但Alpha全量替换退化，不支持B0整体/Alpha部件分工。
3. 208对历史输出中THGS58.8550→ReLaGS64.0068的增益高度集中：THGS单方空贡献4.4685pp；双方非空贡献.9008pp，而kamaboko贡献2.1191pp。不是当前服务器新跑的结果。

方法凝练要求：说明新增的是什么信息/状态/目标；哪项旧实验已经否定简单解释；对哪个真实函数和数据接口做最小改动；为什么能改变最终范围而非只改善余弦/NDCG；哪项结果会使我们放弃。先保留“冻结候选、空间支持核验及对应置换控制”作为候选假设，允许证据驱动地否定它。

新颖性核对须覆盖直接相关THGS/ReLaGS/VALA/LaGa/EviSplat/DIGS/FLAIR/query-adaptive方法。保留多视角、多个向量、query注意力或层级路由本身不能直接声称创新。先基于原始证据凝练，不运行整套默认ARIS idea discovery/pilot。

历史ReLaGS HF-PT参考协议topk5/level_until3/filter_small50/remove_small=True、语义渲染阈值.5仅为追溯线索。当前基线跑通后以实际入口重新确认，不能从THGS的Top3默认值迁入，也不能为追历史数值改协议。

当前可以做：基于已核验HF结果制定方法决策、整理证据与配置。当前不做：自动调用模型开始完整研究循环、GPU实验、重建基线。用户本次询问下一步选择，推荐(d)，不视为已经批准实验启动。输出中文；后续每次只推进一条假设，保留原始失败结果。
