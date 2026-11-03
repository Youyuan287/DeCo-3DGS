# ARIS 执行前规划审查：重复视角共识偏置

你是实际 DeepSeek planner。只按下述已读代码和资产核查制定最小可执行方案，不声称运行或检索。不发明实验结果。

用户最新要求：Codex主代理只统筹监管和审查，ARIS工作流执行端完成其他工作。本轮先交付执行前门，不启动GPU、不改正式代码。研究对象是 LERF-OVS figurines（56 query-view），不是 LERF-mask。只考察固定最终NAG、2D特征、映射和evaluator时，重复视角投票/几何视角组等权聚合。不能重建NAG、SAM、重新编码特征、训练、扫参；GT仅最终评测。原缓存只读，存在符号链接，不可就地覆盖。数据盘16GiB可用，预留10GiB；系统9.3GiB预留5GiB。最多1GPU，但此门禁止GPU。

## 已核查代码
当前远端 ReLaGS_Reproduction/ReLaGS/merge_proj.py：
- _get_superpoint_mask_ratio 按 gau2sp 与 pt_sp_label 计数，去背景，按节点行归一化。
- proj_gaussian_features 内，低于0.3的mask比例归零；sp_feat=sp_mask_mat @ view_level_feature；逐视角节点向量先L2归一化，再乘可见高斯比例（render_point weight>0.0001）。
- _filter_feature_outliers：两两余弦；mean_sim=(sum(cos)-1)/(N-1)；mu/sigma按原始观测数计算；keep=mean_sim>mu-tau*sigma；全空回退argmax；对保留可见性加权向量求均值并归一化。
- extract_gaussian_features 每view/level用 torch.normal 随机20维标签码，经trace和0.85相似度过滤生成 pt_sp_label。
- 原程序只保存最终 sai_nag_3.0.pt 和 saifine_labels.npy，没有保存 pt_sp_label、逐节点逐视角特征、可见性权重或保留列表。

## 资产核查和历史反例
本轮只读定位了四个readout模型及评测目录。R0真实存在model_r0/sai_nag_3.0.pt与eval_r0/per_query.csv（历史56.7254%）；作者PT基线63.2871%另算。各readout最终NAG数不同，不能拼成固定图消融。
原2D特征和2D seg_map缓存仍在，但不是3D映射。ARIS旧spatial_support full目录只有 query_diagnostics、per_query、manifest等；未发现原始逐观测缓存。尚不能精确回放历史R0或作者PT聚合。不能说所有服务器不存在，只能说已查目录未找到、生产代码不保存。
历史空间乘法/数量控制相对作者PT都负，停止该路。先前有2D/3D特征不对齐、context crop黑背景等不干净对照，不用其宣布新机制成立。

## 待审最小方案
1. 当前可做CPU精确公式单元反例（合成输入仅证明算子对复制敏感，不能代表真实数据受损），并穷尽相关当前输出目录的缓存清单。
2. 真实实验需要原始逐观测状态。若缺失，必须停止“精确历史回放”说法。只可以另行提交受控新映射缓存采集请求；保持最终NAG和2D特征但重算映射仍属于新基线，不能自动执行。本轮不允许这样补缺。
3. 若找到同源缓存，先原聚合回放并比较保存特征误差、排名、Gaussian并集和最终指标，随后复制预定无GT几何组。只改变一组观测的重数、不改变特征值；逐节点检查保留变化、向量、选择。
4. 简单修正候选：组内可见性权重归一化，组间等权。但只改最后mean不足以保证复制不变，因为ROFA的相似度均值/mu/sigma仍由重数决定。需要明确到底测试融合权重还是完整共识统计；不能一次混改后笼统归因。
5. 最终收益必须来自实际56query-view，按图内query均值再图像均值，不用proxy替代。先figurines，不自动扩场景。

请输出：唯一建议的可执行门；精确需要的缓存；去重规则应如何定义才能避免伪不变性；不超过两种必要控制；最严重混杂；停止条件。如果缺缓存意味着不能做当前计划，请明确说明，不为推进而建议默认重建或伪造历史基线。
