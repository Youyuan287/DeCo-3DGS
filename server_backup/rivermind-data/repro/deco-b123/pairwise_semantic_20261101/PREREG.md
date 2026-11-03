# LOSO query-conditioned pairwise semantic reader — 执行前冻结

2026-11-01。用户明确同意试验。只复用原A候选池（1097条view-candidate、366个scene/query/candidate、208评测、67scene/query）。冻结Gaussian、CLIP、实际NAG向量、query、原mask数量g_A与renderer。现有最好gateE .6760107不变；A原gap=.6392625。新增训练仅学习读出，非training-free方法。

## 分支与唯一主张

- A：原score与原选择，baseline逐条重现。
- S：线性RankNet，真实query的已有标量证据/几何。
- J：同一标量块，加已有文本向量q与实际候选f的逐维交互q⊙f；**唯一方法主分支**。
- Q：与J相同容量，仅新增交互块改用错配query；标量块仍用真实query。

不使用family、GT IoU、main_iou、跨query类名表作为输入。剔除随view变化的area_ratio，避免以GT标注view集合构造未定义的query-level部署输入。不使用agree/中心化一致性作为新评分。原有CLIP/NAG preprocessing逐位沿用；学习设计矩阵仅按训练数据缩放，不变换CLIP向量。

固定scalar列为：s,r,b,m,rank,deficit,is_leaf,level,root_score,log1p(n_points),log1p(root_n_points),n_points_ratio,log1p(n_leaf),leaf_gain,id_top3,id_mean,cover_frac,log1p(cover_views),mwp_conf,fallback_seg。query常量n_cands/gap12在线性pairwise中抵消，不作为新增特征；不纳入view面积。

## 标签、分折与训练配置

四折leave-one-scene-out。每折只训练另外三场；同scene/query的所有mask和view不跨折。每candidate先将GT单mask IoU按官方图→scene权重聚合为一个query-level质量：每view权重1/(scene图数×该图query数)，再在query内部归一。query总权重保持其在scene评价中的mass，各train scene等权。重复视图不扩张训练样本。

同query所有质量不相等(|ΔIoU|>1e-8)的候选对，较好在前。query内pair权重正比|ΔIoU|并归一，再乘query mass；全train pair权重归一为1。

无非tie pair的query不提供梯度；每个train scene内对有训练对的query mass再归一，然后三个train scene等权，最终总pair权重1。报告每fold有效query/pair与被排除的全tie query数。

损失weighted log(1+exp(-Δscore)) + 0.1/2×||w||²。线性头，zero init，L-BFGS-B，maxiter1000、tol1e-9。一次固定配置，无超参搜索。正则是本RankNet训练的固定weight decay，不是SREM的λ/μ惩罚。

scalar缺失仅以train fold median填充；每块按train pair-difference RMS逐列缩放，再除sqrt(有效维度)，使512维语义块不因维数自动放大。S/J/Q共用scalar scaler；J与Q的interaction scaler分别只由本fold训练交互构造。常量列置零。

错配q仅作用于interaction；train三scene内与heldout单scene内分别按seed20261101确定置换，禁止同unit及同归一化query文本，映射对所有candidate/view固定；不读取GT决定映射。报告不同文本却同embedding的碰撞。不得把heldout文本向量传入train映射或scaler。

## 评测与判据

每head按out-of-fold分数重排同一A池，统一取g_A个，原Gaussian精确渲染。先导出向量复核对前次r点积最大误差≤1e-6，做梯度/toy/fold隔离检查；再单query渲染sanity，A误差≤1e-6、数量完全相同；之后208条full。

主端为J-A的图→scene macro：delta≥+.01、paired image-cluster bootstrap CI下界>0、无scene下降>.01；语义机制还须J-S和J-Q的CI下界均>0。主门槛过但控制不过仅partial；主门槛不过不宣称方法改进。B2000 seed20261101。不得在S/J/Q中事后挑最好冒充主结果。

次端：同query一致的single-candidate oracle/top1质量/排序regret、pair accuracy、train与outoffold差、scene/family、win/loss、score/选择数。单mask诊断不是多mask输出上界。相同四场多轮探索，不称独立验证；成功仍需独立数据和对gateE的直接配对。

## 实现与资源

向量导出只读取现有NAG和CLIP文本编码，不重新编码图像或生成mask。拟合在本地CPU复用numpy/scipy；预测回服务器原renderer。所有服务器新写入在/root/rivermind-data/repro/deco-b123/pairwise_semantic_20261101，缓存/临时文件同根；不装依赖、不写系统盘、不覆盖原源码/模型。GPU导出/渲染预计小于1小时（估计），单GPU顺序执行。

输入与训练模型、fold/scaler/shuffle映射、优化收敛状态、outoffold predictions保存。负结果仅否定本次线性head，不推导信息缺失或mask选择方向无效。
