# 同候选评分分解，执行前冻结

2026-11-01。用户FINDINGS_MASK_COS.md表明GT crop保留正语义响应，提问canonical score是否损害选择。GT crop统计不直接证明实际候选排序受损。本实验保持当前released Gaussian/NAG、实际特征（含原有baseline预处理）、原A候选池、query和renderer不变。

对每个候选记录 r=f·q，b=max_k f·canon_k，m=r-b，s=原compute_similarity=σ(10m)。若f经过原有不renorm的预处理，r/b称点积，不能称单位余弦。

| Arm | 排序分 | 输出数量/规则 | 用途 |
|---|---|---|---|
| A | s | 原gap | 原baseline，逐条复现 |
| B | m | 固定A的数量g_A | 单调性阳性对照，应与A同集合 |
| C | r | 固定g_A | **主比较C-A：canonical参考项对同query排序的影响** |
| D | m | 各自原gap | 次比较D-A：sigmoid曲率对gap的影响 |
| E | r | 各自原gap | 次诊断，排序与集合长度同时变化 |

采用已完成early_root实验的A候选身份；不使用B新候选，不引入新特征、新mask、新语料、额外中心化/一致性或λ/μ。不能将D/E事后最佳作为主结果。单向量单调性结论不外推到逐token sigmoid后再平均。

四场208条。先figurines首条sanity，A误差≤1e-6且A/B同集合才full。主指标图内query均值→scene内图均值→四scene均值，paired image-cluster bootstrap B2000 seed20261101。C-A主比较需delta>0、CI下界>0且无scene下降>.01才支持原canonical项损害当前排序的端到端主张；否则只报告实际结果，不调温度或负词救实验。

GT仅评价，在所有分支候选确定后读取。额外记录同query中IoU≥.5与≤.1候选对，raw正而margin负的反转、反方向反转及最佳候选rank。重复视图不是独立query；报告scene-prompt单位。GT正mask跨不同词的均值差不作为此排序因果证据。

score<.5不代表当前NAG拒绝；现有>.5是已选Gaussian的渲染二值化。本次不做存在性/弃权/CFAR。已有0.6760107 gateE仍为独立anchor，A为.639262原gap。

所有服务器新文件、日志、缓存、临时输出位于/root/rivermind-data/repro/deco-b123/score_readout_20261101。复用环境，不安装依赖。估计小于1 GPU小时，无训练，无扫参；同一四场多轮探索，不宣称独立确认或CVPR新颖性。
