# Score readout：结果支持的主张

2026-11-01。`claim_supported: no`；冻结主检验 `FAIL`；`review_independence: same-family`，`acceptance_status: provisional`。

本判定针对“canonical 参考项损害同 query 排序，直接删除它足以提高实际分割”这一具体主张。结果没有支持该主张；不否定已有候选的语义选择方向，也不证明缺少观测或空间表示。

| 分支 | 图→场景 macro mIoU | 相对 A | 作用 |
|---|---:|---:|---|
| A：原 sigmoid 排序与 gap | 0.639262 | — | 原 baseline，非 gateE |
| B：margin 排序，固定 A 数量 | 0.639262 | 0.000000 | 单调性对照 |
| C：query 点积排序，固定 A 数量 | 0.615477 | −0.023786 | 唯一主比较 |
| D：margin 排序，自身 gap | 0.639262 | 0.000000 | 诊断 |
| E：query 点积排序，自身 gap | 0.594130 | −0.045133 | 诊断 |

C−A 的配对 image-cluster bootstrap 95% CI 为 **[−0.060288, +0.008362]**。正增益、区间下界为正、无场景下降超过 .01 三项门槛均失败。区间包含 0：应说“未获得正向改善证据”，不能说“已统计证明删除 canonical 有害”，也不能声称等效。E 的诊断区间为 [−0.064966, −0.025004]，其同时改变排序和输出长度，不能代替 C 的主结论。

四场 C−A 分别为 figurines +.031108、ramen +.032157、teatime 0、waldo_kitchen −.158408。query-view 直接均值为 +.001290，使用了不同权重，不能覆盖预先冻结的负向主指标。

## 支持的机制观察

- 208/208 条完成，A 与归档逐条误差 0；A/B、A/D 的输出集合及 IoU 均相同。B 符合单向量读出下 `s=σ(10m)` 的单调性。D 只说明本次样本中移除 sigmoid 未改变最大 gap；一般非线性变换仍可改变最大 gap。
- canonical 确实能双向改变同 query 候选顺序：695 次正负候选对出现中，伤害与帮助各 42 次；raw 与 margin 正确各 614 次。它们来自 170 个 query-view、56 个 scene/prompt；去重后 264 个有向候选对仍共享 query 和场景，均不能当独立统计样本。
- raw 令正 mask 排名改善 16 个 view、恶化 22 个、相同 132 个；两种分数平均正 mask 排名同为 1.547059。C 改变了 40 个 view、15 个 scene/prompt 的选择集合。不存在整体排序改善证据，也不能由 42:42 推出两种评分在质量或风险上等效。
- 个案具有双向性：ramen/frame_00006 的 sake cup 从 0 提至 .964087；waldo_kitchen/frame_00154 的 sink 从 .821319 降至 .014253。不能据此用 GT 或事后 family/query 分支选择部署策略。

## 不支持的延伸

GT crop 有正语义响应，不等于实际候选能由单一分数正确排序。当前 `r=f·q`、`b=max f·canon` 使用继承的特征预处理，不能一律称单位余弦。`s<.5` 仅表示 `r<b`，不是当前 NAG 的拒绝规则；renderer 对已选 Gaussian 支持的 `>.5` 是另一操作。

这是固定 **A 生成的候选池** 上的条件性重排实验，未改变上游 root 截断；没有检验全层级候选总体、早期 raw 检索、学习 reader 或不可恢复的信息缺失。208 个 view 复用 67 个 scene/prompt 决策，四场已多轮探索；image bootstrap 不代表新场景泛化。A=.639262 不能当作最好 gateE=.6760107；标准单向量、top3、gateE、single 与 multi 仍须分开报告。本次不产生与 THGS .816147 或官方指标的可直接比较结果。

## 下一项方法实验

只推进一个机制 probe：**按 scene 留出的同 query pairwise 语义匹配头**。保持当前 A 候选身份、现有 f/q、原预处理、Gaussian 与 renderer 不变；训练时比较同 query 候选的质量顺序，输入在已有标量之上加入 `q⊙f`，测试完整语义交互是否超出 scalar/rank/面积等旧标量的信息。训练偏好仅来自训练场景 GT；按 scene 四折留出，同一 scene/prompt 的所有视图与候选必须同折，重复视图先聚合为候选质量，不扩充独立样本数。推理无 GT。

同一次冻结实验包含标量 pairwise 头、加入正确 `q⊙f` 的头、同架构同容量但错配 q 的交互头。错配 q 仅作用于新增交互分支，原有标量保持真实 query；置换在各自数据划分内部固定。主输出统一保留 A 的预测数量 g_A，避免重新混入 gap 尺度效应；single 另列诊断。一次固定训练配置，不做温度、canonical 惩罚或阈值扫描，不重复 CFAR，不新增 encoder/mask。

这是把 [RankNet 的同 query 成对排序目标](https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf) 迁移到现有 mask 选择的待验证假设，不是新颖性结论。只有留出预测同时改善实际分割并优于标量头和同容量错配 q 控制，才支持“现有语义交互可用于可迁移读出”；否则只否定该具体头。当前四场 probe 成功也仍需独立未用场景和对 gateE 的直接配对验证。

证据：本目录 `PREREG.md`、`run_score.py`、`results/*_score.csv`、`results/*_details.jsonl`、`SUMMARY.json`、`REPORT.md`、`full_driver.log`，以及 `.aris/evidence_precheck_score.json`。预检查的 3 个数值存在性均 verified；这不升级语义主张。未启动新实验、未修改实现、未新增审计。
