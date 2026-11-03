# 开放词汇三维分割研究审计与路线裁决

审查日期：2026-10-23。范围：本地可访问代码、缓存、掩膜、结果表、研究文档与 Codex 任务记录；联网核验直接相关原文和作者仓库。本轮没有启动训练、编码器、SAM、建图、渲染或 GPU 扫描，没有恢复或覆盖历史源码。

## 结论先行

**这半年不是没有有效发现，而是尚未把“已有好候选，如何在未知 query 下选对其空间范围”做成一个闭合、可反驳、跨场景成立的方法。** 不能继续把 FCE、BSHRINK、HCONS 默认当作已被普遍证明的贡献，也不能把一次高 oracle、一次局部修复或一次 GT 触发的重选当作完整答案。

最值得保留的三个发现：

1. **候选选择确有可利用空间，而且不只在二维。** 六月的 fine-proposal oracle 属于二维候选；七月在同源 NAG 上检查的四个 query、20 个 query-view，正确三维节点 IoU 均可达 0.5 以上，却排在文本第 9–19 名。后来的 52-query 冻结三维候选诊断也显示较大检索差距。但刀等反例确有单节点覆盖不足；不能把“所有目标都已有答案”作为前提。[E06、E13、E14]
2. **区域观察方式有因果作用，却没有一个全面更强的替代编码器。** Alpha 的精确 mask 优于同模型 bbox/wrong-location 控制；但 155 个二维配对样本回退多于恢复，207 个三维样本的正式加权结果从 B0-remap 56.65% 降到 Alpha 46.27%。它在 ramen、teatime 有收益，在另外两场景大幅退化。应保留区域条件识别的事实，放下“B0 整体专家、Alpha 部件专家”的叙事。[E19–E23]
3. **总体 mIoU 很容易被少数灾难性失败的恢复主导。** 本轮重新读取 208 对掩膜：THGS→ReLaGS 的正式宏平均是 58.8550%→64.0068%，+5.1519 pp；THGS 单方空的恢复贡献 +4.4685 pp，双方非空合计 +0.9008 pp，其中 kamaboko 单项 +2.1191 pp，其余非空项合计 −1.2183 pp。这不是“无语义收益”，而是尚无广泛、均匀语义改进的证据。[E01、E02]

应修正或停止：

- 修正高平均余弦＝语义塌缩、跨编码器余弦近零＝空间正交、冻结高斯＝冻结候选图、相同 mIoU＝代码没生效、空 mask＝CLIP 分数低于 0.5 等解释。
- 停止把当前 FCE/BSHRINK/HCONS 全栈继续作为默认主线；保留版本明确的条件收益。停止全局扩大 Top-k、统一 NMS5、局部竞争惩罚、仅做层级分数校准，以及依据测试场景名选择最优模块的路线。
- 67.48% 不能作为正式盲推理成绩；62.19%、62.15%、58.85%、57.09% 不能拼成一个逐项可加的消融序列。

**唯一推荐主线：冻结三维候选，对候选的“查询空间支持”做显式核验，改变候选打分目标，而不是先换编码器、加维度或重建层级。** 这是有条件继续的机制假设，不是已验证的新贡献。第一版只利用已有、构建时与 query 无关的二维 proposal、特征和三维候选投影，在推理时比较覆盖完整性与过覆盖；不使用 GT 决定触发、不建立场景查表、不训练新头。它必须先胜过“仅排除不可见候选”的工程控制，尤其在双方非空、去除 kamaboko 后仍改善，否则不值得方法化。

下一项决定性工作是一个**冻结候选的空间对应破坏对照**：同一批候选、同一语义分数，比较原选择、仅可见性控制、真实二维—三维空间支持核验、保持语义/面积分布但打乱空间对应的核验。它不是再证明 oracle 高，也不是再扫权重；它直接检验新增空间对应信息是否提供超出语义和可见性的判别力。只设计，不在本轮执行。详见末节。

## 资料发现、权限与证据等级

### 实际找到的资产

| 资产 | 本轮核验内容 | 限制 |
|---|---|---|
| `C:/Users/TYY/Documents/ChatGPT/ZCY-4090/ablation_complete_20261022/` | 177 行结果、配对收益、候选变化、源码快照、原审计 JSON；本轮重新统计 177 个唯一配置和三 seed 完整性 | 模型张量留在远端；本轮未重新加载它们；原 AUDIT 的张量检查是既有记录 |
| `C:/Users/TYY/Documents/3D-OVS/full_report_work_20260730/` | 62.15/62.19 报告、修订审计、候选逐 query 排名与精确 oracle CSV、层级分析 | 62.19 生成时源码/NAG 来源链不完整；多数历史原始日志不在本地副本 |
| `C:/Users/TYY/Documents/开放词汇语义分割/` | 六月细粒度记录、fine adapter/trace、parent selector、scene fill、ReLaGS 旧评估脚本 | 单个脚本不等于当时实际运行版本；不能将该目录旧 ReLaGS evaluator 自动认作当前 64.01 的入口 |
| `C:/Users/TYY/Documents/3D-OVS/remote_thgs_patch/` | 六月层级继承、残差、软路由、parent-part、SoM 的可读实现和启动脚本 | 与正文里某些默认参数不同；缺逐次启动配置时不能反推最优值来源 |
| `C:/Users/TYY/Documents/deepseek-harness/default-workspace/` | Alpha 记录、MWP 四场景 CSV、从零 baseline 汇总、ReLaGS 代码审查说明 | `relags-review-src` 目录访问被拒绝，未绕过权限 |
| `C:/Users/TYY/Documents/ChatGPT/ZCY-4090/` 下 Alpha、granularity、margin、hubness、region、diagnostic 目录 | 2D/3D 编码器配对、失败分类、原始 query CSV、后续反例、质量打分代码和状态 | 部分后续 52-query 全表只在远端，详见下文分级 |
| `C:/Users/TYY/lerf_vis/` 与 `E:/CVPR 2026/lerf_vis/` | 找到要求的 XLSX 的配套原始 CSV；读取两方法 208 个预测及各自 208 个 GT PNG 重算 | 没有用 Excel 汇总数字替代掩膜统计；只由二值掩膜不能确定空输出的上游原因 |
| 可访问 Codex 任务 | 核对后续校准、局部竞争、质量打分停在何处；只读，无跨任务发消息 | 最终回复属于二级证据，不替代其所引的原始结果文件 |

本地工作区 `master` 没有可解析的 HEAD 提交；THGS-main、VALA 副本也没有获得独立可用历史，不能编造本地 commit。历史报告引用 README baseline 的 commit `c6423453fc7aa74772ca8883961ed696121b039c`；ReLaGS 静态审查引用 `52cc0c98b0ccf9afebea31901bf79246650286b3`，后者不是当前 64.01 运行版本的核验证明。可读源码的内容 SHA256 存于 `research_audit_assets/source_manifest.csv`。

访问限制与缺项：

- HN-JMP `root@hn01-ssh.gpuhome.cc:30572`（指定 IP 153.0.191.130）两次只读 SSH 都返回 `banner exchange ... Connection refused`；排除本地配置后仍失败。仅说明本执行环境本轮未连通，不代表用户不能连接或数据不存在。
- 未访问六月 `/home/Groups/group2/Working/tyy/project/THGS-main` 和七月 `/root/autodl-tmp/...` 原主机；本轮没有有效连接凭据/当前端点证据。不把其资料写成已丢失。
- `EXPERIMENT_LOG.md` 在已检索的项目、Documents、Claude 资料范围未定位到可读原件；交接文档引用的行号只能作为索引，不能充当原日志。扫描不是对整台机器所有卷的穷尽证明。
- 校准、score-quality 的远端原始 `csv.gz`、旧生产模型的逐视角随机投影状态、当前 ReLaGS 的运行源码/config/候选 ID/连续渲染图，尚未在可访问副本完整找到。
- `ablation_complete_20261022/historical/` 多个 CSV 实际只有 1 字节，不能当作已有完整扫描；用 `historical_all.csv`、已有 eval inventory 分别索引，不将空占位文件算成验证。

本文使用四种证据标记：**A** 原始产物本轮重新统计；**B** 可读代码与既有结果相互支持但未重跑/部分 lineage 缺失；**C** 汇总或任务最终回复记录、缺原始产物；**I** 仅实现/建议，无方法效果。CSV 每行分别标注“提出、实现、执行、有效对照”，不把它们合并。

## 先拆开 baseline 和数字口径

主口径为：图内 query 平均 → 场景内图像平均 → 四场景等权。记样本权重

\[
w_{svq}=\frac{1}{4\,|V_s|\,|Q_{sv}|},\qquad M=\sum_{svq}w_{svq}\operatorname{IoU}_{svq}.
\]

不是 208 行直接平均，也不是先对每种 query 跨视角平均。像素准确率、IoU 达标比例 Acc@0.25 不能因为都写 mAcc 而混用。

| 数值/标识 | 真正的用途与身份 | 本轮可采用程度 |
|---|---|---|
| **54.62%** | 七月 README 自生成全流程：50.94/32.44/81.21/53.88，历史 commit 如上；报告给出 `configs/lerf_full_repro.yml`、`output/render/lerf_full_repro_20260714` | 修订审计已定位其来源；旧文件“未找到”已过时。但本轮没有原始预测/运行日志，定为 B/C，不冒充重新复现 |
| **58.8550%** | 当前可读 `lerf_repro` 掩膜/历史已保存官方 PT 路线：54.9343/42.0733/81.8710/56.5412 | A：208 对掩膜重算；是导入产物参考，不是本轮重新跑 merge 得到的 baseline |
| **58.8467%** | 七月 common-stack **无 FCE**：59.5171/36.0867/80.8295/58.9536；含 Robust、B、H、AQ 等 | B：与上行几乎同数却是完全不同配方！不能称纯 baseline |
| **57.0868±0.1916%** | 十月统一 NAG/投票 wrapper 的 pure，自生成四场景×三 seed，AQ 关闭 | A（表重算）/B（模型 lineage）：不是 58.85 PT 的精确恢复，也不能证明等同于更早挑出的最佳 57% PT |
| **62.1538%** | 七月 common stack + ramen-only signed-power FCE **gamma=0.7** 的可恢复运行 | B：相对对应 no-FCE 为 +3.3071 pp；不能替换为十月 gamma=0.55 的结论 |
| **62.1882%** | 历史 `*_best` 掩膜结果 | 结果指标有记录，但生成时源码/NAG被覆盖，逐字节来源不完整；不当作当前完整系统复现 |
| **63.50 / 64.13%** | 六月 scene-fill / 再加 parent-conditioned fine selector；含二维 mask fallback | C + 实现可读：不是冻结三维候选的新表示收益；需补原启动配置和逐样本产物才能升格 |
| **67.48% / oracle 73.62%** | 六月 GT 低 IoU 触发的 SoM 替换 / 其候选单选 oracle | C（成绩），B（GT 触发已读代码确认）；不是正式盲推理，也不是全 NAG 任意组合 oracle |
| **64.0068%** | 当前 ReLaGS 输出按同 GT、同正式权重重算：63.2871/51.1494/80.9687/60.6221 | A：确认输出可比；不等于剪枝/ROFA/关系各自贡献已被隔离，也不是论文所有任务复现 |

十月 177 行不是 177 次独立重复：14 个主配置×4场景×3seed=168，加 teatime 固定 alpha 的9行；其中27个特征相同配置复用评测。源 merger SHA256=`26506bef77c34eaa9c07c6d379af0ad594e82609d9dc38e8aa3ee4172df6d03b`。本轮重算 177 个唯一键、seed0/1/2、配对表全部 `nag_identical=True`；没有再次读取远端177个模型。

旧“54.62 尚未定位”报告与后来的 `corrected_cross_project_audit.md` 必须有先后区分；同理 `STAGE_6_STATUS.json` 写未进入后续，不可据此否认后来任务记录已经执行的校准和局部竞争实验。

## 按机制归并的研究路线

### 多视角聚合、方差收缩与层级修正

**原始问题。** 投影错配、异常视角和稀疏观测让区域特征不可靠；希望剔除异常、稳定方差、借助子层纠正父层。可信基线是十月共享 NAG/逐视角投票的 pure，不是旧官方 PT。[E03–E05]

实际调用链见封存 `source/merge_proj_bshrink_control_seed0.py`：

- `SAI3D.proj_gaussian_features_x`（504行起）先对 superpoint—proposal 计数归一化，去背景列，`mask_ratio_tau=0.3` 截断，乘二维特征，逐视角归一化；用可见 Gaussian 比例累积。这里 `gt_feature` 变量是二维教师特征，不是测试 GT。
- R 不只是均值改 trim：收集的权重还包括平均渲染贡献×可见比例；`robust_weighted_mean`（47行附近）按到初始中心的余弦裁掉20%视角。故 pure→R 包含权重和鲁棒裁剪的共同改变。
- B=1 切换到 `irls_consensus`（69行），不是 trim 后再加一个独立 B 模块。残差 \(r_v=1-\langle x_v,\mu\rangle\)，\(\tilde\sigma_c^2=(n_c\sigma_c^2+n_0\sigma_{pool}^2)/(n_c+n_0)\)，\(\tau_v=\exp[-r_v/\sqrt{\max(\tilde\sigma_c^2,10^{-12})}]\)，用 \(w_v\tau_v\) 重估共识，三轮。全局 prior 来自跨节点 pooled residual，`n0` 为观测数中位数，另测×2.25。**缺 IRLS-no-shrink 这项干净对照，不能将 R→RB 全归因于方差收缩。** 也不是在几何上缩 mask。
- H 是 `apply_hierarchy_consistency`（679行）：child 特征按 Gaussian 数加权成 parent prior，\(\alpha=0.5(1-\cos(f_p,\bar f_{child}))\)，归一化混合后向上继续。它改 parent，不是六月的 child 继承 parent。高余弦0.965时 alpha约0.0175，效果弱有明确数学可能，但不能仅由这个例子证明全场景原因。

最干净的已执行控制与反例（百分数，三 seed 样本标准差）：

| 配方 | 四场景宏平均 | 解释 |
|---|---:|---|
| pure | 57.0868±0.1916 | 本轮最高均值的统一配置 |
| R | 56.1575±0.6641 | figurines/ramen受益，teatime/waldo退化；不是稳健底座 |
| RB | 55.3462±0.8803 | 相对R：teatime +2.2612 pp，figurines −6.9998 pp |
| RBH | 55.9004±0.5770 | H修复部分B退化，不代表超过R或pure |
| RBFH | 55.5861±0.7990 | 全栈未超过pure |
| RBbestH | 56.5560±0.6179 | 本轮增强配置最好均值，仍低于pure |

H 在 R 后对前三场景指标无收益；在 RB 后对 figurines +3.1109 pp、waldo +1.3672 pp，却在 teatime −2.2612 pp。teatime 的固定 alpha .25/.50/.75 也没有恢复对应 no-H 的结果。**同一个模块的“修复某底座”和“新增净能力”必须分开。** query_changes.csv 记录有序列表变化而并集不变的情形；前向执行、特征变化、候选变化、最终掩膜改善是四件事。

历史七月 Full-minus-B 保护 teatime +6.49 pp、Full-minus-H 小幅正贡献等仍保留，但它们与十月的底座、实现和AQ不同，且不是统一随机重复；不能以旧表否定新反例，也不能用新反例宣布所有历史成功都是假的。

**版本核对尤其重要。** 本地 `THGS-main/thgsv3/torch_ops.py` 的 `robust_fuse_torch` 是另一版每节点 IRLS，所谓 global 同源范围不等于跨节点 pooled；`historical.py:fuse_pairs` 已改为跨节点统计，但每轮重新估计 global prior、零方差保护与封存旧版也不同。`enhance_features_torch` 是中心残差增强，不是 signed power。要看 `method_profile` 的实际分派，而不能仅凭文件名 canonical 宣称等价复现。历史问题已有修正版，不能跨版本一概否定。[E05]

中断/转向依据：同源投票消融已暴露跨场景符号反转；逐视角生产回放又因原随机状态未落盘而无法完成。**裁决：保留作可靠性工程和可比较控制，停止将全栈作为当前核心贡献；不要再围绕最佳 seed 调参求回到某个数。** mask-purity residual blend曾在固定NAG表上55.76→56.91，值得存档，但缺完整多seed/保留组，不升级为已证明的通用方法。[E26]

### 特征幂变换与查询阈值

**想验证的机制。** 通过特征重映射改善检索对比，或用分布适应输出阈值。[E04、E07、E08]

FCE实际为 \(f'=\operatorname{norm}(\operatorname{sign}(f)|f|^\gamma)\)，只改视觉端坐标、文本端不变。幂变换会放大小坐标相对权重；它不保证任意两特征余弦下降，更不保证正确 query 的 margin提高。因此应当检验“正确—错误候选的排序是否改善”，而不是仅报告平均余弦。

历史作者 ramen PT：42.0733→49.2547，gamma=.55有正向记录（与.5同分）；七月common-stack采用gamma=.7，36.0867→49.3150。但十月同NAG R→RF在ramen为51.4826→50.0987，三seed全负，−1.3840±0.0413 pp。这说明**坐标重加权是条件有效的 trick，不能证明本征的“距离拉开”普适原理**。不是否认其曾成功，而是有效性边界已经明确。

封存 merger 807行的 FCE确实由场景名包含ramen触发。它不应被写成全场景通用模块。历史 `SCENE_ADAPTIVE` 某版只更新未传给 `self.args.thres_merge` 的局部变量，其正负消融差值不能归因于合并阈值；本地新 `merge_proj.py:664–667` 的 `scene_policy=effective` 则真的写回参数，不能继续说所有版本都no-op。

AQ至少存在三种口径：固定0.5、历史备份/重建256-bin Otsu、本地core的bounded percentile/median阈值。后者 `adaptive_query_threshold` 是基于90–10百分位跨度和中位数的小幅修正，不是Otsu。“开AQ”不是充分配置描述。移除Robust还可能让其依赖模块失效，Full-minus-R不等于孤立Robust作用。

最有力反例是同输入FCE在新底座反向，而不是“新机器环境不好”。中断原因：迁移和改写使同名模块语义变化，继而耗费大量时间追旧数值。**裁决：FCE按配方存档；停止扩大gamma扫描、全场景强用FCE及将阈值收益当语义机制。**

### 候选构建、层级检索与节点组合

可信参照是 `output/lerf_ready_official` 的冻结NAG；对照必须固定 `nag` 的全部label张量、候选ID和renderer。[E13–E18]

- 七月四query：sake cup单节点oracle均值.8616、plate .6920、onion segments .7992、bear nose .9749；实际几乎为零，正确节点全在第9–19名。可访问CSV在 `C:/Users/TYY/Documents/3D-OVS/full_report_work_20260730/reports/minimal_falsification_11_candidate_ranking_20260728/query_view_exact_oracle.csv` 和同目录 `query_summary.csv`；报告内所写原远端 `data/` 在此副本已被展开。这已经回答“正确三维候选是否存在”，不能重新包装成发现。
- 九月同模型Top-k对照：开发集四场景平均Top3 57.21，Top5 52.57，Top10 49.04，Top20 50.74。waldo改善，ramen大幅退化；ramen预测面积从约1.02倍GT扩到3.80倍。全局扩候选不是解法。
- Gaussian集合IoU去重选5：waldo开发集38.89→61.55，但全场景53.51低于57.21。事后给waldo用NMS5拼出的62.88只是开发集查表，不是冻结模型全量结果。knife oracle低/另视角排名331，反驳Top20解决全部错误。
- 后来52-query冻结诊断：production Top3 exact IoU .545759，Fixed3 oracle .839538；可变数量比Fixed3只多约.010647。**该oracle是协议内搜索到的参考值，不是已证明的所有候选组合全局最优。** 目前优先级应是选对，而非先动态决定数量。[B/C：本地综合报告加远端任务记录]
- 后续local-contrast A1实际执行并失败：exact Top3 .545759→.171746，四场景下降；不能再建议直接惩罚局部语义密度。后续审计认为local competition不是可靠负证据，不能把失败唯一归因于L2 bias。[C]
- 五种跨层校准也已执行：C0 RetrievedBestIoU@3 .577365，zscore .578639，+0.127 pp，CI跨0；robust-z .547097。**RetrievedBestIoU@3是被检索到的最佳单候选质量，不是Top3联合渲染IoU。** 分数顺序变化不是最终分割提升。[C]
- score-quality补充：NDCG_FULL .7056、Spearman .0926；robust-z的QAPNear有改善而full-list NDCG只+0.17 pp、CI跨0。原任务把top-decile lift .0136又写作top-decile mean，源码把两者分别计算，应以原表列名为准；本轮不复用这个歧义数字作结论。[C]

**中断原因不是没想到重排。** 检索方法已经具体试过，local-contrast被结果否定，校准不足，原聚合回放缺缓存；其后转成quality-aware评分。`quality_aware_scoring.py`已实现大小+child agreement乘semantic score，CPU sanity通过，状态明确`official_miou:null`；因当时CUDA不可见未正式执行。不能称它有效，也不能称它已经失败。该query-independent质量偏好不足以判断同一实体的整体/部件范围，不建议直接扩成主方法。[E18]

本轮还从Top-k和NMS的原始逐query CSV重算了每arm的156行，确认上述57.2121%、50.7410%、53.5101%等开发集结果，见 `research_audit_assets/candidate_dev_recheck.csv`。另有H1文档分别报告156个query-view和52个query聚合的oracle，数值不同；原表未完整到手时不将其自动认作同一均值，也不将其写成新增方法收益。

**裁决：继续候选选择这一问题，但停止旧local-contrast、纯层校准、全局NMS和扩大k的具体答案。保留旧实现作反例和固定控制。**

### 二维 fine proposal、parent-part、继承/残差与 SoM

六月已经有完整的技术分叉，而不是只有口头建议。[E09–E12]

- `scene_saturation_probe.py`诊断的细粒度proposal oracle约.855–.919，属于各帧2D proposal，不代表NAG能表达同一区域。
- `hybrid_fill_empty.py`对空/近空mask替换2D proposal，记录58.85→63.50。可读实现与工程动机一致，但“恢复输出”不能归因于更好的三维语义表示。
- `fine_scene_adapter_render.py` / `fine_trace_part_render.py`将proposal分数投到Gaussian连续场；teatime约.225–.329，高召回但区域很散。中心投影改trace改善对应处理，却没有自动消除错误标签。正确鼻子proposal被标成egg等反例说明标签筛选不充分。
- `parent_part_proposal_render.py:choose_part_mask`以parent包含率、面积比例log惩罚和弱文本加分选2D候选；记录teatime81.87→84.36，partish5.98→38.83，fine-opt整体64.13；bear nose .0642→.9840，hooves仍为0。局部有效，但没有证明父锚点和面积先验可泛化。**可读版本已有nose=.13、hoof=.025、小part=.04的专用cover，而六月文档曾写最佳.02**：版本/参数不齐，不能认作同一次运行。
- 六月 `nag_data.py:get_hierarchy_features` 已实现 child加gated-parent与child-parent residual；`get_soft_gaussian_scores` 已实现phrase/term、child-parent正差分、跨层softmax加权Gaussian场。前向代码存在且汇总说做过；未找到足以隔离每项贡献的同源完整结果。不能说它们“从未做过”，也不能将parent selector成功归因于特征继承。[E11]

**SoM关键核验。** `remote_thgs_patch/llm_som_object_reroute.py:772–774`先算baseline与GT的IoU，默认仅保留不超过0.10者；`scripts/run_lerf_gpt54_som.sh`传`--all_prompts --low_iou_threshold 0.10`。`all_prompts`只取消词表过滤，**不取消GT低分过滤**；真正绕过它的是`--include_all_targets`，该启动器没有用。`apply`先复制baseline，只替换tasks里的样本。候选图与提示不显示GT，不能消除“哪些样本被替换”已经由GT决定的问题。汇总54个低分样本、总体67.48、单候选oracle73.62，应记为GT-assisted selective intervention。[E12]

脚本有全量选项，但选项存在不等于有全量无GT触发运行。此次可访问资料**未找到后续全量、无GT触发的tasks/selection/raw-response/预测配套结果**；不写成“后来绝对没有做”。此外GPT可选择多个mask并集，而summary oracle选最佳单mask，oracle未必是该GPT策略整个合法输出空间的严格上界。

中断依据：文档明确从parent-part转向object级SoM，因为局部改善未解决大量object失败；SoM又依赖闭源选择和GT触发，轻量重排/三维bank的后续方案未形成完整证明。**裁决：保留空间约束与显式候选比较的经验；停止把2D补洞、GT选择性干预及若干成功图当作完整3D方法。**

### 编码器、上下文与 B0/Alpha 互补

这是最容易被后续叙事忽视反例的一条路线。[E19–E25]

可信二维控制：同SAM bank、同RGB、同proposalID，Alpha同权重仅改alpha条件；Exact Mask Top1 5/13，bbox1/13，wrong-location0/13。Full-image输入对各候选完全相同而打成平局，不能因为并列rank=1算成功。这个控制支持mask conditioning有作用，但不隔离Alpha与B0之间所有权重/预处理差异。

更大155样本：B0二维Top1 31.61%、Alpha21.94%；Alpha恢复14个B0失败，却破坏29个B0正确。B0-or-Alpha oracle Top1 40.65%，不是无GT的可部署router。13个困难样本改善不能代表总体替代成立。

三维实际代码 `alpha_3d_run.py:mapping`读取原始`sai_nag.pt`并assert label逐项相等；原Gaussian、相机、SAM `_s.npy`、proposal IDs和数量冻结。原随机Gaussian→proposal关联没有保存，于是B0-remap与Alpha共享新种子`100*sorted_view_index+level_index`重建映射；它们是合法配对，B0-original不是该变量的干净对照。此独立脚本没有重新region-growing，**本次换特征没有改变NAG候选集合**；不能外推为其他全流程换特征也不改图。

文本侧也变了：B0用OpenCLIP ViT-B/16 `laion2b_s34b_b88k`，Alpha用`clip_b16_grit1m_fultune_8xe.pth`配套文本编码器；同canonical negatives不等于同文本编码器。B0 masked bbox crop/pad与Alpha fullRGB+mask+center-crop不同。因此“只改变视觉特征”的描述需修正为**替换成配套的区域视觉—文本管线**。

| 正式图内/图像/场景加权，207样本 | figurines | ramen | teatime | waldo | 宏平均 |
|---|---:|---:|---:|---:|---:|
| B0-remap | 56.46 | 47.68 | 78.67 | 43.78 | 56.65 |
| Alpha | 18.55 | 61.40 | 82.54 | 22.60 | 46.27 |

不要改用B0记录文档里的207样本直接平均57.71→51.96，来弱化正式宏平均退化；两者是不同口径。waldo少1个GT导致B0-original宏平均58.69，不是208样本58.85。

反例具有裁决力：

- `granularity_audit` 的4个整体—部件配对，B0双向正确3/4，Alpha2/4；熊的3视角B0均分清whole/nose。B0在全bank失败，不能推成输给这张whole-bear候选。Alpha的粒度偏向随视角变化，不是稳定part专家。
- 某view Alpha预处理仅保留鼻子约35%、整体约56%；失败混合了裁剪损失，不能纯解释成语义能力。
- Alpha在ramen特征平均余弦很高却有最大增益，直接削弱“平均余弦高必然检索差”的解释。跨编码器坐标轴没有对齐，点积近零不能证明语义空间正交。
- `hubness_stage2` 的12个scene-level组合没有一个超随机null的p95，hubness主因被削弱；`hier_margin_minimal`的错误10条中8条unrelated、2条sibling，不支持所有错排都由parent压制造成。
- SCLIP/ClearCLIP和text-region条件对照在困难子集没有稳定突破；OpenSeg、RegionCLIP、MaskCLIP有blocked记录，不应写为模型效果失败。
- 更早shape-preserved context二维LOSO可到top1_mIoU .5567，但直接3D context融合 .5509/.5512，未超过其crop baseline .5618。二维平均margin改善不是三维分割收益。[E25]

中断/转向是大范围替换退化、局部机制解释被配对实验削弱，随后回到query打分。**裁决：停止整体/局部双专家和盲目换backbone；Alpha保留为区域观察方式对照和可能的局部证据源，不按场景名挑编码器。**

### 几何、可见性、ReLaGS 与 VALA

ReLaGS当前64.01的输出有效，但从THGS到ReLaGS同时改变几何剪枝、层级构建、聚合及选择/输出，不能只凭总分证明ROFA或关系推理贡献。[E01、E27–E29]

本轮另找到MWP四场景表：从零baseline54.9464，统一prune=.0005为56.0664、统一.1为56.5814；事后各场景取最优为59.7448。后者不是可部署“adaptive”策略，除非另有不看GT的规则及验证。剪枝还改变几何与候选空间，不能和冻结NAG特征消融混用。

VALA七月原始208行在其原“场景内样本均值”口径下53.8927%，本轮按正式图像加权重算为54.5587%（`research_audit_assets/vala_metric_reweight.csv`）；不能将两者差额当作方法改进。原报告0.60→0.65阈值只约+0.03 pp；不支持一切差距都能靠阈值恢复。它的2D中间特征评估不是VALA三维表示能力；Acc@.25不是pixel accuracy。

THGS实际文本得分为canonical-negative relevance：

\[
r_c(q)=\min_{n\in\{object,things,stuff,texture\}}
\sigma\big(10[\langle f_c,t_q\rangle-\langle f_c,t_n\rangle]\big).
\]

但THGS `SemanticNAG.get_related_gaussian`按分数选L2/L3最终Top3，直接把所选Gaussian设为1；**不要求其文本得分>0.5**。随后renderer alpha-composite，再对语义覆盖图阈值>0.5。故一张空mask可能来自选到不可见节点、贡献太小或最后二值化，不能直接归因为relevance不足。相同颜色/几何渲染背景也不能证明语言选中节点可见。

当前只有ReLaGS每query `n_candidates`汇总，未获得对应节点ID、贡献图、过滤原因等运行配套，不能把新ReLaGS所有空mask归到同一条代码。六月旧`eval_relags_query.py`采用dominant-Gaussian标签图+score阈值，是另一条旧输出方式，更不能拿它解释当前64.01。

**裁决：ReLaGS输出作为当前较强参照和失败分析对象；保留可见性/输出控制。暂停移植更多关系/GNN模块，直到有关系query证据；不把可见性补洞当成语义学习贡献。**

## 重算后的增益集中性：哪些话可以说

本轮离线脚本 `docs/audit_existing_assets.py`只读PNG/CSV，无torch模型导入；208组GT像素一致，预测面积与CSV一致，IoU最大差4.983e−7。完整权重、每query贡献、排除query重算见 `research_audit_assets/`。

| 子集 | 样本数 | 原正式权重下对总增益的贡献 |
|---|---:|---:|
| 双方非空 | 175 | +0.9008 pp |
| 仅THGS空 | 15 | +4.4685 pp |
| 仅ReLaGS空 | 8 | −0.2174 pp |
| 双方空 | 10 | 0 |
| 合计 | 208 | +5.1519 pp |

THGS共25个空、ReLaGS18个空。单向THGS空恢复占总净增益86.73%；**双向空相关的净贡献**是4.2511 pp，约82.52%，不是86.7%。空与非空划分由两方法预测事后确定，是描述性分解，不是随机化因果分解。

0.9008 pp是保留全体分母的**贡献**；175个双方非空样本直接平均的条件增益为+2.0762 pp；若沿用原正式权重并在该子集归一化，则+1.0818 pp。三者不能互称。以前任务把+0.90写成conditional gain，应改为contribution。

kamaboko七视角对四场景总增益贡献+2.1191 pp，比双方非空净贡献还大，其余非空部分合计−1.2183 pp。不能据此说所有非kamaboko都退化，waldo、buoy duck等确有正例；但可否定“双方非空存在均匀提升”。将kamaboko排除后重新计算每帧分母，总体仍+3.0280 pp——这主要仍包含空输出恢复，不能作为广泛语义收益的反证。

按场景，双方非空对各场景增益的贡献为figurines +0.5208、ramen +5.9343、teatime −0.9023、waldo −1.9498 pp。ramen其中kamaboko折算场景贡献约+8.4763 pp，其他非空query合计为负。**应保留kamaboko的真实成功，同时以leave-one-query-out和空输出控制约束新主张，而不是删掉难看的query或把它称异常值。**

## 为什么没有收敛：证据最支持的阻碍

### 已核验事实

1. **重复定位已知瓶颈，具体修复却不断更换。** 六月2D候选、七月3DNAG、九月Top-k、十月oracle都指向部分选择失败；后来校准/local-contrast已经有否定结果。新建议仍回到“重排、改粒度、保留多视角”，但经常未带上这些反例。[E06、E13–E18]
2. **贡献的算法对象和消融名字不一致。** B同时换聚合器；R同时换权重；AQ并非唯一算法；FCE有两种完全不同公式；scene-adaptive存在no-op和effective两版。于是积累的是多条系统配方，不是同一方法的可累计证据。
3. **候选/生产状态没有作为实验资产持续冻结。** 随机20D mask编码影响proposal关联及region-growing；相机顺序与seed一起决定NAG。官方PT只存最终特征/labels，缺逐视角映射，造成“信息被聚合丢失”的决定性检验无法回放。`STAGE_5B_FEATURE_LINEAGE.md`对此有明确缺项，不是还没想到怎么统计。
4. **核心反例已存在，但旧解释仍被沿用。** Alpha全量退化、粒度对照不支持专家分工、hubness null不支持集中性异常、2D context提升不兑现到3D、统一Top-k/NMS退化；这些都应迫使方法修改，而不是再给同一叙事找一个好场景。
5. **干预口径混淆放大了成绩。** 67.48含GT低分触发；scene-fill改变输出空间；MWP“adaptive”包含各场景事后择优；同名58.85有不同配方；207/208、dev/full、逐样本/逐图平均混杂。它们不都无效，但不能相加或共同证明一个原理。
6. **实现/接口失败确实截断过有价值的检验。** 聚合回放缺原随机状态；输出搜索曾在完成昂贵部分后两次报告失败；quality scoring实现后没有official结果；历史脚本覆盖源码、最佳mask与后来的PT不同源。这些是证据链中断，不证明方法本身必然错。

### 有依据的推断

主要阻碍是**“问题定位—新增信息—可检验判别规则—最终掩膜”的链没有完成，而不是实验数量不足**。有时只换了一个既有分数的单调/坐标变换；有时换输入使局部任务变好，却未检查它是否改变候选图；有时补空输出提高mIoU，却被解释成语义判别进步。上述情况是不同问题，不应统一归因于“单向量压缩”。

最大的路线风险是：用已有CLIP分数产生“支持图”，再拿它给同一批候选加权，可能只是原偏差的循环包装；用size/一致性产生“质量”则可能偏爱大物体、误伤真部件。下一步必须增加并隔离**空间对应**这一可检验信息，而不是给分数再起名字。

### 仍待验证，不能写成事实

- 鲁棒聚合是否系统性删除了少数但有用的部件视角证据；原58.85生产回放缺缓存，尚未得到证据生存率的有效对照。
- 单向量是否是本项目主瓶颈；2D阶段已有错排、部分融合修复，不能只从最终相似度或oracle gap推出。
- 查询范围与识别证据不一致是否是多数错误原因；小规模整体—部件对照并未支持普遍偏置，许多错候选甚至unrelated。
- ReLaGS空输出恢复究竟发生在剪枝、特征、搜索、可见性还是二值化；PNG分解不能定位上游因果。
- 新的空间支持核验能否超越简单可见性工程和现有方法；本轮不承诺成功。

方差也应拆开解释：同一冻结输入上十月pure三seed标准差只有0.1916 pp；waldo的R/RH可达2.57/4.55 pp，而teatime几乎不变。历史73/77/81的落差混合了候选拓扑、随机映射、实现和口径，不能都叫seed方差，更不该挑最高seed当新算法胜出的证据。多视角重复query也不是独立语义样本，bootstrap应按query/实体或场景聚类，不能把同一概念七视角当七次独立成功。

## 文献实时核验与创新边界

2026-10-23通过联网检索核验原文和作者仓库；使用可用搜索接口，不能确认其底层为Google，故不声称已完成Google专属搜索。以下只比较与裁决直接有关的内容，不把论文数字移入本项目消融。

| 工作 | 已覆盖的内容 | 对本项目的约束 |
|---|---|---|
| THGS | 图分区、多级合并、语义重投影和层级查询 | 不能把有层级或object/part查询本身当新增贡献。[原文](https://arxiv.org/html/2504.13153v1)、[作者代码](https://github.com/Atrovast/THGS) |
| ReLaGS | MWP、ROFA、层级与关系图；附录算法按child/parent分数决定下降，并按分数落差选节点 | 查询自适应层级搜索、剪枝、鲁棒对齐已覆盖。关系功能不能自动解释普通类别分割；本地64.01的具体实现还需运行快照核对。[原文§3/附录12](https://arxiv.org/html/2603.17605v1)、[代码](https://github.com/dfki-av/ReLaGS) |
| VALA | 可见贡献门控、余弦空间流式加权几何中位数融合 | “加可见性权重/鲁棒聚合”不是本项目新主线。其门控不等同于仅按候选大小乘分。[项目原文入口](https://vala3d.github.io/)、[代码](https://github.com/changandao/VALA) |
| LaGa | 跨视角分组，多描述子聚类和权重化语义检索 | 多向量、聚类后query聚合已有直接先例。[原文§5.4/附录](https://arxiv.org/html/2505.24746v1)、[代码](https://github.com/SJTU-DeepVisionLab/LaGa) |
| EviSplat | 逐实例保留观测bank，Gaussian局部证据分布，query-guided证据聚合 | “查询前保留证据、查询后再聚合”已被直接覆盖；不能把它作为我们独有的新思想。未定位作者代码链接，不能据此声称无代码。[官方原文§4](https://arxiv.org/html/2609.34853v1) |
| DIGS | 每Gaussian top-M标签分布、tempered Dirichlet更新、late-argmax、视觉与LLM描述编码 | 延迟硬决策、保留多峰标签、隔离实例命名与几何也已覆盖；本次不建议重做该整套映射。[原文§3](https://arxiv.org/html/2610.16177v1)、[作者仓库](https://github.com/ZephyrZos/DIGS) |
| FLAIR | 以文本条件attention pooling读取局部image tokens；与只做global CLIP点积的入口不同 | “加入文本条件注意力/更细粒度表示”本身不新；换模型还需token缓存和新的代价，现有512D行不能还原tokens。[官方论文](https://arxiv.org/abs/2412.03561)、[作者代码与接口](https://github.com/ExplainableML/flair) |
| QAGaussian | 学习query-conditioned Gaussian slots、关系推理、不同粒度/属性/part分支路由 | 广泛query-adaptive提法已覆盖；它含训练与query条件候选，不符合本项目query仅推理出现、候选构建与query无关的既有边界。其协议并非直接可替换本项目training-free冻结NAG。[原文](https://arxiv.org/html/2608.16103v1)、[代码](https://github.com/zqeslwyz/QAGaussian) |

THGS原HTML的占位会议DOI不能引用为真实DOI，以上采用官方arXiv/作者仓库。新近论文的结果仅按作者报告理解，本轮没有复现它们。本项目目前**没有足够证据证明一个区别于上述工作的完整新机制**。可继续的是一个受历史反例约束的具体假设，不能预先承诺投稿等级。

## 路线裁决：最多三个可行选项

| 选项 | 有利证据 | 主要反证/成本 | 决定 |
|---|---|---|---|
| 继续聚合/BSHRINK/HCONS体系 | 个别场景、特定底座有真实条件收益；已有共享投票工具 | 全栈未过pure，B归因不纯，旧production回放缺状态；与VALA/ROFA高度重叠 | **暂存**，仅作控制，不再投入参数搜索 |
| B0/Alpha或context的query路由/多表征 | 局部可恢复，oracle union有空间 | 总体替换退化；专家语义不成立；文本/尺度混杂；LaGa/EviSplat/FLAIR已有相关内容 | **不选主线**，保留观察对照 |
| 冻结候选的query空间支持核验 | 正确3D候选确有，纯语义重标定不足，扩候选FP失控，范围与可见性需要分开 | 同一CLIP可能循环自证；原缓存覆盖未确认；尚无新模块分割结果 | **唯一有条件推荐**，先做一次决定性对照，再决定是否实现正式方法 |

### 推荐方法具体改变什么

算法对象是**推理时对已有候选的打分目标**，新增query-independent的空间对应索引作为输入；不先改变Gaussian状态、NAG候选、候选数量或训练目标。不把多向量存储本身称创新。

已有状态：`sai_nag.pt = {nag, nag_feat}`，二维`*_s.npy`像素→proposal行号，`*_f.npy`存512D特征。生产过程本来就计算proposal关联、可见性和候选语义；但旧PT不完整保存这些信息。未来真正需要保存的是**可以核对某candidate为何覆盖某片像素的对应关系**，而不是再造一个融合后的向量。

建议独立sidecar（新接口，尚未实现）保存：

```text
candidate_key = (model_hash, level, node_id)
view_key = (camera_hash, image_name)
proposal_features[view, proposal]              # 既有特征，不提取新特征
proposal_masks[view, proposal]                 # 既有SAM图，构建时无query
candidate_visible_mask[candidate, view]        # 已有缓存投影/掩膜，或已保存对应索引
visible_mass[candidate, view]
valid_pixel_mask[view], mapping_hash
```

必须核对mask对应同一Gaussian/NAG/相机/分辨率；不能用新seed映射冒充旧生产状态。没有的缓存标记缺失，本轮不自动补采。

落点与接口变化：

1. `THGS-main/merge_proj.py:SAI3D.proj_gaussian_features_x`：未来可只追加对应索引/权重sidecar，不改变融合和labels；历史封存版本对应504行。若已有模型缺状态，先恢复匹配缓存而不是直接重跑覆盖。
2. `THGS-main/utils/vlm_utils.py:ClipSimMeasure.compute_similarity`：保留原relevance，新增对缓存proposal逐行计算的调用；不改变文本端或canonical negatives。
3. `THGS-main/test_lerf.py:training` 在算完`scores`、调用`get_related_gaussian`前加独立核验分支；输入原scores+sidecar+query，输出相同长度candidate scores和逐项诊断。
4. `THGS-main/nag_data.py:SemanticNAG.get_related_gaussian` 继续选相同L2/L3最终Top3、相同Gaussian并集；新分支先不改变这个选择空间。
5. renderer、固定0.5和官方 evaluator不改。现有本地THGS-main是带profile的副本，正式接入前需固定内容哈希；不能自动覆盖远端`merge_proj.py`。ReLaGS的运行入口尚未核对，本轮不虚构已修改函数。

### 一个可反驳的首版定义

这不是宣称最优公式，而是只选一个、不在测试集扫描系数的可实施起点。设 \(m_{vo}\) 为原二维proposal，\(r_{vo}(q)\) 为原编码器relevance。推理时在每个源视角选同一冻结bank里分数最高的proposal \(o_v^*(q)\)（固定ID打破平局），令 \(P_{qv}=m_{v o_v^*}\)。该proposal选择本身也可能错，必须输出其独立质量，不能称之为目标真值。

设 \(B_{cv}\) 为候选c的既有可见投影二值mask，仅在有效像素上计算

\[
I_{cvq}=\sum_p B_{cv}(p)P_{qv}(p),\quad
U_{cvq}=\sum_p B_{cv}(p)+\sum_p P_{qv}(p)-I_{cvq},
\]
\[
Q_c(q)=\frac{\sum_{v\in V_c}\omega_{cv}\,I_{cvq}/(U_{cvq}+\epsilon)}
                  {\sum_{v\in V_c}\omega_{cv}+\epsilon},\qquad
S_c(q)=r_c(q)Q_c(q).
\]

\(V_c\)、\(\omega_{cv}\) 来自query无关的几何可见性/贡献缓存；不看GT，不把不可见view当语义反例。`Q`通过分母同时惩罚缺失和多余覆盖；对相同区域的whole/part query可以不同，而固定size/child agreement无法做到这一点。候选在所有可用view无支持时标记不可核验，不默认制造高质量；确定性回退到原分数并记录覆盖率，不能只报可核验成功子集。

```text
for query in inference_queries:
    r = original_candidate_relevance(query)
    P = top1_existing_2d_proposal_per_source_view(query)  # 不用GT
    for candidate in frozen_L2_L3_bank:
        Q[candidate] = visible_weighted_mask_overlap(candidate, P)
    scores = r * Q where verifiable else r
    selected = original_top3(scores)
    mask = original_renderer_and_fixed_threshold(selected)
```

单个top1二维支持对多实例/多部件可能不完整，这是明确的失败预测，不准失败后直接加多套专家和可调注意力。若它只能恢复object空mask、在part或多实例退化，就停止这个首版，回到支持信息本身，而不是把oracle proposal接入推理。

与历史的**实质区别**仅在于：不把2D预测直接覆盖最终图，不把2D分数平均lift成新Gaussian场，不按nose/hoof设置cover，不以GT触发SoM，不只变换现有scalar score；而是使用固定三维候选在多个视角上的**完整投影范围**检验与推理query所选二维证据的一致性。是否构成相对现有文献的新贡献仍需结果和进一步核对；若独立信息不存在，这个区别只是一种实现变体。

计算/存储：已有proposal特征可复用，不训练，不用新文本模型；读出约 \(O(\sum_v O_v d)\)，mask交并可按bitset/稀疏mask计算，约 \(O(\sum_{c,v}|pixels_{cv}|)\)。不能把数千候选×数百高清view全量dense存储说成轻量；若用稀疏重叠表保留 \(I_{cvo}\)、候选面积、proposal面积，则推理查表，存储 \(O(\#非零对应)\)，但首次生成该表可能需要renderer，**不属于本轮授权动作**。已有单候选IoU标量不足以还原这些交集或任意并集；要明确缓存是否存在。像素bitwise OR也不必等于Gaussian并集经alpha渲染后阈值，最终成绩只能取同源exact缓存或日后获准的正式渲染。

### 唯一下一项决定性验证（本轮只设计）

问题：**真实二维—三维空间对应，是否为冻结3D候选选择增加了原语义分数和可见性无法提供的判别信息？**

使用同一模型/相机/候选哈希的已有投影mask、proposal bank和得分，包含所有缓存可覆盖query，不按GT低分筛任务；只对齐资产，不重提取、不训练、不改候选。原输入缺失必须公布覆盖清单；不得拿另一个seed或wave的缓存替代。

四个预先固定的对照属于同一实验，不是一串新流水线：

- 原relevance+原Top3，作为精确回放参照。
- 原relevance，仅排除在所有指定评估视角中完全不可见的候选，检验简单输出工程能解释多少；它是view-aware控制，须明确使用测试相机信息，不冒充纯3D检索能力。
- 上述真实空间支持核验，仍用相同候选Top3。
- **破坏空间对应控制**：不移动/改变任何mask，不改candidate投影；在同视角、同SAM层级、预定义的`floor(log2(mask_area))`面积分箱内，用固定种子0置换proposal特征行与mask ID的对应。逐view的分数多重集、完整maskbank及其面积/形状分布、候选/相机/可见权重均不变，改变的是语义证据对应哪片像素。不能宣称每个query选中的mask面积完全相同，需报告实际面积差、可置换覆盖率和singleton分箱；GT不参与置换或样本纳入。完整读取与真实组同样的输入，不引入新的编码器差异。

首先用现有单候选mask和逐query质量表比较Top1及固定大小候选pair的正确排序；GT只在评分完成后定义评价标签和可见/非空分层。成对比较要包含未改善query、低oracle query、whole/part局部反例和unrelated高分干扰，不能只看parent/child。若候选Top3 exact渲染已有同源缓存则同时报告；没有则只报告选择质量，**不从proxy生成一个mIoU胜利**。新的正式GPU验证须另行授权，本轮不执行。

预注册裁决建议（不是已跑结果）：

1. 真实组相对可见性组在双方非空、去除kamaboko后仍有正向候选质量改善，且明显优于空间破坏组，至少3场景同向，query/实体聚类置信区间支持正收益，才值得正式接入。若有完整exact缓存，额外要求正式宏平均净增益至少0.5 pp；绝不以oracle达到多少作验收。
2. 只有空mask减少、可见性组已取得同等收益：裁决为输出工程修复，停止语义机制叙事。
3. 真实与破坏空间对应组相同：放弃“空间对应提供新增判别信息”的本版机制，不能继续调融合系数追结果。
4. 二维支持本身错、候选oracle高：问题在支持预测/观测信息，不应继续微调旧分数；是否引入外部模型是新研究决策，不自动开始。
5. 支持正确却3D候选无法匹配：才有理由转候选构建/组合；不是一开始预设多向量或增维。

这些LERF query已被研究多次，即使做回顾性leave-one-scene-out也不是从未触碰的留出集。需要如实称开发证据；任何泛化/正式方法主张以后必须有新冻结的独立测试，不许把旧holdout标签重新命名恢复独立性。

**为什么不重复旧实验：** 六月SoM改二维输出且由GT触发；fine adapter把分数lift成场；七月oracle只证明存在性；后来的校准不增加空间信息；Alpha换了观察/编码器；原聚合回放只追信息生存且未通过缓存门；size+child quality尚无正式结果。上述实验没有做过“同语义、同候选、同可见性，只破坏空间对应”的有效对照。此次检验的是新增信息是否必要，能直接决定继续或放弃，而不是再增加一张高oracle表。

## 交付与可复核入口

- `experiment_evidence.csv`：按机制列出基线、改动、版本、结果、口径、限制和裁决；E编号是证据索引，不是新增实验阶段。
- `research_audit_assets/offline_verification.json`、`paired_mask_recheck.csv`、`scene_recheck.csv`、`gain_decomposition.csv`、`query_gain_concentration.csv`、`leave_one_query_out.csv`：本轮PNG重算。
- `research_audit_assets/ablation_recomputed.csv`：本轮由177行重算各主配置三seed宏平均。
- `research_audit_assets/candidate_dev_recheck.csv`、`alpha_metric_recheck.csv`、`vala_metric_reweight.csv`：由既有逐query表重算开发集候选策略、Alpha正式加权结果和VALA口径差异。
- `research_audit_assets/source_manifest.csv`：本轮实际可读代码和表的SHA256。
- `research_audit_assets/task_evidence_excerpt.json`：可访问任务的最终回复摘录，明确是二级证据，不冒充远端原始实验。

复核离线统计的命令（只写本审计衍生目录）：

```powershell
python C:/Users/TYY/Documents/ChatGPT/ZCY-4090/docs/audit_existing_assets.py
```

本轮研究审查的取舍是：**停止围绕历史最好数字证明旧解释；保留已有候选、区域条件观察与灾难性失败集中性三项发现；只给“空间支持是否提供独立判别信息”一次明确的证伪机会。** 如果它不能越过上述工程控制，就坦率承认当前证据还不支持新机制，不再扩成新的模块组合。
