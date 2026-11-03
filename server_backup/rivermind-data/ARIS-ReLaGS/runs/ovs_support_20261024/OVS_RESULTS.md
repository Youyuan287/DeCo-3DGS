# LERF-OVS figurines：空间支持核验首轮结果

日期：2026-10-24。用户批准单场景四组验证，并指定主数据集为LERF-OVS。

## 结论

**当前实现未提升指标，应停止扩场景及权重扫描。** 原始baseline为63.2871%，空间支持52.7068%，下降10.5803个百分点。真实对应优于打乱对应，说明此系统对对应关系敏感，但不能抵消其低于原始baseline的事实，更不能据此宣布核心贡献成立。

这不是“代码没生效”：空间组25/56条query-view的Gaussian并集改变；每query四组的pre-gap候选池哈希相同。新结果揭示两种不同机制：正确候选被不可靠二维top1支持直接否决，以及正确候选仍排第一、却因分数尺度改变而被最大gap规则附加其他候选。二者不能统称检索排序错误。

## 基线身份及数据集边界

- 本轮是**LERF-OVS**，figurines 4张评测图、56个query-view、21个不同query词。
- 模型是`/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines/sai_nag.pt`及该目录iteration_0几何，来源为作者发布PT。当前没有找到可完整追溯的自生成ReLaGS OVS PT，本轮不冒称从零语义场实验，也不将“未找到”写成“没有做过”。
- 本轮只修改推理候选评分，可在固定作者PT上得到严格配对；不支持声称改进了自建特征聚合或建图。
- 先前73.1269%的重放属于**LERF-Mask**、28query/4图，完全不进入本表。
- 本轮仅figurines，不更新四场景64.01%的历史均值，不与其他数据集成绩拼接。

## 四组实测

| 组别 | mIoU（%） | 相对baseline（pp） | query-view数 | 空mask数 |
|---|---:|---:|---:|---:|
| 原始baseline | 63.2871 | 0 | 56 | 2 |
| 可见性重评分 | 49.4952 | −13.7919 | 56 | 1 |
| 真实空间支持 | 52.7068 | −10.5803 | 56 | 2 |
| 打乱对应 | 16.6072 | −46.6799 | 56 | 23 |

主口径为图内query均值→四张图像均值。每图query数为17/13/11/15，不能用56条简单均值替代。baseline简单均值为62.7834%，与正式63.2871%不同。单次固定推理没有多seed均值±标准差，不伪造误差条。

完整阶段用时37.54秒，torch峰值已分配显存1838.31MiB，81个不同候选节点投影至8视角，新产物约0.88MiB（不含日志/源码）。另有首prompt sanity 12.78秒，仅3个query-view，不作场景指标。

## 实际实现

独立代码`../../scripts/run_ovs_support.py`与`../../scripts/support_ops.py`，导入原仓库，不覆盖原`nag_data.py`。

1. 原CLIP B16/laion2b_s34b_b88k文本及canon打分保持不变。
2. 原`_get_root_candidates`→`_build_root_candidates_with_leaf_info`取根top5→`_analyze_leaf_candidates`得到pre-gap候选池。parent/child提升始终使用原始语义分数。
3. 仅对该池重打分，再用与原`_filter_by_similarity_gap`一致的最大相邻gap保留前缀。单候选/全并列可能返回空的原行为保留，没有修复基线。
4. 仍从选中节点构造二值Gaussian并集、使用原renderer语义第1通道>.5生成最终mask。

所有组固定topk5、level_until3、filter_small50、remove_small=True、渲染阈值.5，同一NAG/几何/相机/候选池/evaluator。

### 新评分及空间证据

固定支持视角为frame_00001/00044/00086/00129/00172/00215/00257/00301。按缓存帧名排序、排除四个评测帧后均匀取8帧；不按query或GT选视角。每帧缓存源RGB与ReLaGS输入RGB的SHA256逐字节一致。

仅复用`/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features`的既有512维proposal特征及4层ID map，没有重新SAM或编码图像。

令r_c为原候选语义相关性，B_cv为节点在支持视角v的语义投影（二值化后最近邻降至1/4宽高），V_cv为该mask非空；M_qv为在该帧真实有mask的proposal中，以原CLIP规则选择的query最高分proposal。

```
baseline:    s_c = r_c
visibility:  s_c = r_c * mean_v V_cv
spatial:     s_c = r_c * mean_v [V_cv * IoU(B_cv, M_qv)]
permutation: 同spatial，但将最佳feature映射至打乱后的mask
```

只在整个query池都不可见时回原分数。本轮21query均未触发该回退。可见但IoU为0保持0，不能结果不佳后另加回退挽救。

置换在同帧/同SAM层/log2面积箱内随机循环错排，seed0为基础、帧名哈希确定子种子；非单例均移动，单例不动。保留整个mask bank与score bank，不声称每query精确面积不变。

## 有效性核验

- 15项本地及服务器CPU单测通过；前置检查包括评测帧真实匹配并排除、历史CSV覆盖全部评测query。
- 独立runner的baseline每query选中节点及Gaussian并集与原selector一致；56条IoU与历史六位CSV在舍入误差范围内匹配，pred_px逐条完全相同。见`baseline_recheck.json`，differences为空。
- 21query中每个query的四组pre-gap池哈希相同。候选选择/并集实际变化见`query_diagnostics.json`。
- 置换全bank单例121/2226=5.4358%；query top1 mask像素改变视角比例的中位数100%、最低75%，达到预注册识别要求。
- 真实top1 proposal降采样后消失2/168次，这是已记录限制，不能解释所有退化。
- 推理模块只接收任务词表，GT polygon在全部推理完成后用于评分；支持视角排除评测帧。
- 未改训练、NAG、原仓库、共享环境，无模型/数据删除。运行结束screen自动退出，GPU恢复空闲。

## 收益与损失分布

空间组相对baseline：4条改善、13条退化、39条不变；25条Gaussian并集改变。空mask总数虽都是2，但并非相同样本：bag在frame_00195被救回，tesla door handle在frame_00105新增空输出。

| 空间组贡献项 | 对正式场景mIoU的贡献（pp） |
|---|---:|
| 双方非空53对 | −12.0116 |
| 空/非空转换及双方空 | +1.4313 |
| 合计 | −10.5803 |

这里是对总体指标的加权贡献，不是对应子集的条件均值。**退化主要在非空预测范围，不能解释为“CLIP低于0.5导致空mask”。** 本轮.5是渲染后的语义二值化阈值，不是CLIP拒绝阈值。

逐query词加权贡献：pirate hat −6.5953pp，red apple −3.5657pp，rubber duck with hat −1.6335pp，miffy −1.5054pp；jake +1.2882pp，bag +1.4313pp，其余净贡献0。该分解保留全部query，不删失败类。

### 反例一：pirate hat——空间支持否决了本来正确的候选

原baseline选(level2,id75)，370Gaussians；四图IoU约.916/.820/.946/.913。该节点在8个支持视角7个可见，但与query全局最高分proposal的IoU八视角均为0。其原语义分数.469789因乘法变为0；空间组改选(level2,id70)，1102Gaussians，四图IoU均为0。

已核验事实：原正确候选并不缺可见性，是目前二维top1支持与它不重合。推断：单个全局top1 proposal不能可靠充当三维候选的硬否决证据。尚不能断定不重合究竟来自二维语义误认、proposal范围差异还是其他投影/观察偏差；不能据此宣称所有空间证据无效。

### 反例二：red apple——正确节点仍第一，但截断范围扩大

baseline选(level3,id43)，1531Gaussians。空间重评分后该节点仍是第一，分数约.10003；其他两个候选.06283/.05465，其余0/0。原最大gap规则此时截在第3名后，选中节点数从1变3，并集变4603Gaussians，导致范围污染。

所以不能把该退化解释成“正确节点掉出排名前列”。重评分与自适应候选数量共用一个score，排序和基数被同时改变，是明确的接口耦合。

对21query做无GPU排序反事实统计：空间组11个query的最终节点集合不变；其余10个中，有3个（rubber duck with hat、red apple、jake）若仅固定为baseline候选数量，新分数的前缀集合即与baseline相同。**这不是固定数量版本的最终mIoU结果**；其中jake恰有正收益，故也不能直接宣称“固定数量就能提高总指标”。

### 局部正例不能过度归因

- bag/frame_00195从0变.85877，visibility与spatial选择相同节点并同样救回，不能独占归因为查询空间核验。
- jake两图提升约+.347/+.427 IoU，说明规则有局部正例，但无法抵消其他query退化。
- 真实对应相对置换高36.0996pp，但置换本身严重破坏了选择；这是对应关系敏感性的证据，不是相对原方法的有效改进证据。

## ARIS角色及裁决边界

DeepSeek提供OVS方法规划，GLM独立检查实现。两者没有shell或执行权限，用户授权与执行责任在当前Codex。

GLM第一次调用耗尽推理预算，只有截断字符，不计为通过；根据[官方参数文档](https://docs.z.ai/api-reference/llm/chat-completion)设置reasoning_effort=low后取得有效审查，提出两项前置检查及修正DeepSeek结论比较方向。均已处理，未将模型意见当作指标。

规划正文：`planner_OVS_20261024.json`；有效实现审查：`reviewer_OVS_20261024_low.json`。本轮结果另交GLM复核，最终是否有有效正文以实际结果审查文件为准，不预填通过。

结果复核已完成，有效正文见`reviewer_OVS_result_20261024.json`。GLM同意本轮失败、停止扩场景及空/非空拆账；指出pirate hat的“top1不可靠”机制解释仍需一次投影/支持mask的独立可视化复核，才能排除坐标/尺度等实现因素。本报告把数字事实与该推断分开；不能直接认定二维语义一定误判。

### 无GPU的选择变化分桶

已在本轮缓存分数和节点列表上完成，明细见`selection_change_buckets.csv`。互斥分桶避免把“改选”与“排序恶化”重叠计算。空间组如下：

| 节点选择变化类型 | query-view数 | 对场景指标贡献（pp） |
|---|---:|---:|
| 最终节点集合不变 | 31 | 0 |
| 改回baseline候选数即可恢复原集合 | 10 | −3.9109 |
| 候选数不变、成员改变 | 4 | −6.5953 |
| 排序成员与数量同时改变 | 11 | −0.0741 |
| 合计 | 56 | −10.5803 |

第二桶是确定的节点集合反事实，不是新GPU方法成绩；第三桶对应pirate hat四图。第四桶不能不加干预就拆成独立“排名损失”和“数量损失”。本轮未保存支持视角的每个节点投影mask，仅保存分数、候选及最终输出，故GLM提出的一帧支持可视化尚未补齐；需要相同配置的诊断渲染或未来持久化投影缓存，不能拿另一视角替代。没有为此启动新GPU。

## 下一步裁决

当前乘法空间支持规则**停止扩场景，不叠FCE/BSHRINK/HCONS、不扫alpha、不追历史数字**。这不等于永久否定空间信息。

最小决定性后续应分清两件事，而非再证明高oracle：

1. 排序与截断的耦合：在现有分数缓存上先完成固定baseline数量的节点选择反事实清单；如需真实指标，另行预注册唯一一次对应渲染，不重建NAG/提特征。
2. 单top1空间证据的可靠性：针对pirate hat这类“高准确baseline节点被零支持否决”的反例，先查缓存中的支持来源和范围，不直接扩大融合权重。只有证明某种新的支持证据能保留这些候选，才值得再实现规则。

本轮已做完选择集合离线反事实，尚未运行固定数量版本的GPU评测；不能给它填写mIoU，也没有启动新实验。是否继续此路线须受这两个反例约束，而不是承诺更换权重必能提分。

## 产物与复现

本地同目录：`per_query.csv`（224行）、`query_diagnostics.json`、`manifest.json`（输入/源码哈希）、`baseline_recheck.json`、`FINAL_STATUS.json`，以及baseline/visibility/spatial/permutation四组预测mask目录。

服务器：`/root/rivermind-data/ARIS-ReLaGS/runs/ovs_support_20261024/`，包含sanity.log/full.log、tests.log、完成标志及上述产物。旧screen=aris-relags在完成后自然退出；日志仍可查看：

```bash
tail -n 50 /root/rivermind-data/ARIS-ReLaGS/runs/ovs_support_20261024/full.log
cat /root/rivermind-data/ARIS-ReLaGS/runs/ovs_support_20261024/full/FINAL_STATUS.json
```

如自行原样重跑，先确认GPU空闲及数据盘预留10GiB，再用**尚不存在的新输出目录**，不要重用本轮目录：

```bash
cd /root/rivermind-data/ARIS-ReLaGS
export RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
export PYTHONPATH="$RELAGS_REPO/ext:$RELAGS_REPO:/root/rivermind-data/ReLaGS_Reproduction/ext:/root/rivermind-data/ReLaGS_Reproduction"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python -u scripts/run_ovs_support.py --out runs/ovs_support_manual_new
```

程序拒绝覆盖已有输出目录。无--sanity为全部56query四组，--sanity仅第一query，不可把sanity成绩当场景指标。
