# GLM独立实现审查（LERF-OVS）
用户已批准发送以下材料和单场景实验。请核对下列冻结设计、DeepSeek裁决和完整实现，不把其建议当实验结果。
重点：1. 是否保持原baseline候选搜索/最大gap行为？2. 是否有GT泄漏？3. 真实空间与打乱/可见性控制是否可识别？4. 是否存在阻塞运行的代码错误、缓存/相机接口错配？5. 指出DeepSeek可识别性段落中“sp不优于perm才弱一致”这句话逻辑是否颠倒，不盲从它。不要新加实验或调参，最多3项阻塞问题。若可跑则明确PASS_WITH_LIMITATIONS或PASS，不宣称有效。
当前baseline模型来源明确作者PT，非自建PT。只做OVS，未跑新方法。缓存编码权重仅发现旧提取代码默认B16同权重，尚无生成日志链，标注限制性pilot。全套12项CPU测试已通过，实际GPU baseline等价性由runner逐query与历史OVS CSV检查，不按结果改代码。计划先 --sanity 第一prompt，再固定完整56pairs。

# LERF-OVS figurines 空间支持核验：当前唯一有效任务

用户已明确纠正：LERF-OVS和LERF-Mask为两个数据集，必须在LERF-OVS取得实质性进展。先前LERF-Mask73.13%只作旁支重放记录，不进入本轮方法表。旧prompt涉及LERF-Mask的路径和数字已经作废。用户已授权发送本项目方法、非敏感路径和评分代码至本API供规划/审查。

历史约束：二维oracle/正确3D节点存在但没选中已做过；局部语义contrast全场景退化，跨层zscore未兑现为最终指标，扩大topK及统一NMS退化；Alpha全替换不支持整体/部件双专家；FCE/BSHRINK/HCONS全栈不优于匹配pure。只做一个受控推理假设，不追旧数字，不重新证明oracle，不直接二维补洞。

本轮真实资产：
- LERF-OVS evaluator=/root/rivermind-data/ReLaGS_Reproduction/scripts/eval_lerf_relags.py
- code=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS，SemanticNAG.search_matched_superpoint_in_mhtree
- baseline model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines/sai_nag.pt（作者PT来源，明确标注，不冒充自建PT），四层labels，234534 Gaussians，feat表2103/463/177×512。
- source=ref/hf_only/figurines，GT=ref/hf_lerf/lerf_ovs/label/figurines，56query-view/4图；历史mIoU=63.2871%，四场景64.0068%仅为该HF轨道参考。
- OVS原训练图对应缓存=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features，299帧，帧名frame_00001等，features[N,512]float16，seg[4,728,986]int32，-1背景、ID全局引用feature表。存在feature ID无mask的情况，query选proposal必须过滤实际有mask的ID。
- 未确认缓存具体编码器权重，不能只因512维宣称与baseline完全同源；执行端检查编码日志/源码及相机图像对齐。若未完全溯源明确限制，不新提特征。

已获准四组：baseline/visibility/spatial/permutation。保持topk5、level_until3、filter_small50、remove_smallTrue、语义渲染>.5；根top5及parent-child提升仍用原始CLIP sims。只在_analyze_leaf_candidates返回后与_filter_by_similarity_gap之前重打分，严格相同pre-gap候选集合。保持原最大gap规则（单候选可能全空，本轮不修复）。

拟固定公式（不扫描）：8个按相机名均匀抽取非评测视角；每候选用已有renderer投影，先>.5再最近邻降至1/4宽高。V_cv=投影mask非空。每帧从有效proposal中以原ClipSimMeasure规则选query top1，M_qv为对应已有mask；不读GT。
visibility证据=mean_v V_cv，spatial证据=mean_v[V_cv*IoU(B_cv,M_qv)]。新分数=原score×证据。若整个query池的可见性全0，所有组回原分数；若可见但空间IoU全0，不能只因全0就回原分数，要明确是否保持零分触发原gap空输出。请裁决这一点。
permutation在同帧/同SAM层/log2面积箱中固定seed0打乱feature→mask对应，保留同一个原始top1 feature、映射到打乱mask。非单例用随机循环错排，单例保留，记录真正top1被移动和实际像素变化率。保留mask bank、分数bank、面积总体分布，不声称每query精确面积不变。

请用不超过1200字裁决：是否允许开展此窄pilot、需修正的最多3项问题、具体fallback/公式、可识别性和停止规则。不要扩大路线，不输出长泛泛综述。当前没有证明创新或增益。

{
  "role": "planner",
  "model": "deepseek-v4.1-flash-1017",
  "status": "ok",
  "content": "**事实**：LERF-OVS figurines 56 query-view、4图；baseline `sai_nag.pt` 为作者PT；缓存299帧；本轮只允许LERF-OVS。**推断**：空间可见性与mask-IoU可能改善同候选集选择。**假设**：在相同pre-gap候选上，固定证据乘法可改变选择并提升配对mIoU，置换对应关系应削弱该效应。当前不证明创新或增益。\n\n**裁决**：建议允许开展此窄pilot，仅作单场景配对诊断；不授权扩大、不新增特征、不引用LERF-Mask结果。执行授权仍归用户。\n\n**最多3项修正**\n1. 预注册相机列表、非评测视角、投影尺寸与有效mask过滤；四组共享同一pre-gap候选集、topk5/level3/filter50/remove_small/gap规则。编码器权重未溯源则标注“限制性pilot”，不得称同源增益；不读GT。\n2. 裁决零IoU：仅当整个query池的可见性全0时，所有组回原分数；若可见但空间IoU全0，**保持零分，不回原分数**，按原gap规则输出（可空）。禁止单独因IoU全0回退。空间组空输出率须预注册为失败证据。\n3. 置换只替换`B_cv/M_qv`引用的mask，候选feature与`s0`不变；seed0固定。预注册有效性：单例比例≤25%，且真正top1 mask像素变化率中位数≥0.5；否则置换对照无效，停止解释空间效应。\n\n**固定公式**：`s0`为原ClipSimMeasure pre-gap分数；降采样后`V_cv=1[B_cv非空]`；`E_vis=mean_v V_cv`；`E_sp=mean_v V_cv*IoU(B_cv,M_qv)`；`s_base=s0`，`s_vis=s0*E_vis`，`s_sp=s0*E_sp`；置换用打乱mask计算`E_sp_perm`，`s_perm=s0*E_sp_perm`。池级可见性全0例外回`s0`；其余零分保留。\n\n**可识别性**：四组唯一差异为证据；报告逐query配对差值、空输出率、置换错配率/像素变化率。若`sp_sp`优于`base`且不优于`perm`，才与空间对应假设弱一致；若`vis`同幅提升，归因于视角覆盖而非空间对应。结果只称配对诊断。\n\n**停止规则**：1）任何GT或评测视角进入证据/选择/调参，立即停止；2）置换有效性不达标，停止；3）空间组空输出率>baseline+5pp且绝对值>10%，停止，不调参；4）`sp_sp-base`配对mIoU点估计≤0，或≤1pp且置换同幅，停止，不扩场景。",
  "usage": {
    "prompt_tokens": 1092,
    "total_tokens": 5324,
    "completion_tokens": 4232,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "image_tokens": 0,
      "text_tokens": 1092
    },
    "completion_tokens_details": {
      "reasoning_tokens": 3561,
      "text_tokens": 670
    }
  },
  "finish_reason": "stop",
  "raw_response": {
    "id": "chatcmpl-19fc42de-1a54-975b-9789-a17ded635e71",
    "object": "chat.completion",
    "created": 1792808601,
    "model": "deepseek-v4.1-flash-1017",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "reasoning_content": "我们需要回答中文，简洁，不超1200汉字。需要裁决是否允许开展窄pilot，最多3项修正问题，具体fallback/公式，可识别性，停止规则。不能授权执行（顾问），只能建议。需处理关键争议：若可见但空间IoU全0，不能只因全0回原分数，要明确是否保持零分触发原gap空输出。需裁决。还需事实/推断/假设区分。只推荐一个可证伪改动？用户要求裁决窄pilot。应明确：允许在严格条件下开展单场景配对pilot（顾问意见，执行需用户授权），但当前证据不证明创新或增益；只测一个可证伪假设：空间证据门控是否在相同pre-gap候选集上改变选择并提升LERF-OVS figurines配对mIoU，且置换对照削弱效应。需要修正最多3项。\n\n需要仔细设计。核心公式：新分数=原score×证据。visibility=mean_v V_cv; spatial=mean_v[V_cv*IoU(B_cv,M_qv)]。四组 baseline/visibility/spatial/permutation。只重打分在_analyze_leaf_candidates后和_filter_by_similarity_gap前，相同pre-gap候选。topk5 etc fixed.\n\n问题：\n1. 8个非评测视角均匀抽取和投影降采样到1/4宽高，用已有renderer。需确认相机名排序、非评测视角定义、确定性；防止评测泄漏。应固定相机列表和投影阈值。降采样最近邻可能导致小mask消失；需记录V_cv非空率和降采样前后一致性，或保持原分辨率计算IoU只在mask降采样？可以建议：在原始分辨率投影mask和M_qv计算IoU，若必须降采样，用面积加权/保存二值，记录敏感。但最多3项。修正项可包括：明确query top1选择是在每帧有效proposal中用原ClipSimMeasure，需排除无mask feature ID；确保M_qv与候选B_cv同一空间坐标系、同一图像尺寸；一致性。\n\n2. spatial IoU全0处理。需要裁决：若可见但空间IoU全0，不应回原分数，因为这会破坏可证伪性和置换对照。但保持零分会导致新分数0，原gap规则可能在所有候选0时输出空。我们需要明确fallback：采用加性平滑还是零分？用户公式固定乘法，不扫描。若保持零分触发原gap空输出，则spatial组可能大量空，是方法行为，需记录并作为失败/停止。但为了可识别性，不能悄悄回原分数。可设置：V_cv可见但IoU=0时，证据=0，新分数=0；若某query所有候选新分数=0，按原最大gap规则输出（可能空），不得回退。但需确保baseline/visibility对照同样处理。visibility证据若全query池可见性全0则全组回原分数（已定）。空间组若可见但IoU全0，保持零分，不单独回退；记录空输出率。若空输出率超过阈值（如>20%）停止并不调参。或者更细：spatial证据=mean_v[V_cv*IoU] 可为0。但乘法零会让原score完全消失，且候选间无区分，原gap可能选最大间隔但分数全0。原gap规则可能比较分数0，单候选全空。需明确“若整个query池的可见性全0”回原分数仅visibility。对于spatial，如果可见性非零但spatial全0，不 fallback，保持0，这是可证伪预测：置换组应也低，但空间组空输出率会高。也许这会导致无法评估mIoU？空输出可能被计为失败。这可以作为停止规则：spatial空输出率显著高于baseline（比如>2倍或>10%绝对）且mIoU下降，则拒绝。但需要获得信号。用户要求具体fallback/公式。我们可以裁决：不允许“全0就回原分数”；允许保持零分并触发原gap空输出，但需要预先登记空输出率作为失败证据。如果担心全0退化，可改证据为平滑：证据 = mean_v V_cv * (ε + (1-ε)IoU?) 但这是公式扫描，固定ε=0.5? 不行，会改变。用户拟固定公式，不扫描。可建议将空间证据定义为 IoU 本身，而不是乘法？但visibility证据是V，spatial是V*IoU。零分是合理。为保持可识别性，可加一个固定下限 ε=1e-3 给IoU? 这等于非零分数，排序由原score*1e-3，与baseline排序相同，等于空间组退化为baseline，不利于检验。所以最好不fallback，保持0并记录。裁决：允许保持零分触发原gap空输出；禁止把spatial全0解释为回退原分数。停止若空输出率>baseline+5pp或>10%（选一个）。需不超过3项。\n\n3. permutation可识别性。目前置换：同帧/同SAM层/log2面积箱，固定seed0打乱feature→mask，保留同一个原始top1 feature、映射到打乱mask。单例保留，非单例循环错排。需记录真正top1被移动和实际像素变化率。问题：保留同一个原始top1 feature映射到打乱mask，可能导致新候选的mask与feature语义错配，但分数bank? 需要确保打分使用原始score还是重新计算？新分数=原score×证据，原score对应候选feature；permutation中保留top1 feature，映射到打乱mask，那么候选B_cv应使用打乱mask投影？需要明确：permutation只打乱feature→mask对应关系，候选的feature不变？这样证据来自错误mask，模拟空间证据与语义候选解耦。可识别性需要：比较spatial vs permutation，如果空间增益来自真实对应，则permutation应削弱；但若置换后mask面积/层分布保留，仍可能因IoU随机。需要记录实际像素变化率>某阈值（如>50%查询的top1 mask IoU<0.5？）。可识别性还受8视角、1/4降采样影响。修正项可包括预注册置换的“实际错配率”并设置有效性：真正top1被移动的比例、移动后mask像素Jaccard中位数 < 0.5（或变化率>0.5），否则置换对照无效，停止。\n\n还需最多3项修正。可列：\nA. 执行前固定并记录相机列表、坐标/尺寸、有效mask过滤；确保8视角非评测且无GT读取；所有三组共享相同候选集和投影缓存。若编码器权重未溯源，不新提特征，只用现有缓存但标注限制；pilot不比较跨轨道。\nB. 零IoU裁决：spatial组保持零分，不 fallback；仅全体query可见性全0时回原分数。预注册空输出率停止阈值（例如>baseline空输出率+5pp且>10%？需具体）。visibility组可见性全0回原分数。permutation组同规则？置换组可见性使用打乱mask，若全0也回原分数？应同样处理，但需对称。可设仅当“整个query池的可见性全0”回原分数，适用于visibility/spatial/permutation；spatial的IoU全0不触发。这保持可证伪。\nC. 置换有效性：固定seed0，预注册错配率；若实际像素变化率中位数<0.5或真正top1未移动比例>20%，置换对照无效，停止不推广。还需记录单例比例；若单例过多（如>25%）则置换弱。\n\n可识别性：配对同候选集、同gap规则、同topk，四组唯一差异是证据；baseline原分数；visibility只乘可见性；spatial乘可见性×IoU；permutation用打乱mask计算spatial证据但保留原候选feature/score。这样可归因：若spatial>visibility和baseline且permutation不增，支持空间对应证据；若visibility也增，则可能是视角覆盖效应；若spatial与permutation同增，非对应性。需要报告逐query配对差值、空输出率、置换错配率。不声称创新。\n\n停止规则：单场景figurines，4组配对；预先固定。停止条件：1) 出现任何GT读取或评测视角进入证据/选择；2) 置换对照无效（错配率不足/单例过多）；3) 空间组空输出率>baseline+5pp或>10%（两者？可设空间空输出率超过baseline 5个百分点且绝对值>10%）；4) spatial组相对baseline配对mIoU无正增益或低于visibility，停止，不扩展。可加小样本噪声：56query，4图？mIoU。因只有56 query-view，需配对bootstrap？但预算。停止规则可设：若spatial相对baseline的配对mIoU差值≤0，或提升<1pp，停止（因为无实质）。但不要追旧数字。是否需要功效？可设预注册一个主要比较spatial vs baseline；若差值点估计≤0停止。若>0但<1pp且置换也同幅，停止。不扩大。\n\n需要回答“是否允许开展此窄pilot、需修正的最多3项问题、具体fallback/公式、可识别性和停止规则”。需明确“顾问意见，不能授权执行”。最终正文不超1200汉字。需要简洁。约900-1100字。需区分事实/推断/假设。可用小段落。\n\n注意用户要求“只推荐一个可证伪改动”。本pilot就是一个可证伪改动：在pre-gap相同候选集上以固定空间证据乘法重打分。需说明。不要推荐多个改动。但四组是同一改动对照。修正项不是新方法。\n\n需要裁决：允许开展，但仅作为诊断pilot，不授权扩大；结果不证明创新。若执行端未确认编码器同源，仍可用baseline缓存做“限制性pilot”，但不得宣称同源增益。是否允许？用户已授权发送材料，但执行需授权。我们可说“建议允许在以下预注册条件下开展；执行授权仍归用户”。因为当前没有证明增益。\n\n需具体公式：固定所有组：\ns0=原ClipSimMeasure候选分数（pre-gap）\nV_cv=1[投影mask非空>0.5]，降采样1/4最近邻后非空？注意先>.5再最近邻降至1/4。V_cv基于降采样后mask非空？用户写先>.5再最近邻降至1/4宽高。V_cv=投影mask非空。应基于降采样后mask非空。M_qv对应已有mask，也降采样。IoU(B_cv,M_qv)在降采样后计算。证据：\nE_vis=mean_v V_cv\nE_sp=mean_v V_cv*IoU(B_cv,M_qv)\ns_vis=s0*E_vis; s_sp=s0*E_sp; s_base=s0; permutation: 用打乱后的mask M'_qv 计算 E_sp_perm，s_perm=s0*E_sp_perm，但保留原top1 feature。注意对于每个候选c，M_qv是每帧query top1对应的mask，不随候选变？用户写每帧从有效proposal中以原ClipSimMeasure规则选query top1，M_qv为对应已有mask。所以M_qv是每帧固定的query mask，候选c与它IoU。permutation打乱feature→mask对应，保留同一个原始top1 feature、映射到打乱mask。这里有点歧义：打乱的是候选feature→mask对应还是query top1？需明确：在每帧/层/面积箱内打乱feature→mask映射，候选c的mask B_cv来自打乱后的mask；query top1 feature保留，其M_qv映射到打乱mask。或者只打乱feature→mask，保留原始top1 feature但取打乱mask作为M_qv。为可识别性，需确保候选feature和query feature不变，只替换mask。可写：只替换B_cv/M_qv所引用的mask，候选feature与s0不变。证据公式同spatial但用置换mask。若单例保留，记录。\n\nFallback：\n- 仅当整个query池所有候选的E_vis=0（投影全不可见）时，该组所有候选回s0。这适用于vis/sp/perm? 对spatial，若E_vis=0则E_sp=0，但也回s0？用户已定“若整个query池的可见性全0，所有组回原分数”。是。\n- 若E_vis非全0但E_sp=0，不回s0；保持s=0，按原gap规则，可能空输出。记录。\n- 不得因IoU全0单独回退。visibility组若E_vis=0且非全query池？对单候选E_vis=0则s=0，不回退。仅池级全0。\n\n可识别性：\n- 四组共享pre-gap候选集、topk、层、filter、gap规则；唯一差异证据。\n- permutation有效性预注册：seed0；记录top1被移动比例、mask像素变化率（1-IoU原始/置换）中位数；若单例比例>25%或变化率中位数<0.5，置换对照无效，停止解释。这个阈值需合理。变化率=1-IoU(B,B')，中位数<0.5意味着mask太相似。循环错排可能相似。可设>0.3? 需具体。选择0.5保守。\n- 若encoding未溯源，标注限制，不新提特征，不能称同源。\n\n停止规则：\n1. 任何GT/评测视角进入证据、候选选择或调参，立即停止。\n2. 置换无效（单例>25%或实际像素变化中位数<0.5），停止，不解释空间增益。\n3. 空间组空输出率>baseline空输出率+5pp且>10%（绝对值），停止，不调参。\n4. 主要配对比较s_sp - s_base的mIoU差≤0，或s_sp不优于s_vis且不优于s_perm，停止；不扩场景。若s_sp>0但<1pp，也停止（无实质）。是否加？可设“点估计≤1pp停止，不扩”。但可能过于严格。用户要停止规则。可设点估计≤0停止；若0<差值<1pp，暂不扩展，仅报告。不要太多。\n\n最多3项修正问题：可列三点：\n1. 预注册相机/投影/有效mask，编码器溯源限制；所有组共享缓存；不读GT。\n2. 全IoU零的裁决：保持零分不fallback；仅池级可见性全零回s0；预注册空输出率。\n3. 置换有效性指标和阈值；确保只替换mask不换feature/score；记录错配率。\n\n需要提及“当前没有证明创新或增益”作为事实/推断：历史数字是事实记录；本pilot是假设检验。\n\n最终正文不超过1200汉字。需要精炼。用标题。避免长综述。可约1000字。需明确“允许/不允许”：裁决：允许在严格预注册下开展窄pilot；不授权扩大。允许？顾问意见可说“建议允许”。因为用户问是否允许开展。我们不是授权者，但给裁决建议。用“建议：允许开展，但仅作为单场景配对诊断；执行授权须用户另行确认”。可以。\n\n需要确保不混入LERF-Mask。可以提“不引用LERF-Mask结果”。\n\n事实/推断/假设：开头三行。事实：资产和约束；推断：空间证据可能影响选择；假设：固定空间证据乘法在相同pre-gap候选上可改变并提升figurines配对mIoU，置换应削弱。这样清楚。\n\n公式简写。不超过1200汉字。数一下大概。用中文。",
          "content": "**事实**：LERF-OVS figurines 56 query-view、4图；baseline `sai_nag.pt` 为作者PT；缓存299帧；本轮只允许LERF-OVS。**推断**：空间可见性与mask-IoU可能改善同候选集选择。**假设**：在相同pre-gap候选上，固定证据乘法可改变选择并提升配对mIoU，置换对应关系应削弱该效应。当前不证明创新或增益。\n\n**裁决**：建议允许开展此窄pilot，仅作单场景配对诊断；不授权扩大、不新增特征、不引用LERF-Mask结果。执行授权仍归用户。\n\n**最多3项修正**\n1. 预注册相机列表、非评测视角、投影尺寸与有效mask过滤；四组共享同一pre-gap候选集、topk5/level3/filter50/remove_small/gap规则。编码器权重未溯源则标注“限制性pilot”，不得称同源增益；不读GT。\n2. 裁决零IoU：仅当整个query池的可见性全0时，所有组回原分数；若可见但空间IoU全0，**保持零分，不回原分数**，按原gap规则输出（可空）。禁止单独因IoU全0回退。空间组空输出率须预注册为失败证据。\n3. 置换只替换`B_cv/M_qv`引用的mask，候选feature与`s0`不变；seed0固定。预注册有效性：单例比例≤25%，且真正top1 mask像素变化率中位数≥0.5；否则置换对照无效，停止解释空间效应。\n\n**固定公式**：`s0`为原ClipSimMeasure pre-gap分数；降采样后`V_cv=1[B_cv非空]`；`E_vis=mean_v V_cv`；`E_sp=mean_v V_cv*IoU(B_cv,M_qv)`；`s_base=s0`，`s_vis=s0*E_vis`，`s_sp=s0*E_sp`；置换用打乱mask计算`E_sp_perm`，`s_perm=s0*E_sp_perm`。池级可见性全0例外回`s0`；其余零分保留。\n\n**可识别性**：四组唯一差异为证据；报告逐query配对差值、空输出率、置换错配率/像素变化率。若`sp_sp`优于`base`且不优于`perm`，才与空间对应假设弱一致；若`vis`同幅提升，归因于视角覆盖而非空间对应。结果只称配对诊断。\n\n**停止规则**：1）任何GT或评测视角进入证据/选择/调参，立即停止；2）置换有效性不达标，停止；3）空间组空输出率>baseline+5pp且绝对值>10%，停止，不调参；4）`sp_sp-base`配对mIoU点估计≤0，或≤1pp且置换同幅，停止，不扩场景。"
        },
        "finish_reason": "stop",
        "logprobs": null
      }
    ],
    "usage": {
      "prompt_tokens": 1092,
      "total_tokens": 5324,
      "completion_tokens": 4232,
      "prompt_tokens_details": {
        "cached_tokens": 0,
        "image_tokens": 0,
        "text_tokens": 1092
      },
      "completion_tokens_details": {
        "reasoning_tokens": 3561,
        "text_tokens": 670
      }
    }
  }
}
"""Pure cache-only spatial support primitives. No GT, model loading or I/O."""
import numpy as np


def select_feature(scores, seg):
    scores, seg = np.asarray(scores), np.asarray(seg)
    ids = np.unique(seg[seg >= 0])
    if not len(ids) or ids.max() >= len(scores):
        raise ValueError('No valid proposal masks or invalid feature indices')
    return int(ids[np.argmax(scores[ids])])


def mask_iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Mask shape mismatch')
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def evidence_scores(ious, visible):
    ious, visible = np.asarray(ious, float), np.asarray(visible, bool)
    if ious.shape != visible.shape or ious.ndim != 1:
        raise ValueError('Evidence must be matched one-dimensional arrays')
    if not np.isfinite(ious).all() or ((ious < 0) | (ious > 1)).any():
        raise ValueError('Invalid IoU')
    if not len(ious):
        return 0.0, 0.0
    return float(visible.mean()), float(np.where(visible, ious, 0).mean())


def gap_select(candidates, scores):
    """Original ReLaGS prefix rule, including empty single/tied candidate cases."""
    if len(candidates) != len(scores):
        raise ValueError('Candidates and scores differ in length')
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    gaps = [0.0] + [ranked[i][1] - ranked[i+1][1] for i in range(len(ranked)-1)]
    cut = gaps.index(max(gaps))
    return [item[0] for item in ranked[:cut]]


def rescore(base_scores, evidence, all_unobserved=False):
    if len(base_scores) != len(evidence):
        raise ValueError('Evidence and candidates differ in length')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in evidence):
        raise ValueError('Invalid evidence')
    if all_unobserved:
        return list(base_scores)
    return [float(s * e) for s, e in zip(base_scores, evidence)]


def proposal_permutation(seg, n_features, seed):
    """Feature ID -> mask ID derangement within view/level/log2(area) bins.

    Keeps the mask bank and score bank unchanged, NOT each selected mask's exact
    area. Singleton bins are unchanged and must be included in coverage reports.
    Feature IDs must be level-disjoint as in the actual LangSplat cache.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3 or n_features < 0:
        raise ValueError('Expected level x height x width segmentation')
    if (seg < -1).any() or (seg >= n_features).any():
        raise ValueError('Segmentation ID outside feature table')
    rng = np.random.default_rng(seed)
    mapping = np.arange(n_features)
    seen = set()
    eligible = moved = singleton = 0
    areas = np.zeros(n_features, dtype=np.int64)
    for layer in seg:
        ids, counts = np.unique(layer[layer >= 0], return_counts=True)
        if seen.intersection(ids.tolist()):
            raise ValueError('Feature IDs shared across levels')
        seen.update(ids.tolist())
        areas[ids] = counts
        bins = np.floor(np.log2(counts)).astype(int)
        eligible += len(ids)
        for bin_id in np.unique(bins):
            group = ids[bins == bin_id].copy()
            if len(group) == 1:
                singleton += 1
                continue
            rng.shuffle(group)
            mapping[group] = np.roll(group, 1)
            moved += len(group)
    return mapping, {
        'eligible_features': eligible, 'moved_features': moved,
        'singleton_features': singleton,
        'total_mask_area_preserved': bool(areas.sum() == areas[mapping].sum()),
    }

"""Frozen-pool LERF-OVS pilot. Independent inference, followed by GT evaluation.

Never writes the upstream repository or existing results. No training, SAM,
feature encoding or NAG construction. Proposal embeddings are existing caches.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from support_ops import evidence_scores, gap_select, mask_iou, proposal_permutation, rescore, select_feature

R = Path('/root/rivermind-data')
RA = R/'ReLaGS_Reproduction'
MODES = ('baseline', 'visibility', 'spatial', 'permutation')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def nodes(pool):
    return [[int(t[0]), int(t[1]), int(t[2]), float(t[3]), int(t[6])] for t in pool]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sanity', action='store_true', help='First prompt only, not a scene score')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing existing output directory: '+str(a.out))
    for path, reserve in [('/', 5), (R, 10)]:
        if shutil.disk_usage(path).free < reserve*2**30:
            raise SystemExit('Disk reserve violated: '+str(path))
    a.out.mkdir(parents=True)
    start = time.time()
    import cv2
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    spec = importlib.util.spec_from_file_location('ovs_original_eval', RA/'scripts/eval_lerf_relags.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    model = RA/'ref/relags_hf/figurines'
    source = RA/'ref/hf_only/figurines'
    gt_dir = RA/'ref/hf_lerf/lerf_ovs/label/figurines'
    cache = R/'VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features'
    repo = RA/'ReLaGS'
    # Query vocabulary and evaluation image names are public task inputs. Do not
    # pass polygon annotations into the inference context below.
    query_by_image = {}
    for path in sorted(gt_dir.glob('*.jpg')):
        objects = json.loads(path.with_suffix('.json').read_text())['objects']
        query_by_image[path.stem] = list(dict.fromkeys(o['category'] for o in objects))
    del objects
    prompts = list(dict.fromkeys(q for qs in query_by_image.values() for q in qs))
    if a.sanity:
        prompts = prompts[:1]
        query_by_image = {im: [q for q in qs if q in prompts] for im, qs in query_by_image.items()}
    # Strict non-evaluation views only. Choice cannot depend on query or GT IoU.
    frame_names = sorted(p.name[:-6] for p in cache.glob('*_f.npy') if p.name[:-6] not in query_by_image)
    support_names = [frame_names[i] for i in np.linspace(0, len(frame_names)-1, 8).astype(int)]
    manifest = dict(dataset='LERF-OVS', scene='figurines', sanity=a.sanity,
                    baseline_origin='author_released_PT_not_self_generated',
                    topk=5, level_until=3, filter_small=50, remove_small=True,
                    threshold=0.5, support_downsample=4, support_seed=0,
                    support_views=support_names, query_by_image=query_by_image,
                    score_rule='raw_score * mean(V * IoU); visibility=raw_score*mean(V)',
                    fallback='All modes use raw scores ONLY if entire pre-gap pool unobserved; zero-IoU observed pool remains zero',
                    input_sha256={str(p): sha(p) for p in [model/'sai_nag.pt',
                        model/'point_cloud/iteration_0/point_cloud.ply', repo/'nag_data.py',
                        repo/'utils/vlm_utils.py', repo/'gaussian_renderer/__init__.py',
                        RA/'scripts/eval_lerf_relags.py', Path(__file__), Path(__file__).with_name('support_ops.py')]})
    dump(a.out/'manifest.json', manifest)
    (a.out/'command.txt').write_text(' '.join(sys.argv)+'\n', encoding='utf-8')
    p = argparse.ArgumentParser()
    lp, pp = ev.ModelParams(p), ev.PipelineParams(p)
    p.add_argument('--path_pred', default='')
    pargs = p.parse_args(['-s', str(source), '-m', str(model)])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = ev.GaussianModel(dataset.sh_degree, 20)
    scene = ev.Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device='cuda')
    cams = {c.image_name:c for c in scene.getTrainCameras()}
    assert all(im in cams for im in query_by_image), 'Evaluation camera missing'
    assert all(im in cams for im in support_names), 'Support camera missing'
    nag = torch.load(model/'sai_nag.pt', map_location='cpu')
    snag = ev.SemanticNAG(nag['nag'], nag['nag_feat'])
    snag.feat = [f.cuda().float() for f in snag.feat]
    assert gaussians.get_xyz.shape[0] == snag.gaussian_num
    vlm = ev.ClipSimMeasure()
    vlm.load_model()
    banks = {}
    for name in support_names:
        fp, sp = cache/(name+'_f.npy'), cache/(name+'_s.npy')
        f, seg = np.load(fp), np.load(sp)
        assert f.shape[1] == snag.feat[0].shape[1]
        cam = cams[name]
        assert seg.shape[1:] == (cam.image_height, cam.image_width), 'Camera/cache pixel mismatch'
        h, w = seg.shape[1:]
        mapping, stats = proposal_permutation(seg, len(f), seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
        small = np.stack([cv2.resize(layer.astype(np.float32),(max(1,w//4),max(1,h//4)),interpolation=cv2.INTER_NEAREST).astype(np.int32) for layer in seg])
        banks[name] = dict(features=torch.from_numpy(f.astype(np.float32)).cuda(), seg=small,
                           valid=np.unique(seg[seg>=0]), mapping=mapping, stats=stats)
        manifest['input_sha256'][str(fp)] = sha(fp)
        manifest['input_sha256'][str(sp)] = sha(sp)
    manifest['permutation_stats'] = {name:b['stats'] for name,b in banks.items()}
    dump(a.out/'manifest.json', manifest)

    projection_cache = {}
    predictions = {}
    selections = {}
    query_records = []
    with torch.no_grad():
        for qi, prompt in enumerate(prompts):
            print(f'QUERY {qi+1}/{len(prompts)} {prompt}', flush=True)
            vlm.encode_text(prompt)
            sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
            vals, ids = snag._get_root_candidates(sims, 50)
            roots = snag._build_root_candidates_with_leaf_info(ids, vals, sims, len(snag.labels)-2)[:5]
            pool = snag._analyze_leaf_candidates(roots, len(snag.labels)-2)
            pool_hash = jhash(nodes(pool))
            selected_base = gap_select(pool, [float(t[3]) for t in pool])
            official_rel, official_nodes = snag.search_matched_superpoint_in_mhtree(sims, topk=5, level_until=3, filter_small=50, remove_small=True)
            assert nodes(selected_base) == nodes(official_nodes), 'Pool replay not original selector'
            assert torch.equal(snag._get_related_gaussians(selected_base), official_rel), 'Baseline Gaussian union differs'
            supports, support_log = {}, []
            for name, b in banks.items():
                pscores = vlm.compute_similarity(b['features']).cpu().numpy()
                # Full-resolution valid IDs, including tiny masks possibly lost on downsample.
                ids_valid = b['valid']
                feature_id = int(ids_valid[np.argmax(pscores[ids_valid])])
                perm_id = int(b['mapping'][feature_id])
                real = np.any(b['seg']==feature_id, axis=0)
                perm = np.any(b['seg']==perm_id, axis=0)
                supports[name] = (real, perm)
                support_log.append(dict(view=name, feature_id=feature_id, perm_mask_id=perm_id,
                                        id_changed=feature_id!=perm_id, mask_changed=not np.array_equal(real,perm),
                                        feature_score=float(pscores[feature_id]), real_pixels=int(real.sum()), perm_pixels=int(perm.sum())))
            visibility, spatial, permutation = [], [], []
            per_candidate = []
            for candidate in pool:
                level, _, index = map(int,candidate[:3])
                key = (level,index)
                if key not in projection_cache:
                    gaussians._semantics = (snag.labels[level]==index).float().unsqueeze(1).expand(-1,20).contiguous()
                    projected = {}
                    for name in support_names:
                        sem = ev.render(cams[name],gaussians,pipe,bg)['semantics'][0]
                        binary = (sem>0.5).cpu().numpy().astype(np.uint8)
                        h,w = banks[name]['seg'].shape[1:]
                        projected[name] = cv2.resize(binary,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
                    projection_cache[key] = projected
                vis, real_iou, perm_iou = [], [], []
                for name in support_names:
                    bmask = projection_cache[key][name]
                    vis.append(bool(bmask.any()))
                    real_iou.append(mask_iou(bmask,supports[name][0]))
                    perm_iou.append(mask_iou(bmask,supports[name][1]))
                vv, ss = evidence_scores(real_iou,vis)
                _, ps = evidence_scores(perm_iou,vis)
                visibility.append(vv); spatial.append(ss); permutation.append(ps)
                per_candidate.append(dict(node=[level,index], visibility=vv, spatial=ss, permutation=ps,
                                           per_view_visible=vis, per_view_real_iou=real_iou, per_view_perm_iou=perm_iou))
            all_unobserved = not any(visibility)
            raw_scores = [float(t[3]) for t in pool]
            mode_scores = dict(baseline=raw_scores,
                               visibility=rescore(raw_scores,visibility,all_unobserved),
                               spatial=rescore(raw_scores,spatial,all_unobserved),
                               permutation=rescore(raw_scores,permutation,all_unobserved))
            record = dict(prompt=prompt,pool=nodes(pool),pool_sha256=pool_hash,
                          supports=support_log,evidence=per_candidate,all_unobserved=all_unobserved,modes={})
            for mode in MODES:
                # Every mode consumes exactly the same immutable query-specific pool.
                assert jhash(nodes(pool)) == pool_hash
                chosen = gap_select(pool,mode_scores[mode])
                rel = snag._get_related_gaussians(chosen)
                union = rel[:,0].numpy().astype(np.uint8)
                record['modes'][mode] = dict(pool_sha256=pool_hash,scores=mode_scores[mode],
                                             selected=nodes(chosen),gaussian_count=int(union.sum()),
                                             gaussian_union_sha256=hashlib.sha256(union.tobytes()).hexdigest())
                selections[(mode,prompt)] = record['modes'][mode]
                gaussians._semantics = rel.expand(-1,20).cuda()
                for image, qs in query_by_image.items():
                    if prompt not in qs: continue
                    sem = ev.render(cams[image],gaussians,pipe,bg)['semantics'][0]
                    pred = (sem>0.5).cpu().numpy().astype(np.uint8)
                    d = a.out/mode/'masks'/image
                    d.mkdir(parents=True,exist_ok=True)
                    safe = prompt.replace(' ','_').replace('/','_')
                    cv2.imwrite(str(d/(safe+'.png')),pred*255)
                    predictions[(mode,image,prompt)] = pred
            query_records.append(record)
            dump(a.out/'query_diagnostics.json',query_records)
            print(f'QUERY_DONE pool={len(pool)} projection_nodes_cached={len(projection_cache)}', flush=True)
        # Inference complete. GT polygons become relevant only now.
        rows = []
        for image, qs in query_by_image.items():
            anno = json.loads((gt_dir/(image+'.json')).read_text())
            h,w = cams[image].image_height,cams[image].image_width
            for prompt in qs:
                gt = np.zeros((h,w),np.uint8)
                for obj in anno['objects']:
                    if obj['category']==prompt:
                        gt = np.maximum(gt,ev.polygon_to_mask((h,w),obj['segmentation']))
                for mode in MODES:
                    pred = predictions[(mode,image,prompt)]
                    union = np.logical_or(gt,pred).sum()
                    iou = float(np.logical_and(gt,pred).sum()/union) if union else float('nan')
                    selection = selections[(mode,prompt)]
                    rows.append(dict(mode=mode,scene='figurines',image=image,prompt=prompt,iou=iou,
                                     pred_px=int(pred.sum()),gt_px=int(gt.sum()),n_candidates=len(selection['selected']),
                                     gaussian_union_sha256=selection['gaussian_union_sha256'],
                                     pool_sha256=selection['pool_sha256']))
        with (a.out/'per_query.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        # Verify against existing OVS baseline without tuning to its score.
        historical=list(csv.DictReader((R/'ReLaGS_run_20261023/lerf_query_metrics.csv').open()))
        historical={(x['image'],x['prompt']):x for x in historical if x['scene']=='figurines'}
        base_rows=[x for x in rows if x['mode']=='baseline']
        differences=[dict(image=x['image'],prompt=x['prompt'],iou=x['iou'],historical=historical[(x['image'],x['prompt'])]['iou']) for x in base_rows if abs(x['iou']-float(historical[(x['image'],x['prompt'])]['iou']))>0.00000051 or x['pred_px']!=int(historical[(x['image'],x['prompt'])]['pred_px'])]
        dump(a.out/'baseline_recheck.json',dict(n_queries=len(base_rows),differences=differences))
        assert not differences, 'OVS original baseline replay differs; no method conclusion permitted'
        summary=[]
        for mode in MODES:
            rr=[x for x in rows if x['mode']==mode]
            means=[np.mean([x['iou'] for x in rr if x['image']==im]) for im in query_by_image if any(x['image']==im for x in rr)]
            summary.append(dict(mode=mode,miou=float(np.mean(means)),flat_miou=float(np.mean([x['iou'] for x in rr])),n_queries=len(rr)))
        status=dict(status='complete',dataset='LERF-OVS',sanity=a.sanity,summary=summary,
                    wall_seconds=time.time()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                    projection_nodes=len(projection_cache),new_artifact_bytes=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file()))
        dump(a.out/'FINAL_STATUS.json',status)
        print('FINAL_RESULT '+json.dumps(status),flush=True)


if __name__=='__main__':
    main()


