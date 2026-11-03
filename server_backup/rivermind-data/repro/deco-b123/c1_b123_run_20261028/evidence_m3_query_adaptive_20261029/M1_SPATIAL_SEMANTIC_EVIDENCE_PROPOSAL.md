# M1：让模式描述符与真实二维语义关联的三维支持绑定

状态：完成代码定位与只读数据审计；下述为待验证方案，未修改 M1、未建 bank、未运行新性能实验。目标量是“描述符所表达的语义与被关联 Gaussian 的一致性”，并非模式数量或 Softmax 分数大小。以下可靠性指标是几何与语义关联代理，不是正确类别的概率。

## 1. 现有实现实际上构建什么

实际 ReLaGS 代码根目录：`/root/rivermind-data/repro/deco-b123/frozen_ramen_relags/`。

- `scene/dataset_readers.py:116-118,276-278`：读取 `language_features/<image>_s.npy` 的二维分割索引与 `<image>_f.npy` 的原始 CLIP 行；`fg_mask = seg_map != -1`。二维 mask 的编号与 CLIP 行号必须同源。
- `merge_proj.py:217-251`，`SAI3D.extract_gaussian_features()`：先把各层 mask 重编号为背景0、前景1起；将随机20维编码写入 mask，调用 `trace()`，从 `gaussian_semantics` 与 mask 编码的最大相似度恢复每 Gaussian 的 mask 标签，`sim.max > 0.85` 才接受，否则置0。输出 `[G,V]`。这不是以 Gaussian 中心直接投影到最近像素，而是现有 ray tracing 关联。
- `merge_proj.py:79-94`，`_get_superpoint_mask_ratio()`：以 root 标签和逐 Gaussian mask 标签做 bincount，去背景列后，在非背景关联计数内归一化。因此即使只有很少 Gaussian 被可靠关联，剩余关联也可能获得很高比例；该比例不是完整 root 的观测覆盖率。
- `merge_proj.py:133-215`，`proj_gaussian_features()`：小于0.3的 mask 比例置0；同一 `(root,view)` 只生成一个 `sp_feat = sp_mask_mat @ view_level_feature`，归一化后导出。一个描述符可以混合多个二维 mask 的 CLIP，并非一个 mask 一个 observation。
- 同一函数 `render_point()['weight'] > 1e-4` 得到 visible Gaussians，权重是可见数/root Gaussian数；导出的 visible CSR 是 `gau_mask & (gau2sp == root)`，没有同时要求 Gaussian 属于参与该描述符的二维 mask。member CSR 则是整个 root。

实际 M1 代码：`/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/lerf_ovs/evidence_modes.py`。

- `build_bank()` 先归一化原始 observation `f[N,512]`，以整场均值方向 `u` 构造去偏分组特征 `h=normalize(f-(f·u)u)`。
- 在每 root 内，以 `0.75*(h_i·h_j)+0.25*(2*Jaccard(visible_i,visible_j)-1)` 选中心并分组；K仍为3。
- 模式最终 `features[R,K,512]` 依然是成员原始 CLIP `f` 的 visibility-weighted 均值再归一化。去偏只改变分区，不会从最终描述符中去掉背景语义。
- `B1.npz` 当前 `b2_enabled=False`，因此 B1 辅助模式没有经过 `rofa_keep()`；主路径的原始 ROFA 仍存在于冻结 NAG 的生产描述符中。不能把“主路径有ROFA”写成“所有辅助观察都已被ROFA认证”。

`evidence_readout_v2.py:descriptors_v2()/auxiliary_correction()` 则将模式按原始 root 的 C1均值和 canon评分，正优势只加不减，并只修正 root 分数。最终输出仍是原始层级检索选择的 root/child Gaussian集合，没有使用“该高分模式对应的局部语义支持”进行局部定位。因此一个模式局部语义正确，也可能抬升一个包含背景的大root。

## 2. 四场景现有 bank 的直接审计

只读取本轮官方 `sai_nag.pt` 所绑定的 B1 bank、members、observations；不采用历史 `sai_nag_3.0.pt` 六图案例作同源性能证据。

|scene|observations|modes|单view模式|同root模式对|visible并集Jaccard中位数|均值|完全相同支持对|Jaccard≥0.9对|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|figurines|16800|531|54|531|0.290065|0.388592|13|73|
|ramen|8470|722|189|708|0.266635|0.364224|46|90|
|teatime|20096|3608|1327|3562|0.285714|0.403871|296|676|
|waldo_kitchen|13425|1152|327|1118|0.238095|0.372810|71|188|

方法：每模式取其 `observation_indices` 对应 visible CSR 的 Gaussian 并集，在同root模式间精确计算集合Jaccard。四场景 observation region 与成员所在 root 的不一致计数均为0。审计脚本为本地工作区 `audit_existing_m1_support.py`，远端执行副本 `/tmp/audit_existing_m1_support_20261029.py`。

解读：模式具有部分不同的可见空间支持，但它们不构成互斥空间划分，也不能仅凭低Jaccard证明语义互补。不同视角遮挡与相机视锥足以造成低重叠；完全同支持模式则更接近同一root的不同视角CLIP聚合。成员分组是observation分区，不是Gaussian分区。单view模式保存在bank，但当前辅助读出要求至少两个独立view，不等于它们已通过可靠性认证。

## 3. 为什么目标和背景可同时高响应

现有数据可以确认的结构原因：

1. root/view描述符是 mask CLIP 的混合；原始单个mask的CLIP还可能含裁剪上下文。去偏分区并不剥离最终原始CLIP中的这些成分。
2. visible CSR只证明“该root的这些Gaussian被该view看见”。CLIP向量可能主要来自root内另一部分Gaussian关联的mask。若用于分组的几何支持覆盖整个可见root，其判别力受限。
3. 混合比例在非背景计数内归一化，不能反映未关联/遮挡Gaussian的量；现有observation权重又按全部可见Gaussian算，两者分母不同。
4. 正优势读出没有“非目标”证据抵消；root级修正经gap检索可造成不连续的空间增删。ramen/corn中+57472 FP且TP不增、tesla中目标恢复、nori中删除背景均是这种候选竞争的不同结果。

不能从当前NPZ判定的部分：高分具体来自哪一个原始二维mask、该mask是否正确覆盖目标、是否为遮挡误关联或CLIP上下文泄漏。四场景现有NPZ仅含 features/weights/region_ids/view_ids及member/visible CSR八个键，没有 mask ID、CLIP混合系数、trace置信度或Gaussian到mask标签。没有这些关联，不应把所有高分模式称作正确互补语义，也不能把它们全部认定为噪声。六个nori的收益主要是候选删背景，不证明模式新观察到了目标。

## 4. 唯一建议：以语义关联支持替换 visibility-only 的分组几何项

最适合改动的函数是 `evidence_modes.py:weighted_overlap()`（支持定义）及其调用者 `build_bank()`（在相同K、分组策略和先验下接入）。必须先补齐数据边界：在 `merge_proj.py:proj_gaussian_features()` 的 `sp_mask_mat` 阈值化、矩阵乘前后保存关联sidecar；在 `extract_gaussian_features()` 返回前保存mask标签及最大相似度。只增加导出，不改原始trace、mask比例、CLIP或主路径ROFA。此处仅为修改计划，本轮未实施。

方案只替换几何相似度的来源，保留原始CLIP描述符和现有K=3、geo_weight=0.25、先验、M3参数。这是第一项最小可验证变化；ROFA与支持可靠性检查分别保留审计，不能把多个改动同时计作该消融的效应。

### 输入与需要保存的数据

- `f[N,512], w[N], region_ids[N], view_ids[N]`：沿用现有冻结observations。
- `label[V,G]`：原始trace接受后的局部mask标签，背景/不可信为0；`trace_conf[V,G]`：原始mask编码最大相似度；`visible[V,G]`：原有1e-4阈值布尔值。大张量可按view分块写CSR，不必常驻显存。
- 每view的 `seg_min/seg_max/feature_level`、局部mask标签到 `_f.npy` 原始行号映射。`local m>=1` 映射到原行 `seg_min+m-1`，必须用实际seg_map验证边界，不能假设每层mask从0起。
- 每observation的保留mask ID与阈值后比例 `beta[o,m]`（稀疏，保留原0.3规则）、归一化前CLIP混合向量范数、前景关联数/可见数/整个root数，区分三个分母。
- `semantic_support_offsets[N+1]`, `semantic_support_ids[L]`, `semantic_support_weights[L]`；每条support的原mask ID，便于追溯混合成分。
- frozen NAG、PLY、bank、observation、camera顺序、原二维_s/_f、trace实现与参数/随机编码seed或编码矩阵SHA。沿用之前已认证provenance，sidecar显式绑定源observation SHA。

现有NPZ无法无损逆推出上述label和beta；`features[512]` 混合归一化后丢失分解。可先读取现有冻结导出代码/缓存中已有逐Gaussian标签；缺失时只能在后续单独获准的数据补全中以同冻结模型/相机/mask重新执行原trace并保存sidecar，不重算CLIP、不训练、不生成NAG。本轮没有执行该过程。若随机编码/seed无法恢复，sidecar属于新版本关联，不能声称逐字节恢复旧observation。必须重算原公式的f,w并与旧observation逐条核对，匹配失败即阻断同观测消融；不准用近似中心投影或任意mask替代。

### 数学定义（工程假设，待验证）

对observation o=(root r, view v)，G_r为冻结root Gaussian集合，V_vg为原渲染可见性布尔值，L_vg为接受后的mask标签。原mask混合比例beta保持不变。定义非负语义支持：

`s_o(g) = 1[g∈G_r] V_vg Σ_m beta[o,m] 1[L_vg=m]`。

仅对原本参与描述符的mask求和；不把所有可见Gaussian当成语义支持。背景0、原trace未通过置信筛选的Gaussian贡献为0。trace_conf保留供审计，本版不再引入新的置信阈值或温度。`s`可以是一个描述符多个mask的加权支持，不伪称单mask实例。

替换现有binary可见Jaccard为：

`J_sem(i,j) = Σ_g min(s_i(g),s_j(g)) / Σ_g max(s_i(g),s_j(g))`；分母0时设0。

同root的模式发现仍用 `S_ij = 0.75*(h_i·h_j) + 0.25*(2*J_sem(i,j)-1)`，其余选中心、分配、加权原始f得到最终descriptor全部沿用。该变化使“看到相同root”变为“有一致的、参与CLIP混合的Gaussian语义关联”。不保证自动找到正确类别，也可能把不同局部目标分开；收益必须实测。

输出继续为 `features[R,K,512], priors[R,K], valid[R,K], region_ids[R]` 和相同结构members；新增每模式语义支持稀疏分布、支持独立view数、空支持标记、观测到mask的provenance。生产bank另存不覆盖。

### 可见性、少数可靠视角与ROFA

V仅回答是否可见；s回答该Gaussian是否被原trace关联到实际参与CLIP描述符的mask。两者必须分别保存。s正确也不能证明mask语义正确：背景裁剪上下文、多目标mask及共同错误仍是剩余风险。

不以模式质量等同于多数票或大prior。两条独立有效view在同一局部语义支持上重复观察，虽为少数也可保留；已有至少两view辅助资格继续使用，单view不因高canon得分获得资格。重复同一view不算独立支持。两view语义一致仍可能共同错误，不能称绝对可靠。

原始ROFA主路径完全保持。现有 `merge_proj.py:_filter_feature_outliers()` 以原始CLIP的mean pairwise similarity做lower-tail抑制；提议的支持Jaccard不能替代它。辅助bank的B1当前没有该过滤，应如实报告，不能宣称本方案自动继承了其抑错能力。

若后续需要把ROFA扩展到辅助模式，应作为另一独立条件调用已有 `rofa_keep()`，对空间语义一致的模式内原始CLIP做原版本lower-tail筛选，保留少数可靠局部模式，不在整root跨语义模式中多数表决。必须核对 `rofa_keep(std ddof=0)` 与生产 `_filter_feature_outliers(torch.std 默认校正)` 的具体差异，使用生产确切版本及tau，不能冒称两个实现完全相同。模式内样本太少时无法可靠识别离群：保持原有fallback并记录，不能把两条一致观测当成ROFA已证明正确。全局主路径过滤掉但辅助分组保留的成员需显式列出，量化是否重新引入错误视角；跨view共同错配也无法单靠ROFA消除。

## 5. 一个最小有效性消融（仅预注册，不执行）

在同一完整sidecar、同一冻结observations、同一raw CLIP f与所有参数下，只比较现有visibility Jaccard与上述semantic-association weighted Jaccard。保持debias、K、先验公式、所有ROFA开关及原主路径完全一致；M2关闭；M3使用既有全域绝对semantic读出及固定参数。

先做数据门槛：mask行号与view一一对应、重算f/w匹配旧观测、support ID属于对应root且为visible子集、beta匹配原阈值公式、无空支持静默fallback。恢复失败则停止并报告数据缺口。

再观察机制：模式语义支持跨view重现性与支持重叠、mask混合程度、单view/有效多view比例，以及被生产ROFA排除成员的重新参与率。GT只用于评测，不用于分组、支持定义或模式选择。

最后只做一次四场景对比：scene/flat mIoU分列，208条IoU、候选三元组、Gaussian真实集合、mask XOR与TP/FP/FN；专门检查nori/corn/tesla/ottolenghi。若只提高内部支持一致性但背景FP不减或净mIoU不增，应保留负结果。无需增加模式数。当前root级读出仍可能把正确局部模式扩成大root，这属于方案能力边界，本轮不扩展空间路由。

回答研究问题：M1需要首先恢复“原始mask语义—实际Gaussian支持”的可追溯绑定，并用该绑定替换visibility-only的分组几何信息，使相似模式具有可重复的局部语义依据；仅对同一区域跨view的混合CLIP做残差聚类，不能保证目标判别价值。该机制尚未获性能验证。
