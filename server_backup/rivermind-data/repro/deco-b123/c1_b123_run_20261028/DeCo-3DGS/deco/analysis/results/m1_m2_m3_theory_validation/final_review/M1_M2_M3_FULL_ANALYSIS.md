# M1–M3 理论验证完整分析

本报告只整理已完成的固定输入验证，不运行新实验、调参、NAG/observation/bank 重建。统一命名：M1=去偏多证据构建，M2=模式内 ROFA，M3=查询条件组合，Single Evidence=单向量控制。C1 固定 OFF，text=canon，renorm=0，sel=gap，topk=5，level_until=3，filter_small=50，temperature=0.08。

## 一、完成情况

| 环节 | 状态 | 记录 | 结论 |
|---|---|---:|---|
| 查询与三维区域关联 | COMPLETED | ramen 17,679；waldo 8,800 | root mask 与 GT 支持已生成 |
| M1 语义恢复 | COMPLETED | 26,479 recovery rows | 数值差异存在，但非目标区域增益更高 |
| M1 观测来源 | COMPLETED | ramen 722；waldo 1,152 mode rows | 成员、view、weight、visible Gaussian 已追踪 |
| M2 删除分析 | COMPLETED | 120 + 185 删除 | 成员差已恢复 |
| M3 权重追踪 | COMPLETED | ramen 17,608；waldo 8,734 | softmax 公式误差 0 |
| 三维/GT 可视化 | PARTIAL | 40 HTML candidates | 已生成 root mask 与候选页面，未完成人工语义确认 |

## 二、输入与可追溯性

### ramen
- NAG: `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag.pt` SHA256 `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324`
- observations: `/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz` SHA256 `6ae30f6e03b01177b14e900a447d71ee788ef57a31cac9909b0abf0d33358232`
- Single Evidence bank SHA256 `844c4b04aa2494d64baab9b67ca2fdaa438c8e84cad6bbcfa76c74847b32a8c1`
- M1 bank SHA256 `73e4511df01725fb7b6e6ed235971c509434aa6603d1aaf233eba5cdd6f54aa1`
- M2 bank SHA256 `68f1b7ad3eee33670539caba1d49c853cc6bbef587f3bf4e4b0ec80235bdc81e`
- regions in support render: `249`; support rows `17679`; B1 trace `17608`; B3 trace `17608`

### waldo_kitchen
- NAG: `/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen/sai_nag.pt` SHA256 `6cd39db45d60031ab04eac18630a918c311890de22220e07678118a6da903098`
- observations: `/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz` SHA256 `2586d87fb86eb4dee8885889b6e057363450ef710e06d4e2ca8d9fa72702fbf8`
- Single Evidence bank SHA256 `ab45de3e74192c9fdad03c0027ddcb800a499fe9b9b92d29b5be7965a71a6ce6`
- M1 bank SHA256 `8bf6b51bce581045c588a5967b9219c7b67fdf818dc10e8193754c90e70a7ad3`
- M2 bank SHA256 `1c07ec101b9dfbb43f09d701fe0297779e7e92ba9d15aab0aa43225f4bf20ff2`
- regions in support render: `400`; support rows `8800`; B1 trace `8734`; B3 trace `8734`

## 三、M1：模式是否真实存在

| 场景 | mode 原始 CLIP pair cosine mean/min | residual pair cosine mean/min | mode pairs | 聚合方向 vs Single Evidence |
|---|---:|---:|---:|---:|
| ramen | 0.770134/0.465986 | 0.143910/-0.500417 | 708 | 见 theory audit summary（约 0.99999） |
| waldo_kitchen | 0.766769/0.396072 | 0.175614/-0.464301 | 1118 | 见 theory audit summary（约 0.99999） |

模式差异、模式查询响应和正确互补语义是三个不同命题。当前数据支持前两个，不自动支持第三个。

## 四、M1：查询恢复

| 场景 | 目标相关 mean recover_gain | 非目标 mean recover_gain | 目标 max-slot | 非目标 max-slot |
|---|---:|---:|---:|---:|
| ramen | 0.045981 | 0.059470 | 0.528435 | 0.458961 |
| waldo_kitchen | 0.056209 | 0.064671 | 0.558206 | 0.443579 |

目标相关集合固定为 intersection_px > 0，不能等同于正确语义区域。非目标区域的恢复增益更高，故 M1 目标语义恢复为 INCONCLUSIVE，不能写成成功。

## 五、M2：删除了什么

- ramen 删除 120 条 observation；waldo_kitchen 删除 185 条。
- 删除记录保存在 `M2_REMOVED_DETAIL.csv`，包含 region、mode、observation、view、visibility weight。
- 成员集合差与 bank 构建的组内 ROFA 配置一致，但没有自动证明删除观测是错误证据。
- 目标相关/非目标归属必须结合具体图像和 GT；当前没有人工确认，因此“删除有效目标证据”与“删除异常背景证据”均为 INCONCLUSIVE。

## 六、M3：是否正确使用证据

- M3 使用同一 M2 bank；adaptive weight 与 softmax(score/0.08 + log prior) 最大误差为 0。
- `FULL_EVIDENCE_TABLE.csv` 逐 mode 保存 Single Evidence、M1/M2 slot score、prior、fixed weight、M3 weight、GT precision/recall/IoU。
- M3 权重是区域级证据权重，不是 Gaussian 级路由。
- M3 的公式实现 SUPPORTED；M3 是否选择了语义正确模式仍 INCONCLUSIVE。

## 七、理论逐项判定

| 理论要求 | 判定 | 数据依据 |
|---|---|---|
| M1 保留不同语义模式 | SUPPORTED | 多模式区域、模式成员可重建、raw/residual pair cosine 非退化 |
| M1 恢复查询相关信息 | INCONCLUSIVE | slot score 有差异，但目标区域增益未超过背景 |
| M1 不普遍放大背景 | CONTRADICTED | 两场景非目标区域平均 recover_gain 更高 |
| M2 保留有效少数证据 | INCONCLUSIVE | 有删除数量，但无人工语义标签证明被删证据无效 |
| M2 抑制异常证据 | INCONCLUSIVE | 尚未建立删除前后目标/背景 canon 变化的人工确认链 |
| M3 能选择相关模式 | SUPPORTED（实现层）/INCONCLUSIVE（语义层） | softmax 公式精确，语义目标选择未确认 |
| M3 不放大错误证据 | INCONCLUSIVE | 需要人工核验代表案例 |
| 空间范围支持查询 | INCONCLUSIVE | root mask 已生成，但粗区域纯度和语义案例未人工确认 |

## 八、最早断裂点

当前最早有数据支持的风险点是 M1 的语义目标性，而不是数据映射、M2 实现或 M3 softmax 实现：模式形成和查询差异成立，但恢复增益在背景区域同样甚至更高。具体证据是 ramen 的 0.059470 > 0.045981、waldo 的 0.064671 > 0.056209。由于没有人工语义确认，不能进一步强行定位为 M1 模式构建、M2 删除或 M3 选择中的单一根因。

## 九、案例状态

- 已按客观规则生成 40 个候选 HTML 案例；规则和索引见 `semantic_evidence_case_index.csv`。
- 当前没有确认的正例或反例；CASE_GALLERY 只列候选，不将模式空间覆盖等同于语义正确。
- 需要人工查看 RGB、GT、root projection、mode visible Gaussian projection 后才能完成四类案例判定。

