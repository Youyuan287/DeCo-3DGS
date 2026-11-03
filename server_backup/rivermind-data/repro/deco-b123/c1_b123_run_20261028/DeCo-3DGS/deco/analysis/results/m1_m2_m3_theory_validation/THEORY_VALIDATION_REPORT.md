# DeCo M1/M2/M3 理论—数据一致性验证报告

本报告不使用 mIoU 作为理论正确性证据；只分析固定 official NAG/observation/bank、原始 DeCo canon 评分和 Gaussian renderer 投影。C1 固定 OFF，`text=canon`、`renorm=0`、`sel=gap`、`topk=5`、`level_until=3`、`filter_small=50`、temperature=0.08。

## 1. 核心理论预期

- M1 应由原始观测和真实 visibility weight 重建多个互补证据模式。
- M2 只在模式内移除不稳定成员，不跨模式合并或删除。
- M3 应使用已有模式分数和 prior 进行查询条件权重，而不是 Gaussian 级路由。

## 2. 全部数据验证结果

| 场景 | 区域 | 多模式区域 | 观测 | B1遗漏 | reconstruction | B2实际删除 |
|---|---:|---:|---:|---:|---|---:|
| ramen | 249 | 240 | 8470 | 0 | PASS | 120 |
| waldo_kitchen | 400 | 392 | 13425 | 0 | PASS | 185 |

B1/B2 的 region IDs、NAG SHA、observation SHA 均已核验；所有 M1 mode 向量均可由原始 CLIP 特征和成员权重重建。这里的 B2 删除量由 B1/B2 `members.json` 成员集合差恢复，不采用 metadata 的 `dropped_observation_count`。

## 3. 语义恢复诊断

| 场景 | 目标相关区域数 | 非目标区域数 | 目标 mean recover_gain | 非目标 mean recover_gain | 目标 mean max-slot | 非目标 mean max-slot |
|---|---:|---:|---:|---:|---:|---:|
| ramen | 277 | 17402 | 0.045981 | 0.059470 | 0.528435 | 0.458961 |
| waldo_kitchen | 111 | 8689 | 0.056209 | 0.064671 | 0.558206 | 0.443579 |

目标相关集合固定为 `intersection_px > 0`，包含边界接触和低纯度区域，不等于语义真值。两个场景中非目标区域的平均恢复增益反而更高，因此只能证明 M1 槽位产生了不同查询响应，不能宣布其恢复的是目标语义。

## 4. M1 观测来源与案例

- 自动选择了四组案例：目标相关最大恢复、非目标相关最大恢复、模式分数差异最大、最高分模式非最大先验。
- ramen 与 waldo 各生成 20 个去重 HTML 案例和 `semantic_evidence_case_index.csv`。
- 案例记录成员 observation/view、visibility weight、visible Gaussian ID 和 mode prior。
- 目前没有把任一案例标记为“语义确认正例”或“确认反例”：二维投影支持只说明模式看到了空间区域，不能替代人工 GT 语义核查。

## 5. M2 归因

- ramen 删除 120 条观测，waldo_kitchen 删除 185 条。
- 删除记录包含 observation ID、view ID、原始特征范数、visibility weight 和 visible Gaussian CSR。
- 在没有完成逐案例语义人工核查前，不能判定被删除观测是有效目标证据还是噪声。

## 6. M3 归因

- B3 使用同一 M2 bank；没有重新建 bank。
- `m3_weight_trace.csv` 对每个 query-region 记录 mode canon 分数、prior、fixed score、adaptive score 和 adaptive weights。
- 两个场景的 softmax 实现最大公式误差均为 0.0。
- M3 权重是区域级证据权重，不是 Gaussian 级路由。

## 7. 最早理论—实现偏差

数据映射、成员重建和多模式形成没有断裂；模式分数也确实随 query 改变。当前最早的可疑环节是固定 prior 融合可能把多个模式压回接近 Single Evidence 的整体方向，但 M1 恢复增益并未更多出现在目标相关区域。因此目前优先问题是“模式差异是否具有目标语义”，而不是继续改 M2 或 M3 常数。

## 8. 唯一优先修改建议

在任何算法修改前，优先为 `evidence_readout.combine(mode="fixed")` 保留并导出 query-region 的 mode 分解，同时完成自动选定案例的人工语义核查；不改变 bank 构建、ROFA tau、temperature 或 C1/C2。若人工核查确认模式差异与目标无关，才应将问题定位为 M1 的模式构建；若模式语义正确但固定融合压缩，再审查 fixed 读出。

## 9. 产物

- `query_region_support.parquet`
- `evidence_recovery.parquet`
- `evidence_recovery_summary.csv`（场景子目录）
- `mode_origin_summary.csv`
- `m2_removed_observations.csv`
- `m3_weight_trace.csv`
- `semantic_evidence_cases/`
- `semantic_evidence_case_index.csv`
