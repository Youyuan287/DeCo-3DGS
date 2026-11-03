# DeCo-LI 方法说明

本轮仅替换 V2 的模式优势聚合器，保留原始 canon 评分、有效模式过滤、正向截断、coefficient/cap、NAG 得分写入和 gap 搜索。

## 聚合定义

- Softmax：冻结 V2 公式 `softmax(mode_score / 0.08 + log(prior))` 后对 positive advantage 加权。
- MaxSim：有效模式中 `max(clamp(mode_score-main, min=0))`。
- Prior-MaxSim：`max((prior/max(prior))*positive_advantage)`，非正先验不参与。
- M2 关闭；没有新增特征、观测、Gaussian 级缓存或 GT 规则。

## 验证
- 单元测试覆盖无有效模式、单模式、多模式边界、先验抑制、置换不变性、无效模式、非有限值和 Softmax 冻结公式一致性。
- E0/E1 先验验收通过后才运行 E2/E3。