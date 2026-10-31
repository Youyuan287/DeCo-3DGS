# Query-Only 文本可靠性实验报告（FAIL）

日期：2026-11-01
目的：在**不使用任何其他类别/场景词表**的前提下，测试文本侧是否还有可用信号。
预注册：`analysis/PREREG_qgate.md`（本次实现按此冻结：4 个固定模板、τ=0.75）。

## 1. 设计（完全合规）

- 对每个 query，只用当前 query 的 4 个固定模板：
  `{}`, `a photo of a {}`, `the {}`, `a {} in the scene`。
- 在 released p5 候选池上，对每个模板算候选分数，取 top-1；可靠性 =
  `max_c (#templates top-1 == c) / 4`。
- 规则：`qgate = gateE  if reliability < 0.75 else C`（文本不确定时用 mask 证据，确定时保守）。
- 不接触场景类别 bank、不接触其他 query；图像特征/候选池不变。

## 2. 结果（B0，四场景 208 查询）

| 信号 | macro | figurines | ramen | teatime | waldo |
|---|---:|---:|---:|---:|---:|
| main | 0.6393 | 0.5859 | 0.5181 | 0.7939 | 0.6592 |
| module C | 0.6603 | 0.5941 | 0.5649 | 0.8223 | 0.6597 |
| mask rerank | 0.6619 | 0.6160 | 0.5791 | 0.7939 | 0.6585 |
| **gateE（当前最好）** | **0.6760** | **0.6129** | **0.6092** | **0.8223** | **0.6597** |
| qgate | 0.6603 | 0.5941 | 0.5649 | 0.8223 | 0.6597 |

qgate 等价于 module C：绝大多数查询 reliability ≥0.75，直接退回 C，丢掉 gateE 的 deep 收益。

分族：

| 族 | n | reliability 均值 | reliability<0.75 比例 | gateE | qgate |
|---|---:|---:|---:|---:|---:|
| found | 124 | 0.901 | 8.9% | 0.8312 | 0.8312 |
| rank2 | 36 | 0.903 | 5.6% | 0.6402 | 0.6402 |
| **deep** | 19 | **0.908** | **0.0%** | 0.6607 | **0.4446** |
| recall | 29 | 0.931 | 10.3% | 0.0198 | 0.0198 |

只有 7 条查询被 qgate 改变，全部是 gateE 本来救回的 deep 查询。

## 3. 为什么失败

1. **模板一致性不区分难度**：found/rank2/deep/recall 的 reliability 都在 0.90–0.93，
   deep 甚至 0.908；文本侧根本不“犹豫”。
2. **deep 的失败不是文本歧义，而是候选/关联失败**：正确候选在池里但排序不对，
   换几种说法仍然会指向同一个错误候选。
3. **qgate 的方向反了**：低 reliability 的查询反而更少；高 reliability 时退回 C，
   正好丢掉 mask 证据对 deep 的修复。
4. 4 个模板与 canon 的 prompt ensemble 高度重叠，信号本身接近饱和。

## 4. 结论

- **query-only 文本可靠性（模板一致性）不可用**；不能作为 gateE 的 gating 信号。
- 在开放词汇约束下（不能用场景词表），文本侧没有找到可用的额外判别信息。
- 这条文本侧线关闭。下一步应回到 **class-agnostic instance proposal / QCMG**，
  解决“候选/实例从哪里来”，而不是继续在文本侧找增量。

## 5. 产物

- 脚本：`analysis/lerf_qgate.py`
- 预注册：`analysis/PREREG_qgate.md`
- 结果：`analysis/results/20261101/qgate/`（四场景逐查询 CSV）