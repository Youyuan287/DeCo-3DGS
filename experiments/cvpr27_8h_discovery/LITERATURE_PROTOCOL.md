# 文献占位检查协议（固定流程）

> 起因：本环境内置 WebSearch / WebFetch 不可用（WebSearch 对任意查询返回空，
> WebFetch 报 "Unable to verify if domain is safe to fetch"）。实测 `curl` 直连
> `https://export.arxiv.org` 正常（200，约 2s），因此把检索固定为下面的脚本流程。
> 2026-11-03 用该流程查实：QAGaussian 已命名 "granularity mismatch / part-whole leakage"，
> DIGS 已做 training-free 实例发现，PTC 已做原型校准——**我们此前当作创新的诊断与机制
> 至少有三条已被占**。这就是本协议必须存在的原因。

## 1. 什么时候必须跑

| 触发点 | 必须回答的问题 |
|---|---|
| **提出任何一个方法/机制之前** | 这个机制是否已有 training-free 版本？ |
| **写下任何一句"我们首次/不同于"之前** | 被谁占了？我们的 delta 一句话能否说清？ |
| **最终报告/投稿前** | 全面复查一遍，写进 `NOVELTY_LOG.md` |

不允许凭记忆判断新颖性。搜不到 ≠ 新颖。

## 2. 怎么查

```bash
cd experiments/cvpr27_8h_discovery

# 自由文本（会自动抽词并 AND；太严会漏，建议配合 --cat）
python tools/litcheck.py "training-free granularity alignment 3D Gaussian" --cat cs.CV --since 2025-01-01

# 精确查询（推荐，可控）
python tools/litcheck.py --cat cs.CV --q 'all:"open-vocabulary" AND all:"granularity"' --since 2025-01-01

# 已知 id，拉摘要
python tools/litcheck.py --ids 2608.16103 --abstracts

# 记录到账本
python tools/litcheck.py --cat cs.CV --q '...' --since 2025-01-01 --record NOVELTY_LOG.md
```

**查询构造三条规则**

1. **按机制查，不按任务查。** 查 `"conformal" AND 3D`，不是查 `"open-vocabulary 3D segmentation"`
   ——任务名会把结果埋掉。
2. **必须加 `--cat cs.CV`。** 不加会把物理学（conformal field theory）灌进来，
   实测 `all:"conformal" AND all:"3D"` 12 条里 10 条是物理。
3. **至少查 2–4 个同义表述**（例如 `granularity mismatch` / `part-whole` / `scale ambiguity`），
   一个措辞的 0 命中不算证据。

## 3. 怎么判

| 判定 | 定义 | 动作 |
|---|---|---|
| **OCCUPIED** | 有工作做了同一机制，且同为 training-free | 放弃该点，或降级为"复现/对照" |
| **ADJACENT** | 机制相近但设定不同（学习式 vs training-free、2D vs 3D、分类 vs 检索） | 可以在论文里用，但必须写清一句话 delta |
| **OPEN** | 2–4 个查询均无同机制工作 | 仍标"未经验证"，不得写"first" |

**0 命中的解读限制**：严格 AND 查询 0 命中只说明"这几个词没有同时出现"，
不等于没人做过。必须再用 2 词查询 + 邻接机制名各查一次。

## 4. 记录字段（写进 `NOVELTY_LOG.md`）

```
日期 | 主张（一句话）| 查询 | 命中 | 判定 | 我们的 delta | 剩余风险
```

## 5. 故障处理

- 脚本退出码 **2** = arXiv 无响应。此时该主张标 **UNVERIFIED**，**不得**当作 OPEN。
- Semantic Scholar API 无 key 会 429，本轮不使用。
- 若以后内置 WebSearch 恢复，可并行使用，但记录仍以本脚本的可复现查询为准。

## 6. 与实验流程的衔接

每个方法类实验在 `EXPERIMENT_DECISION_LEDGER.md` 里的"动作"列，
必须先有一行对应的 `NOVELTY_LOG.md` 条目；没有文献占位检查的实验不进主攻。