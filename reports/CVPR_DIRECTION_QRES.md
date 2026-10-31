# 从 67.60 到 CVPR：一个优雅的核心问题与解法

## 1. 为什么现在还不够

- **和已有工作重合**：top-k late interaction 本质是 ColBERT 式 MaxSim；多视角 mask 特征聚合与 OpenMask3D/Open3DIS 接近；mask 图关联与 MaskClustering 接近。
- **方法是 A+B**：module C（截断保护）+ mask rerank + gateE，工程上有效，但没有一个统一的目标函数。
- **收益集中**：deep（19 条）与 ramen；recall 29 条完全没解决（池内最佳 0.027）。
- **协议风险**：多 token 读出 vs baseline 单向量读出，必须声明；不能当作同读出优势。

CVPR 需要的是：**一个可以被一句话说清的原则 + 一个统一的目标 + 一个能解决 recall 的机制**，而不是更多 gating。

## 2. 我们实验真正揭示的原则

1. **身份是查询相对的（query-relative）**：同一区域，query 是 “table” 还是 “plate”，有效视角不同。
2. **身份是场景相对的（scene-relative）**：不同类特征余弦也高（ramen 间隔仅 0.066），因为共享/背景语义占主导。
3. **一致性/中心化是 query-agnostic 的**，所以它们优化的是共享成分，不能用于评分（found 13.7–15.3%）。
4. **压成一个向量会丢身份**；保留多视角 token + 查询时选择才有效（单向量 +0.56 vs late interaction +2.26）。
5. **候选池是文本驱动的**，池外实例永远救不回来；recall 需要 class-agnostic proposal。

一句话：

> **开放词汇 3D 身份不是特征向量的属性，而是“候选的视角证据集合”相对“当前 query + 场景背景”的关系。**

## 3. 优雅的核心：Scene-Relative Evidence Maximization（SREM）

对每个候选 c，固定 token 场 `T_c = {f_o}`（逐视角 mask 特征，建场不用文本）。
给定 query t，选择一个子集 `S ⊆ T_c`，最大化**相对场景背景的似然比**：

```
max_{S ⊆ T_c, |S| ≤ k}   F(S) = Σ_{o∈S} [ <f_o, t> / τ  −  λ · max_{b∈B} <f_o, b> / τ ]
                          − μ · Σ_{o<o'} <f_o, f_o'>
```

- 第一项：文本证据（object hypothesis）；
- 第二项：**场景背景对比**（background hypothesis），B 是其他候选/场景 token 池；
- 第三项：冗余惩罚，鼓励选择互补视角；
- `k` 不固定：**贪心加入，直到边际增益 ≤ 0**（自适应停止）。

得分 `s_c = F(S*)`；`λ=μ=0` 时退化为 top-k late interaction（我们的 0.6619），所以它是现有结果的**严格推广**。

### 为什么优雅

1. **单一目标**：文本对齐、类间分离、多视角保留、冗余控制，全在一个子模目标里；
2. **有理论**：线性项是模块函数，冗余项是负的成对相似度（非负权重）→ F 是**子模**；
   在“只接受正边际增益”的停止准则下，贪心是单调阶段的 `1−1/e` 近似（否则用非单调子模的
   `1/e`/局部保证）。这一点需要在论文里严格写清，不能笼统说“单调子模”。
3. **自适应 k**：边际增益停止准则等价于一个**序贯似然比检验**（SPRT），有统计解释；
4. **直接针对失败模式**：
   - 同类多视角高 → 第一项保留；
   - 不同类也高 → 第二项用场景背景把共享语义压掉；
   - 平均稀释 → 子集选择，不平均；
   - 一致性错 → 不用一致性；
5. **训练-free、场固定**：文本只在读出端；不违反标准协议。

### 背景 B 怎么建（三种，可消融）

- **候选间背景**：其他候选的全部 token（query-time，最直接）；
- **场景背景池**：全场景 token 的 k-means 原型（query-agnostic，可预计算）；
- **文本背景**：场景类别文本 bank（query-time；注意不要变成 lexicon 读端）。

## 4. 完整故事：Objectness before semantics

SREM 解决“给定候选怎么认”；要解决 recall，需要第二步：

1. **Class-agnostic 3D instance proposal**：逐视角 SAM mask 图 + 几何兼容性（lift 后 Gaussian 支撑重叠/可见性，不用特征一致性）→ 连通分量/谱聚类 → 3D instance tracks。
   - 参考 MaskClustering 的 view-consensus 思路，但用**几何支撑**而不是特征共识。
2. **SREM recognition**：每个 instance track 的逐视角 mask token 作为场，用 SREM 打分。
3. 这样 objectness 与 semantics 解耦：池外实例也能被提出，recall 不再由文本候选生成决定。

这就是 CVPR 级别的问题陈述：

> **现有 3D-OVS 把“物体从哪里来”和“它是什么”耦合在文本相似度里；
> 我们把 identity 形式化为 query-relative、scene-relative 的证据选择，并把 objectness 从语义中解耦。**

## 5. 与已有工作的差异（必须写清）

| 工作 | 做了什么 | 我们不同 |
|---|---|---|
| ColBERT | late interaction / MaxSim | 无背景对比、无子集选择、无自适应 k |
| OpenMask3D | 多视角 mask 特征聚合到 3D instance | 单向量/均值；无 query-relative 背景对比 |
| Open3DIS | 2D mask 引导 3D instance | 无 scene-relative 读出 |
| MaskClustering | 视角共识 mask 图关联 | 只做 association，不做 OV 读出 |
| PartSLIP++ | 多视角多原型 | 无背景假设/似然比 |
| ReLaGS/ROFA | query-agnostic 一致性 | 我们证明一致性不能用于评分 |

**新颖点不是 top-k 本身**，而是：
(a) 把 identity 写成 **scene-relative likelihood ratio**；
(b) **自适应子集选择**（SPRT/子模）；
(c) 与 **class-agnostic mask graph proposal** 结合，解决 recall。

## 6. 下一步实验路线（按优先级）

1. **SREM 读出（最便宜，先验证核心）**：
   在现有 segidentity token 场上实现 SREM（B=其他候选 token；贪心 k；λ/μ 预注册），
   与 id_top3 / id_mean 对比，看 found/deep/rank2/recall。
   若 SREM > top-k，核心成立。
2. **背景源消融**：候选间 / 场景池 / 文本 bank；λ/μ 敏感性；自适应 k 的分布。
3. **理论/实现**：证明 F 的子模性；给出贪心保证；SPRT 停止准则。
4. **Proposal**：mask 图 → 3D tracks；在 recall 29 条上评估 proposal recall；再跑 SREM。
5. **跨数据集/novel-view**：ScanNet/3D-OVS；novel-view 评测，证明不是查询视角捷径。

## 7. 预期贡献（如果 1–5 成立）

1. 一个**统一的目标**（SREM），把文本对齐、类间分离、多视角保留、自适应证据选择放在一起；
2. 一个**理论陈述**：query-relative/scene-relative identity；一致性是错误的不变量；
3. 一个**解决 recall 的机制**：class-agnostic mask graph proposal；
4. 实验：LERF-OVS + ScanNet/3D-OVS，found 不降、deep/recall 提升，novel-view 公平评测。

一句话：**不要再加 gating；把“识别”写成一个 scene-relative 证据最大化问题，把“物体从哪里来”从文本里解耦出来。**