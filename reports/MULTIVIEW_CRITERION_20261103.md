# 多视角判据：形式化方法说明

**用途**：把 DeCo / ReLaGS 管线中"多视角判据"的四个层次写成可判定的形式，供论文方法章、消融设计与失败归因使用。
**状态**：全部公式与参数均来自当前代码；标注「实测」的数字来自本仓库已有产物。
**日期**：2026-11-03

---

## 0. 范围

"多视角判据"在本项目中不是一条规则，而是四个层次的判定谓词。它们回答不同问题：

| 层 | 问题 | 输出 |
|---|---|---|
| L0 观测准入 | 哪个视角的哪份证据算数？ | 二值/实值归属 |
| L1 视角内聚合 | 单视角内如何得到一个向量？ | $h_v\in\mathbb R^d$ |
| L2 跨视角聚合 | 视角之间如何合成？ | $H\in\mathbb R^d$ |
| L3 一致性 | 视角之间是否互相印证？ | $\kappa\in[-1,1]$ |
| L4 判决 | 用这些证据改不改答案？ | 输出集合 $\mathcal S$ |

**核心论断**：L1–L4 都是对 L0 输出的确定性函数。因此**不可逆的信息损失只可能发生在 L0**。这是本说明对后续实验设计的直接约束。

---

## 1. 记号

- 高斯集合 $\mathcal G=\{g\}$，$N=|\mathcal G|$
- 训练视角集合 $\mathcal V=\{v\}$，$M=|\mathcal V|$
- 查询文本 $t\in\mathbb S^{d-1}$（CLIP 联合空间，$d=512$）
- 层级 $\ell\in\{0,\dots,L\}$，标签 $\lambda_\ell:\mathcal G\to\{0,\dots,K_\ell-1\}$
- 候选 $c=(\ell,r)$，其区域 $G_c=\{g:\lambda_\ell(g)=r\}$
- 观测单元 $o=(v,g)$，特征 $\phi(o)\in\mathbb S^{d-1}$，可见权重 $w(o)\in[0,1]$
- 相似度 $s(o,t)=\langle\phi(o),t\rangle$

---

## 2. L0：观测准入判据

### 2.1 归属函数的构造（官方实现）

对视角 $v$，令 $\mathcal M_v$ 为该视角的二维掩膜集合。构造一条**每个高斯至多一个归属**的函数

$$
\alpha_v:\mathcal G\to\{0,\dots,K_v\}\cup\{\bot\}
$$

其实现分三步（`merge_proj.py:193-227`）：

1. **编码**：抽取随机矩阵 $E\in\mathbb R^{K_v\times 20}$，$E_k\sim\mathcal N(0,I)$，逐行单位化。
2. **追踪**：用 max-contribution 光栅化，为每个高斯取**最大贡献像素**的语义编码
$$
\psi_v(g)=\text{argmax-pixel semantics of } g \text{ under view } v
$$
3. **反解与闸门**：
$$
k^*(g)=\arg\max_k \langle \psi_v(g), E_k\rangle,\qquad
\alpha_v(g)=
\begin{cases}
k^*(g), & \langle \psi_v(g),E_{k^*(g)}\rangle>\eta\\
\bot, & \text{otherwise}
\end{cases}
$$
   其中 $\eta=0.85$。

**注**（未在原文说明的标定）：在 20 维随机单位向量零假设下 $\langle\cdot,\cdot\rangle$ 的标准差为 $1/\sqrt{20}\approx0.224$，故 $\eta=0.85$ 约为 $3.8\sigma$。这是一个合理的宽松闸门，但**论文中必须声明这是零假设下的选择**，而不是调出来的。

### 2.2 判据谓词：覆盖率

官方用它算超点—掩膜归属比例：

$$
\pi_{v}(c,k)=\frac{\big|\{g\in G_c:\alpha_v(g)=k\}\big|}{\big|\{g\in G_c:\alpha_v(g)\neq\bot\}\big|}
$$

**分母只含被接受的高斯**。因此 $\pi_v$ 是一个**条件分布**，不是完整分布。这是 L0 最重要的一个性质：

> **C1（覆盖不完整）**：$\alpha_v(g)=\bot$ 的高斯既不参与分子也不参与分母，其证据状态是"未知"，与"不属于"不可区分。

DeCo 侧用另一种覆盖率（`lerf_identity_dump.py:213-221`）：

$$
\rho_v(c)=\mathrm{clip}\!\left(\frac{\sum_{g\in G_c} w(v,g)\cdot\mathbb 1[\alpha_v(g)\neq\bot]}{\sum_{g\in G_c} w(v,g)},\ 0,\ 1\right)
$$

### 2.3 判据谓词：准入

$$
\mathcal V_c(\tau)=\big\{\,v\in\mathcal V : \rho_v(c)\ge\tau\,\big\}
$$

代码另有一步 `dedupe_views`（`lerf_identity_dump.py:113-121`），表面上做"每视角只保留覆盖率最高的一个观测"。**但实测它在当前产物上是空操作**：

> 官方 ramen 导出（`official_nag_export_20261028`，131 视角 / 248 区域）：
> 8 470 个观测，唯一 $(\text{region},\text{view})$ 对也是 8 470 个，**重复组 0**。

因此"一个视角一票"不是由 `dedupe_views` 执行的判据，而是**上游观测导出器强加的数据格式**：导出器对每个 $(\text{region},\text{view})$ 恰好产出一个观测，此刻还没有计算过任何覆盖率。

> **C2′（先验位置）**：单视角单证词的约束位于 L0 的**生成端**，不在聚合端。它不是被验证的假设，而是被数据格式强制的结构。任何需要"同视角多份证据"的方法（如空间绑定的多证据表示）都无法在现有 observational bank 上评估，必须先重做 L0 的导出。

观测密度（ramen，按区域统计被多少个视角看到）：最小 1、四分位 7 / 16 / 49、最大 131；248 个区域中 8 个只被 1 个视角看到，176 个被 ≥8 个视角看到。

> **C2（有效样本量）**：多视角判据的理论样本量是视角数 $|\mathcal V_c^*|$，不是观测数。任何"观测越多越可靠"的叙述在此不成立——尤其在 $\tau$ 提高后，中位区域只剩 7–16 个视角。

---

## 3. L1：视角内聚合判据

### 3.1 官方：掩膜特征混合

$$
\tilde h_v(c)=\sum_{k}\ \pi_v(c,k)\cdot f_{v,k},
\qquad
h_v(c)=\frac{\tilde h_v(c)}{\|\tilde h_v(c)\|}
$$

其中 $f_{v,k}$ 是视角 $v$ 第 $k$ 个掩膜的 CLIP 特征。

**这是唯一一步把"多个不同语义"合成为一个值**：若一个区域在一个视角里同时命中 $k_1,k_2$ 两个掩膜且 $\pi$ 均超过阈值 $0.3$，则 $h_v(c)$ 是两个方向的线性混合。

> **C3（单视角内的多义合并）**：$\tilde h_v(c)$ 的方向由权重 $\pi_v(c,\cdot)$ 唯一决定。仅给定 $h_v(c)$ 无法恢复 $\{\pi_v(c,k)\}$ 与 $\{f_{v,k}\}$ 的对应关系。

### 3.2 可见性加权

$$
\lambda_v(c)=\frac{\big|\{g\in G_c: w(v,g)>\varepsilon\}\big|}{|G_c|},\qquad \varepsilon=10^{-4}
$$

官方把 $h_v(c)$ 乘以 $\lambda_v(c)$ 后再交给 L2。由于 L2 使用余弦，$\lambda$ 只在**加权平均**中起作用，在**离群判定**中不起作用（余弦对正标量不变）。

### 3.3 DeCo：可见性 × 覆盖率的证据权重

$$
a_v(c)=\frac{\lambda_v(c)\,\rho_v(c)}{\sum_{v'\in\mathcal V_c^*}\lambda_{v'}(c)\,\rho_{v'}(c)}
$$

---

## 4. L2：跨视角聚合判据

三种实现并存，**不可互换**：

### 4.1 ROFA（官方，离群剔除后均值）

令 $\mathcal U_c=\{h_v(c):v\in\mathcal V\}$，两两余弦均值

$$
\bar\kappa_v=\frac{1}{|\mathcal U_c|-1}\sum_{u\in\mathcal U_c\setminus\{h_v\}}\langle h_v,u\rangle
$$

保留集 $\mathcal U_c^{+}=\{h_v:\bar\kappa_v>\mu_\kappa-\tau_\kappa\sigma_\kappa\}$，$\tau_\kappa=3$，最终

$$
H(c)=\frac{\sum_{h\in\mathcal U_c^{+}} h}{\big\|\sum_{h\in\mathcal U_c^{+}} h\big\|}
$$

### 4.2 VCLS（视角中心化 + top-$k$）

逐 (区域, 视角) 求观测均值后得 $s_v(c,t)$，再

$$
s'_v(c,t)=s_v(c,t)-\alpha\cdot\frac{1}{|\mathcal R|}\sum_{r'\in\mathcal R}s_v((\ell,r'),t)
$$

$$
S_{\text{VCLS}}(c,t)=\frac{1}{k}\sum_{v\in\text{top-}k_v}s'_v(c,t),\qquad k=3
$$

**注意 $k=3$ 而 $M$ 可达 299**：这是强稀疏选择，不是平均。

### 4.3 一致性加权均值（DeCo 证据侧）

$$
\bar\phi(c,t)=\sum_{v\in\mathcal V_c^*}a_v(c)\,h_v(c)
$$

$$
\mathrm{id\_mean}(c,t)=\Big\langle \tfrac{\bar\phi}{\|\bar\phi\|},t\Big\rangle,
\qquad
\mathrm{id\_top3}(c,t)=\frac{1}{3}\sum_{v\in\text{top-}3}\langle h_v,t\rangle
$$

> **C4（聚合不是冗余）**：ROFA 假设视角间应一致，因而剔除偏离者；VCLS 假设视角间有系统偏差，因而减去公共分量；一致性加权假设视角可靠度不同，因而按可信度加权。三者的失效方向互不相同（见 §7）。

---

## 5. L3：一致性判据

$$
\kappa(c)=\frac{1}{n(n-1)}\sum_{v\neq v'}\langle h_v,h_{v'}\rangle=\frac{\big\|\sum_v h_v\big\|^2-n}{n(n-1)}
$$

居中版本：令 $\mu_v$ 为视角 $v$ 的特征均值，

$$
\tilde h_v=h_v-\mu_v,\qquad \kappa_c(c)=\kappa\big(\{\tilde h_v\}\big)
$$

### 命题 1（居中不可省略）

若所有视角共享公共分量 $\mu$，即 $h_v=\mu+\epsilon_v$ 且 $\|\epsilon_v\|\ll\|\mu\|$，则

$$
\kappa\ \to\ 1-\frac{1}{2}\mathbb E\frac{\|\epsilon_v-\epsilon_{v'}\|^2}{\|\mu\|^2}\ \approx\ 1
$$

对任意 $\{\epsilon_v\}$ 都接近 1，**判据饱和，没有判别力**；而 $\kappa_c$ 只依赖 $\{\epsilon_v\}$。

**实测佐证**：未居中的 $\kappa$ 是反向信号（valid 0.46、deep 0.377）；居中后 $\kappa_c$ 的困难查询 AUC 达 0.807，对区域尺寸残差化后仍 0.758。

> **推论**：任何"跨视角一致性"指标，若不加居中，度量的是视角间公共背景，不是身份证据。

---

## 6. L4：判决判据

令 $S_0$ 为现任答案集（released gap 前缀），$S_M$ 为挑战者集（重排后同长度前缀），

$$
S_c=\mathrm{z}\big(s(c,t)\big)+\lambda\,\mathrm{z}\big(\mathrm{id\_top3}(c,t)\big),\qquad \lambda=1
$$

$$
e(S)=\max_{c\in S}\mathrm{id\_top3}(c,t)
$$

**配对证词判据**：

$$
\text{改判}\iff \Big[\,S_M\neq S_0\,\Big]\ \wedge\ \Big[\,e(S_M)>e(S_0)\,\Big]
$$

> **C5（$\max$ 与 $\mathrm{mean}$ 不可互换）**：$e$ 用 $\max$ 度量"**存在**一个视角强支持"，L2 的 $\mathrm{id\_mean}$ 度量"**普遍**支持"。对细粒度目标，正确候选往往只在少数视角清晰可见，因此 $\max$ 判据与 $\mathrm{mean}$ 判据在 deep 族上方向相反。
>
> **C6（改判必须配对）**：判据只在同一查询的两个答案之间比较，不跨查询、不跨类别。这保证了它不引入场景先验，满足逐 query 独立协议。

### 6.1 低证词数下的弃权规则

实现中对证词数设了**硬下限 $n_{\min}=3$**：覆盖观测数 $<3$ 的候选，其 $\mathrm{id\_top3}$ 与 $\mathrm{id\_mean}$ 均为空值（`lerf_gate_trainingfree.py:48-52`）：

$$
e(c)=
\begin{cases}
\mathrm{id\_top3}(c), & n_c\ge 3\\
\bar e_{q}, & n_c<3
\end{cases}
\qquad
\bar e_q=\text{同一查询中双侧有值候选的均值}
$$

由于改判要求 $e(S_M)>e(S_0)$ 严格成立，**两侧均为填补值时无法触发改判**。因此实际行为是：

> **C7（弃权而非退化）**：证词数 $<3$ 的候选既不进分数（被填补为该查询的均值），也不进判决（无法严格优于对手）。它既不被奖励也不被惩罚，但**永远无法赢得挑战**。

实测分布（4 场景 208 查询 1097 候选）：$n_c\in\{0,1,2\}$ 共 108 个候选（**9.8%**）证据为空；$n_c\ge 3$ 共 989 个。

### 6.2 $e$ 的样本量偏置（重要）

$e=\mathrm{id\_top3}$ 是 $n_c$ 个视角相似度的 top-3 均值，其期望随 $n_c$ 单调上升：

$$
\mathbb E[\mathrm{top\text{-}3} \mid n]\ \text{随}\ n\ \text{递增}
$$

实测（4 场景 1097 候选，查询内去均值后）：

$$
\mathrm{corr}\big(\log(1+n_c),\ e\big)=0.656,
\qquad
\mathrm{corr}\big(\log(1+n_c),\ \mathrm{IoU}\big)=0.290
$$

分桶均值：$n\in[3,7]\to0.195$，$n\in[8,31]\to0.233$，$n\ge32\to0.268$。

> **C8（覆盖率混淆）**：配对证词判据在组内主要以**覆盖观测数**区分候选，而不是以语义身份区分。$e$ 与 $n_c$ 的相关（0.656）是它与正确性相关（0.290）的两倍以上。

这条与两轴分解中 $a_g$ 的失效原因同源：$a_g$ 与"区域语义质量"正相关，因此不可辨识为纯偏置。$e$ 同理——在 LERF 上"覆盖多的候选更可能正确"确实成立（真实物体大、可见次数多），所以 $e$ 有效；但它的有效性**来自尺寸/覆盖先验，而非身份证据**。

---

## 7. 算法

```
输入: G, V, t, 层级 λ, 参数 (η=0.85, τ=0.3, τκ=3, k=3, λ=1)
输出: 预测掩膜 M

# L0 观测准入
for v in V:
    E  ← 随机单位化编码 (K_v × 20)          # 无 seed
    ψ  ← max-contrib-pixel 追踪( G, E, v )
    α_v, conf_v ← argmax_k ⟨ψ, E_k⟩, 连带 η 闸门
    ρ_v(c) ← 覆盖率(α_v, w)
V_c* ← dedupe_views{ v : ρ_v(c) ≥ τ }

# L1 视角内聚合
for v in V:
    h_v ← normalize( Σ_k π_v(c,k) f_{v,k} )   # 多掩膜混合
    h_v ← h_v · λ_v(c)                        # 可见性加权

# L2 跨视角聚合
H ← normalize( mean{ h_v : κ̄_v > μκ − τκσκ } )   # ROFA
# 或 S_VCLS = top-k mean( s_v − α·μ_v )
# 或 φ̄ = Σ a_v h_v

# L3 一致性
κ   ← pairwise_mean_cos({h_v})
κ_c ← pairwise_mean_cos({h_v − μ_v})

# L4 判决
S_M ← top-|S0|( z(s) + λ z(id_top3) )
返回 S_M if S_M≠S0 and max(id_top3|S_M) > max(id_top3|S0) else S0
```

---

## 8. 假设与失效面

| 编号 | 假设 | 失效条件 | 后果 | 实测证据 |
|---|---|---|---|---|
| A1 | $\alpha_v$ 高保真 | 遮挡、弱纹理、掩膜边界 | L0 归属错，后续全部继承 | 0.85 闸门；覆盖率 found 0.907 / deep 0.954 / recall 0.112 |
| A2 | 视角间语义一致 | 同一区域在不同视角被不同掩膜覆盖 | ROFA 无效或误删 | 剔除率仅 **1.32%**（obs 236991 / removed 3123） |
| A3 | 公共背景可分离 | 场景共享语义强 | 一致性判据饱和 | 掩膜特征与场景均值余弦 0.69–0.76 |
| A4 | 配对前提成立 | 闭集单标签任务 | 证词无意义 | ScanNet 上 50.18→45.91 |
| A5 | 证人粒度匹配 | 区域非物体 | 判据自信地改坏 | ScanNet root 为跨墙粗区域；3D-OVS 5 次改判 4 次掉分 |

---

## 9. 性质汇总（可直接引用）

| 命题 | 陈述 | 用途 |
|---|---|---|
| C1 | L0 覆盖率是条件分布，$\bot$ 与"不属于"不可区分 | 解释 recall 族的上限 |
| C2′ | 单视角单证词是数据格式强加的，不是判据 | 解释为何多证据方法无法在现有 bank 上评估 |
| C2 | 有效样本量是视角数，不是观测数 | 反驳"证据越多越好" |
| C3 | 单视角内多掩膜合并不可逆 | 这是唯一的语义合并点 |
| C4 | 三种 L2 聚合的失效方向不同 | 消融必须逐种报 |
| C5 | $\max$ 与 $\mathrm{mean}$ 在 deep 族方向相反 | 解释 F1/F1b 的族级冲突 |
| C6 | 判决必须同查询配对 | 满足逐 query 独立协议 |
| C7 | 证词数 $<3$ 的候选弃权，而非退化 | 定义多视角判据的实际适用域 |
| C8 | $e$ 与覆盖数的相关（0.656）远超与正确性的相关（0.290） | 审计器增益部分来自尺寸先验 |
| 命题 1 | 未居中的一致性判据饱和 | 解释第一次 identity pilot 失败 |
| **主论断** | L1–L4 是 L0 的确定性函数 | **信息损失只可能在 L0** |

---

## 10. 对当前研究的直接含义

主论断给出一个可判定的问题分解：

$$
\underbrace{\text{L0 的离散化}}_{\text{唯一不可逆}}
\ \longrightarrow\
\underbrace{\text{L1–L4 的确定性变换}}_{\text{可逆性或可重算}}
$$

因此，"细粒度语义在哪里丢失"必须回答的是：**L0 的 $\alpha_v$ 是否已经丢掉了本来可分的二维信息？** 若 L0 可分而 L2/L3 塌缩，则问题在聚合规则，可用 $O(1)$ 的改动修正；若 L0 已经不可分，则必须改观测生成，任何聚合层的改进都无济于事。

这也解释了已有负结果的结构：ROFA 剔除率 1.32%（A2 成立）、居中一致性三轮预注册 FAIL（已在 L3 修正到位）、审计器用掉 88.6% 选项 oracle（L4 已饱和）。**四层中唯一未被诊断的就是 L0。**

---

## 11. 参数表（复现用）

| 参数 | 值 | 位置 |
|---|---|---|
| $\eta$ 归属闸门 | 0.85 | `merge_proj.py:224` |
| 编码维度 | 20 | `merge_proj.py:208` |
| 归属比例阈值 | 0.3 | `merge_proj.py:139,167` |
| $\tau_\kappa$ ROFA | 3.0 | `sai_nag_3.0.pt` |
| 可见性阈值 $\varepsilon$ | $10^{-4}$ | `merge_proj.py:138` |
| $\tau$ 覆盖率准入 | 0.3 / 0.4 / 0.5 | `lerf_identity_dump.py:213-236` |
| VCLS $k$ | 3 | `scannet_vcls.py:32` |
| VCLS $\alpha$ | 网格扫描 | `scannet_vcls.py:73` |
| $\lambda$ 融合 | 1.0 | `lerf_gate_trainingfree.py:12` |
| topk / level_until / filter_small | 5 / 3 / 50 | `eval_lerf_relags.py:4` |
| 二值化 | `semantics[0] > 0.5` | `eval_lerf_relags.py:103` |