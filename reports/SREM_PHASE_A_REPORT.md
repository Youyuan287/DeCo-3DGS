# SREM Phase A 报告（预注册 Stage 1，FAIL）

日期：2026-10-31
预注册：`analysis/PREREG_srem.md`
数据：LERF-OVS 四场景 208 查询；Phase A 逐视角观测 token（region observation）。

## 1. 实现（修正后）

- 场：每个候选的覆盖观测 token `X={f_o}`（query-agnostic）。
- 背景 `B`：**不同 root** 的其他候选 token（同 root 的 root/leaf 是同一物体，排除）。
- 相似度在**整个查询的所有候选 token 上**做全局 z-score（不是候选内 z-score；
  候选内 z 会破坏跨候选可比性，第一版因此崩到 found 0.19）。
- 贪心选 token：`base_o = z(<f_o,t>) − λ z(max_{b∈B}<f_o,b>)`，
  边际增益 `base_o − μ max_{j∈S}<f_o,f_j>`，`k≤8`，增益≤0 停止。
- 分数：选中 token 的 mean base − μ × 选中集合的平均两两余弦。
- 主设置 λ=1, μ=1；消融 (1,0) 和 (0,1)。

## 2. 结果（候选命中率，Stage 1）

| 信号 | found 124 | rank2 36 | deep 19 | recall 29 |
|---|---:|---:|---:|---:|
| id_top3（对照） | **109/124=0.879** | 10/36=0.278 | **14/19=0.737** | 7/29=0.241 |
| SREM λ1μ1 | 94/124=0.758 | 9/36=0.250 | 13/19=0.684 | 5/29=0.172 |
| SREM λ1μ0（去冗余） | 100/124=0.806 | 9/36=0.250 | 13/19=0.684 | 5/29=0.172 |
| SREM λ0μ1（去背景） | 95/124=0.766 | 10/36=0.278 | 14/19=0.737 | **9/29=0.310** |

自适应 k 平均约 **5.9**（figurines）。

## 3. 判定

预注册门控：SREM 需在 found 或 deep 上比 id_top3 高 ≥3 点，且 found 不降 >2 点。
实测 SREM 在 found 上降 **12.1** 点、deep 降 **5.3** 点；只有 λ0μ1 让 recall +6.9，
但 found 降 11.3。**Stage 1 FAIL。**

## 4. 为什么失败

1. **背景项惩罚了正确证据**：即使排除同 root，其他候选仍可能与正确物体语义相近
   （共享区域/同类兄弟），`max_b` 对正确 token 也高，λ=1 时把正确候选压下去。
2. **冗余项会换掉正确的 top token**：found 查询里 top-3 文本 token 通常已经对；
   冗余惩罚把其中一个换成“更不相似但更多样”的 token，分数下降。
3. **自适应 k≈6 > top-3**：把弱证据也纳入均值，稀释了强证据。
4. **Phase A token 是区域混合特征**：背景对比在混合特征上噪声很大；
   第一版候选内 z 的崩盘也说明这个信号对归一化方式极敏感。

## 5. 结论与下一步

- 在 Phase A 观测 token 上，**SREM 作为“背景对比 + 子集选择”没有超过简单的 top-3 late interaction**。
- 按预注册，不调 λ/μ；下一步应测试 **Phase B 的 object-level mask token**：
  背景对比在物体级特征上才可能有意义。
- 若 Phase B 仍失败，则“优雅核心”不应建立在背景对比上，应转向：
  class-agnostic mask graph proposal（解决 recall）+ 简单 late interaction 读出。

## 6. 产物

- 脚本：`analysis/lerf_srem_dump.py`（Phase A SREM dump）
- 预注册：`analysis/PREREG_srem.md`
- 结果：`analysis/results/20261031/srem/`（四场景逐候选 CSV）