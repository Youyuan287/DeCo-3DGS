# MULTIEVIDENCE_BUG_REPORT

按严重程度排序。每条都标注：代码位置 → 证据 → 影响 → 是否已数值确认。
`CONFIRMED` = 有实测数字支撑；`LATENT` = 代码可证但当前数据未触发；`EXCLUDED` = 已排除的原疑点。

---

## P0-1 CONFIRMED — 报出来的 0.67 % 不是认证评测的结果

**位置**：`ReLaGS/eval_ramen_evidence_prototype.py:44,46,49`；对照 `ReLaGS/utils/vlm_utils.py:50-62` 与 `ReLaGS/nag_data.py:103-123,134-188`

**事实**：脚本用 `vlm.get_text_encoding([prompt])`（纯 prompt 向量）算原始余弦，再 `np.argpartition(full,-5)[-5:]` 硬取 5 个 level-3 根节点。认证路径用的是

```
sims = [vlm.compute_similarity(f) for f in snag.feat]      # canon 对比 softmax, 温度 10
rel  = snag.search_matched_superpoint_in_mhtree(sims, topk, level_until=3,
                                                filter_small=50, remove_small=True)
```

即「canon 对比概率 + >50 gaussian 过滤 + top-10 内取 5 + level-2 子节点下探 + 按最大相似度间隙截断（通常只剩 1 个候选）」。

**证据**：仓库里已经存在同一读出规则（flat top-5）但用认证评分的对照 —— `results/protocolB.csv`（`n_candidates` 恒为 5，ramen 均值 **31.46**，`results/hf_only/ramen_summary.json`）vs 原型三臂的 **0.67 / 0.65 / 0.57**。同一份 checkpoint、同一批 level-3 根、同样的 top-5 数量，唯一变化是评分函数。

**影响**：`prototype_eval/summary.json` 的 0.0067 与 `results/ramen_summary.json` 的 0.5115 **不是同一测量的两端**；它们的差不能归因于多证据。绝对上限也可算清：71 条查询的 GT 面积占比均值仅 **0.02985**，即「整幅图全选」也只能得到 0.0299 —— 0.0067 是这个荒谬上限的 22 %。

---

## P0-2 CONFIRMED — 三次对比建立在三个不同的三维区域划分上（时间上最早的偏差点）

**位置**：`reproduce_ramem_multievidence.sh` 第 3 步（`merge_proj.py` 重算 NAG）；对照 `scripts/run_hf_only.sh:11`、`scripts/run_all.sh:14`

**事实**：同一条 merge_proj 命令行重跑得到的 level-3 分区数不同：

| 文件 | L1 | L2 | **L3** | 使用者 |
|------|----|----|--------|--------|
| `work/multievidence/ramen/sai_nag_3.0.pt` | 2517 | 587 | **304** | 本次多证据评测 |
| `ref/relags_hf/ramen/sai_nag_3.0.pt` | 1860 | 490 | **245** | 参考副本 |
| `ref/relags_hf/ramen/sai_nag.pt` | 1853 | 507 | **249** | `run_hf_only.sh` → 31.46 |
| `work/lerf/ramen/sai_nag_*.pt` | — | — | **未知（已删除）** | `run_all.sh` → 51.15 |

L0 三者一致（7936）；L1–L3 的联合单元数（3952 / 1147 / 665）说明**不是重新编号，是不同划分**。

**影响**：区域粒度、区域特征、候选池全部不同。任何跨这一层的 A/B 在物理上就不成立。这是**最早**出现的异常，且**与多证据模块无关**——它由第 3 步「重算而非复用参考 NAG」引入。

**最小修复**：第 3 步加 `--nag_path` 复用（或把 `merge_proj` 的聚类随机种子固定），并让所有对照臂指向**同一个** `sai_nag_3.0.pt`。

---

## P1-3 CONFIRMED — 方法自带的读端去中心化被 `alpha=0.0` 关闭

**位置**：`multi_evidence.py:153-173`，`alpha=0.0` 传入点 `eval_ramen_evidence_prototype.py:46`

**事实**：`score_bank_certified_style` 里

```python
ev  = l2n(bank.features);  ev  = ev  - alpha*center
base= l2n(bank.base_features); base = base - alpha*center
```

`alpha=0` 使 `center = base.mean(0)`（场景均值方向）完全不参与。**「残差特征 + 场景中心去除」这条设计在评测时是死代码。**

**佐证**：`cos(u, mean(nag_feat[2])) = 0.990734` —— 新方法估出的"共享方向"几乎就是生产侧场景均值方向，而生产的 `compute_similarity` 已经用 canon 对比消掉了这个方向。也就是说：**同一次去偏被设计了两遍，然后在评测里两遍都没生效**（canon 对比被换成原始余弦、`alpha` 被设为 0）。

---

## P1-4 CONFIRMED — 去偏残差被算出来、存下来、然后丢掉

**位置**：`multi_evidence.py:102`（存 `resid`）→ `multi_evidence.py:114-135,153-173`（从不读 `bank.residuals`）

**事实**：`mode="debiased"` 只改变**分组方式**（在残差空间里切），组值仍是原始特征的加权平均：

```python
p = l2n((a[mask,None] * ff[mask]).sum(0))     # 原始特征 ff
d = l2n((a[mask,None] * hh[mask]).sum(0))     # 残差 hh，只写进 resid
out[z, k] = p;  resid[z, k] = d
```

**证据**：`b3_same_scene`（debiased 分组）0.0057 vs `b2_raw`（raw 分组）0.0065 vs `k1` 0.0067 —— 三者几乎相同。分组方式不同（110/300 个区域组数变化）却没有带来任何差别，因为**打分用的向量在两种模式下本来就一样**（都是 `ff` 的加权平均），只有"怎么切"变了。

**这是本次诊断最重要的单点发现**：它同时解释了「为什么 b2 ≈ b3 ≈ k1」，也解释了「为什么方法不 work」——见 `METHOD_FAILURE_ANALYSIS.md`。

---

## P1-5 CONFIRMED — `k1_max_abs_error = 0.0` 被当成了等价性证明，但它只是自洽检查

**位置**：`analyze_ramen_observations.py:22-30`

**事实**：`direct` 与 `k1.base_features` 用**同一个** `l2n`、**同一套** `a = w/Σw`、**同一个** npz 顺序重算一遍，必然为 0。生产路径多一步 ROFA，而 ROFA 在观测导出**之后**才发生（`merge_proj.py:209` vs `:226`）。

**本次审计补做的真等价性检查（CPU，numpy 独立复现生产聚合）**：

| 对比 | cos 均值 | cos 最小 | >0.9999 的区域数 |
|------|----------|----------|------------------|
| 复现生产 vs `nag_feat[2]` | **0.999591** | 0.938409 | 296 / 300 |
| k1 bank vs `nag_feat[2]` | **0.998997** | 0.919666 | 272 / 300 |
| k1 vs 复现生产 | 0.999406 | 0.919666 | 276 / 300 |

**结论**：观测导出是忠实的，K=1 确实等价于生产聚合（ROFA 在本场景只剔除 0.9 % 观测）。所以「K=1 等价」这件事本身**成立**，只是原脚本的证明方式不成立。

---

## P2-6 LATENT — 观测 sink 静默丢弃非法观测

**位置**：`production_observation_adapter.py:12-13`

```python
if not np.isfinite(x).all() or not np.isfinite(w) or w <= 0 or np.linalg.norm(x) == 0:
    return
```

不计数、不告警、不留痕。一旦某场景出现 NaN 或零向量，观测会**静默少一条**，而 `region_ids` 仍覆盖 0..n-1，下游完全看不出。当前 ramen 数据未触发（11961 条全有限）。

**最小修复**：模块级 `dropped` 计数器 + `save()` 时写进 metadata。

---

## P2-7 LATENT — 评测器把两个分数空间拼进同一个排序

**位置**：`eval_ramen_evidence_prototype.py:47-48`

```python
full = (fallback @ l2n(text)[0])          # 生产特征的原余弦（304 个中 4 个只走这条）
for i, rid in enumerate(bank.region_ids):
    full[int(rid)] = score[i,0]           # 其余 300 个走 bank 分数
```

**当前无害**：`fallback` 用的是 `l2n(nag_feat[-1])`，与 bank 的 `l2n(features)` 同为余弦，量纲一致；4 个未观测区（id 0/199/286/302）的特征是全零向量，其分数为 0.0，在这些区域余弦分布（均值约 0.017）里排名靠后，不会进入 top-5。

**但如果有任何一个 bank 分数为负，这 4 个零分数区域就会挤掉真实候选**。原始余弦确实可以取负，所以这是**真实隐患**而不是纯理论问题。

**最小修复**：给这 4 个区域显式赋 `-inf`，并把 `full` 换成 `np.full(-inf)` 后只填 bank 覆盖的位。

---

## P2-8 CONFIRMED — 方法先验/温度把"多证据聚合"变成近硬 max，但没有退化到均匀

**位置**：`multi_evidence.py:118-133`，`EvidenceConfig.temperature = 0.08`

**实测（b2 真实 priors，300 个随机文本方向）**：

| 指标 | 实测 | 均匀加权(2.15) | 硬 max |
|------|------|----------------|--------|
| softmax 熵 | **0.6344 nats** | 0.767 | 0 |
| 最大权重 | **0.7211** | 0.464 | 1.0 |
| max-prob > 0.99 的比例 | **0.000** | — | — |

**结论**：温度 0.08 让权重偏尖锐，但**没有**退化成「只取一个证据」；同时 `E[max_k cos]` 随 K 增大带来的候选偏置只有 **+0.0008**（K=1 区 0.0164 vs K=3 区 0.0172），相对余弦本身的跨区离散度（std ≈ 0.043）可忽略。

→ 任务书里「温度导致退化为均匀」和「有效证据数造成候选偏置」两条**均降级为不成立**，但**温度确实使方法行为与本意（多条证据加权融合）不符**，属设计问题而非实现 bug。

---

## P2-9 CONFIRMED — 脚本目录名与既有产物不一致，直跑会重算而非复用

**位置**：`reproduce_ramen_multievidence.sh` 的 `RUN=.../work/multievidence_manual/ramen`、`OBS=.../observations_manual/ramen`

**事实**：磁盘上存在的是 `work/multievidence/ramen` 与 `observations/ramen`（2026-10-27 16:47–17:06）；`*_manual` 目录**不存在**。脚本还会在 `$RUN` / `$OBS` 已存在时直接拒绝执行。

→ 现在重跑会得到一份**新的** NAG（又是新分区）和新的观测，而不是复现已有数字。这使 0.0067 这个结果**本身不可复现**。

---

## P3-10 CONFIRMED — shell 脚本没有对照臂，也没有调用认证评测器

`grep -c eval_lerf_relags reproduce_ramen_multievidence.sh` → 0。7 个 stage 里没有一步产出「原样 ReLaGS 在本场景的分数」，所以脚本自带的三个臂**缺少分母**，任何「变好/变差」的判断都无参照。

---

## EXCLUDED（已排除，附数值依据）

| 疑点 | 排除依据 |
|------|----------|
| 可见性权重被平方乘入 | 独立复现 vs `nag_feat[2]` cos 均值 0.999591，296/300 > 0.9999 |
| `weight.npy` 二值 1.0 被当连续权重 | 导出权重 3.669e-05–1.0 连续 |
| `RATIO_THRESHOLD=0.3` 被当 ROFA 阈值 | 仅用于 `merge_proj.py:183` 的 mask×superpoint 重叠比 |
| 2D mask 特征归一化与原始不同 | 同一函数同一行，未分叉 |
| 同视角多 mask 未按覆盖比例混合 | `sp_feat = sp_mask_mat @ view_level_feature`，`--mix` 默认 soft |
| `visibility_portion` 计算错误 | 导出即生产用的同一个值，复现 cos 0.9996 |
| 观测↔区域↔视角↔mask 对应错位 | `region_ids` 稀疏原始 id，与生产可见集完全相等；`view_ids` 0–130 |
| ROFA 用了错误 tau | `--tau 3`，即生产 tau=3；实测只剔 0.9 % |
| 新方法保留了 ROFA 删除后的观测 | 导出于 `merge_proj.py:209`，ROFA 在 `:226` → 保留删除前 |
| 共享方向与特征不同空间 | `u` 由同一批特征算出，`u_norm=0.99999994` |
| 在已单位化特征上重复中心化 | `nf=l2n(f)`；`h=l2n(nf-(nf·u)u)`，一次投影一次归一化 |
| 所有证据被投影成近零向量 | 残差未参与打分（这本身是 P1-4），但 `h` 的范数非零，分组有效（组内 cos 0.754） |
| 训练/推理去偏方式不同 | 无训练阶段；`u` 只有一条计算路径 |
| 原始特征已去偏又被二次去偏 | `alpha=0.0` 关掉了读端去偏；`u` 只进分组不进打分 |
| softmax 沿错误维度 | `axis=1` 即 K 轴，正确 |
| `[N,K,C]` 归约维度错 | `score=(prob*raw).sum(axis=1)` → `[R,T]`，与 `[R,K,T]` 匹配 |
| 不同区域共用同一套证据权重 | `raw=x@t.T` 逐行独立，无广播 |
| 查询后再次场景中心化 | `alpha=0.0`，不存在二次中心化 |
| 全集用了 ScanNet 的 aggregate.py | 未使用；GT/IoU/mIoU 聚合与认证逐行一致 |
| 多证据是伪多证据（K 条相同） | 组内两两 cos 0.7540 |
| 只保存单个噪声观测 | 每族均为加权平均 |
| 区域索引 / Gaussian 标签错位 | `isin(labels, chosen)`，`labels` 长度 372270 = gaussian 数 |
| 图像分辨率不一致 | `cam.image_height/width`，与 GT jpg 同尺寸 731×988 |
| 共享方向让排序对 prompt 不敏感 | 随机方向间 Spearman 0.004/0.029，排序高度 prompt 敏感 |
| 有效证据数造成候选偏置 | `E[max_k cos]` 差 +0.0008，远小于跨区 std 0.043 |
| 查询 softmax 退化为均匀 | 熵 0.6344 < 均匀 0.767，最大权重 0.7211 > 0.464 |

---

## 严重度汇总

- **P0（使结论无效）2 条**：P0-1 评测器不同、P0-2 分区不同
- **P1（使方法失效）3 条**：P1-3 `alpha=0`、P1-4 残差被丢弃、P1-5 等价性证明无效
- **P2（隐患/设计偏差）4 条**：P2-6 静默丢弃、P2-7 分数空间拼接、P2-8 温度近硬 max、P2-9 不可复现目录
- **P3（工程完整性）1 条**：P3-10 无对照臂

**没有任何一条确认的实现 bug 位于"多证据表示"本身**（观测导出、加权、归一化、分组、索引全部通过数值校验）。失效点在**读端**。