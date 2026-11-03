# PATCH_PROPOSAL

**状态：仅提案，未应用。** 未修改 `/root/rivermind-data/ReLaGS_Reproduction` 下任何文件。
所有补丁都标了「是否需要 GPU」。带 `[NO-GPU]` 的可立即验证；带 `[GPU]` 的等显卡回来。

---

## P0-A `[NO-GPU]` 让对照组与实验组用同一份 NAG

**位置**：`reproduce_ramen_multievidence.sh`，`[7/7]` 之前新增 `[7/8]`。

**问题**：现在 0.0067（304 区）与 0.3146（249 区）/0.5115（未知）建立在不同分区上，不可相减。

**最小修改**（不是让 NAG 变确定，而是**在同一份 NAG 上重算基线**）：

```bash
# [7/8] baseline arm on the SAME nag as the new method
$PY -u scripts/eval_lerf_relags.py \
  --scene ramen \
  --source "$RUN/input/ramen" \
  --model "$RUN/ramen" \
  --nag "$RUN/ramen/sai_nag_3.0.pt" \
  --gt_root "$GT" \
  --out "$OBS/baseline_eval" \
  --csv "$OBS/baseline_eval/per_query.csv" \
  --topk 5 --level_until 3 --filter_small 50
```

**为什么这是最小的**：不需要修 `merge_proj.py` 的随机性，只需要一次额外评测，就能把「51.15 → x」的比较变成「同分区 A/B」。这是本次诊断中**性价比最高的一处改动**。

**前置**：`scripts/eval_lerf_relags.py:40` 的 `--csv` 是必填；`--out` 是目录，会写 `<scene>_summary.json`。

---

## P0-B `[GPU]` 用认证读出评测多证据 bank

**位置**：新增 `ReLaGS/eval_multievidence_certified.py`（不改任何现有文件）。

**问题**：`eval_ramen_evidence_prototype.py:44,46,49` 用「原始 prompt 余弦 + 硬 top-5」替代了「canon 对比概率 + 过滤 + level-2 下探 + 间隙截断」。实测换上原始余弦后 top-5 只有 **1.93/5** 重合，且系统性选大区（+16 % gaussian 质量，`hand` +10×）。

**最小修改**：把 bank 里的**每一条证据**当作一个区域特征，走认证的 `compute_similarity`，再在 K 轴上做**同一个** softmax，得到单一 `sims` 向量，喂给认证检索：

```python
# 1) canon 打分：对每条证据单独算认证分数（不是原始余弦）
vlm.encode_text(prompt)                       # → text_feature (5,512)
R, K, C = bank.features.shape
ev = bank.features.reshape(R * K, C)          # 展平，复用生产函数
flat = torch.from_numpy(ev.astype(np.float32)).cuda()
cs   = vlm.compute_similarity(flat).reshape(R, K).cpu().numpy()   # (R,K) ∈(0,1)

# 2) K 轴 softmax，与生产同温度
lo = cs / cfg.temperature + np.log(np.maximum(bank.priors, 1e-12)) * bank.valid
lo = np.where(bank.valid, lo, -1e9)
e  = np.exp(lo - lo.max(1, keepdims=True)); w = e / e.sum(1, keepdims=True)
sim_l3 = (w * cs).sum(1)                      # (R,) —— 与 nag_feat[2] 同量纲

# 3) 填回 304 个根，其余 -inf（不要用 fallback 混排，见 P2-A）
sims_full = np.full(len(cnt), -np.inf, np.float32)
sims_full[bank.region_ids] = sim_l3

# 4) 走认证检索器（filter_small / level-2 下探 / 间隙截断全部保留）
snag.sim_at_levels 由 [sim_l1, sim_l2, sims_full] 组成
rel, cands = snag.search_matched_superpoint_in_mhtree(
    [sim_l1, sim_l2, torch.from_numpy(sims_full).cuda()], topk=5, level_until=3,
    filter_small=50, remove_small=True)
```

**注意**：`sim_l1` / `sim_l2` 仍需用**未修改**的 `nag_feat[0]` / `nag_feat[1]` 算（多证据只改了 level 3），否则 level-2 下探的判据也被污染，就不是等价对照了。

**这一步做完，`K=1` 与 `K=3` 的差异才是「多证据」的净效应。**

---

## P1-A `[GPU]` 把去偏残差真正接进打分（方法层，非修复）

**位置**：`multi_evidence.py:114-135` / `:153-173`。

**问题**：`residuals` 被计算、保存、从不读取（`multi_evidence.py:102` 写，评分只读 `features`）。这是「多证据」与「单证据」在数学上等价的直接原因（组值是同一批观测的凸组合）。

**最小修改**（二选一，不改结构）：

```python
# 方案 1：混合，不是替换 —— 保留主体语义，再叠加残差方向
x = _l2n(_l2n(bank.features) + beta * _l2n(bank.residuals))   # beta=0 时退回现状
# 方案 2：只在残差上做 K 轴重排，特征本身仍是 base
#         w 由残差的匹配分决定，但 score 用 base —— 目前就是这个，无效，不推荐
```

`beta` 的取值不要扫参，先跑 `beta ∈ {0, 0.5, 1.0}` 三点看**方向**。若三点都不改善，说明残差这条路本身不成立，应直接放弃该设计，而不是继续调。

---

## P2-A `[NO-GPU]` 修掉评测器里两个分数空间的混排

**位置**：`eval_ramen_evidence_prototype.py:47-49`

```python
# 现在
full = (fallback @ l2n(text)[0]).astype(np.float32)
for i, rid in enumerate(bank.region_ids): full[int(rid)] = score[i,0]
chosen = np.argpartition(full, -5)[-5:]

# 改为
full = np.full(fallback.shape[0], -np.inf, dtype=np.float32)
full[bank.region_ids] = score[:, 0]
chosen = np.argpartition(full, -5)[-5:]
```

**理由**：当前 4 个从未被观测的区域（id 0/199/286/302）拿到 `0.0` 分。今天无害（真实余弦多为正，它们排在后面），但**原始余弦可以取负**，一旦出现负分，这 4 个零分区域就会挤掉真实候选。这是 latent bug，不是理论洁癖。

---

## P2-B `[NO-GPU]` 给静默丢弃加计数器

**位置**：`production_observation_adapter.py:12-13`、`save()` (`:24-40`)

```python
class ProductionObservationSink:
    def __init__(self):
        ...
        self.dropped = 0
        self.drop_reasons = {"nonfinite_feat":0, "nonfinite_w":0, "w_nonpos":0, "zero_vec":0}
    def add(self, ...):
        if not np.isfinite(x).all():  self.dropped += 1; self.drop_reasons["nonfinite_feat"] += 1; return
        if not np.isfinite(w):        self.dropped += 1; self.drop_reasons["nonfinite_w"] += 1; return
        if w <= 0:                    self.dropped += 1; self.drop_reasons["w_nonpos"] += 1; return
        if np.linalg.norm(x) == 0:    self.dropped += 1; self.drop_reasons["zero_vec"] += 1; return
        ...
    def save(self, path):
        ...
        np.savez(path, ..., dropped=self.dropped, **{f"drop_{k}":v for k,v in self.drop_reasons.items()})
```

**理由**：现在观测少一条，下游完全看不出来，`region_ids` 仍然连续。换场景（figurines 已有导出）时这是不可见的数据损失。

---

## P2-C `[NO-GPU]` 把 `alpha` 变成显式实验变量

**位置**：`eval_ramen_evidence_prototype.py:46`

```python
score, base = score_bank_certified_style(bank, text, alpha=0.0)
```

**理由**：`alpha=0.0` 使 `center = base.mean(0)` 完全不参与（`multi_evidence.py:161-166`），方法自带的读端去偏是死代码。而 `cos(u, mean(nag_feat[2])) = 0.990734` 说明它想去掉的方向与生产 `compute_similarity` 已去掉的是同一个。

**最小修改**：`alpha` 提成命令行参数，默认 0（保持现状不变），并在 `summary.json` 里记录实际值，避免「以为开了实际没开」。**不要**在没有 P0-B 之前扫 `alpha`。

---

## P3-A `[NO-GPU]` 让脚本可复现

**位置**：`reproduce_ramen_multievidence.sh`

三处：
1. `RUN=work/multievidence_manual/ramen` / `OBS=observations_manual/ramen` 与磁盘上既有的 `work/multievidence/ramen` / `observations/ramen` 不一致 → 直接重跑会得到一份**新** NAG，0.0067 不可复现。改成与既有产物同名，或加 `--force` 并显式删除。
2. 脚本拒绝在 `$RUN`/`$OBS` 已存在时运行，但没有 `--force` 出口。
3. `[1/7]` 的文件名一致性检查只 `exit 1`，未打印首个不匹配项，排查困难。

---

## 应用顺序（以及为什么不能跳步）

```
P0-A  [无 GPU]  → 立刻可做，先把比较基础弄对
P2-A  [无 GPU]  → 与 P0-A 同批，消除 latent 风险
P2-B  [无 GPU]  → 一并
      ↓
P0-B  [需要 GPU] → 唯一能回答「多证据是否有增益」的实验
      ↓
若 P0-B 显示 K=3 ≈ K=1：
P1-A  [需要 GPU] → 三点 beta 试探残差路线；仍无改善则放弃该设计
若 P0-B 显示 K=3 > K=1：
      → 现有实现可以保留，问题只在评测器（P0-1）
```

---

## 不提案的改动

- 不改 `merge_proj.py` 的 `RATIO_THRESHOLD`、`WEIGHT_THRESHOLD`、`_mix`、`tau`：已确认均未误用，且 ROFA 在本场景只剔 0.9 % 观测。
- 不改 `nag_data.py` 的检索规则：它是认证协议本身，改了就没有可比基线。
- 不扫 `temperature` / `max_evidence` / `min_group_weight`：见 `METHOD_FAILURE_ANALYSIS.md` H5/H6，这三者不是瓶颈（熵 0.6344 未退化，max_evidence 只是上界，min_group_weight 只影响 5 % 质量的组）。
- 不新增任何大模块。