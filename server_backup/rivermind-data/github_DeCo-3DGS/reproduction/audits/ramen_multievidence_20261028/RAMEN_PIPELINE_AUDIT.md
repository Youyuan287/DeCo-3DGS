# RAMEN_PIPELINE_AUDIT

**审计对象**：`/root/rivermind-data/ReLaGS_Reproduction/reproduce_ramen_multievidence.sh`
**场景**：LERF-OVS / ramen（731×988，131 训练视角，372270 个 Gaussian，71 条查询，7 张 GT 图）
**结论状态**：只读诊断，未修改任何仓库文件。**服务器当前无 GPU**（`/dev/nvidia*` 不存在，`Scene()` 抛 `RuntimeError: Found no NVIDIA driver on your system`），因此所有渲染类验证改为 CPU 侧数值复算 + 代码路径核对。

---

## 0. 一句话结论

> 观测导出链是**忠实**的（可逐区域复现生产特征，cos 均值 0.9996），多证据特征本身也**几乎等同于** ReLaGS 特征（cos 0.9990）；但**没有任何一步把多证据分数送进认证评测器**。报告出来的 0.67 % 由 `eval_ramen_evidence_prototype.py` 自己的「原始 prompt 余弦 + 硬 top-5」读出规则产生，与 51.15 % 的认证路径**不是同一个评测**，两者相差的 50 个点里，约 19.7 点来自候选选择规则、约 30.8 点来自评分函数，多证据表示贡献 ≈ 0。

---

## 1. 脚本真实执行链（逐步核对，非假设）

`reproduce_ramen_multievidence.sh`，101 行，7 个 stage。实际执行顺序如下。

| # | 行号 | 实际动作 | 是否碰模型/GPU | 与原始 ReLaGS 是否一致 |
|---|------|----------|----------------|------------------------|
| 1 | `[1/7]` | 校验 `$SRC0/images` 文件名 ↔ `$FEAT/*_s.npy` 一一对应，不匹配即 `exit 1` | 否 | 新增（合理性检查） |
| 2 | `[2/7]` | `$SRC0` → `$RUN/input/ramen`；软链 `language_features`；`cp -a $MODEL0 $RUN/ramen` | 否 | 一致（`MODEL0=ref/relags_hf/ramen`，与 `scripts/run_official_scene.sh` 相同） |
| 3 | `[3/7]` | `RELAGS_OBSERVATION_OUT=... $PY -u merge_proj.py -s ... -m ... --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0` | **是（唯一 GPU 步骤）** | 参数一致，但**重算 NAG**（见 §1.1） |
| 4 | `[4/7]` | `analyze_ramen_observations.py` obs_l3 → b2_raw_l3 + `structural_check_level3.json` | 否 | 新增 |
| 5 | `[5/7]` | `prototype_b3_ramen.py` obs_l3 → b3_debiased + `b3_same_scene_prototype_level3.json` | 否 | 新增 |
| 6 | `[6/7]` | `eval_ramen_evidence_prototype.py` obs b2 b3 … → `prototype_eval/{summary.json,per_query.csv}` | **是（渲染）** | **全新评测器，非认证评测器** |
| 7 | `[7/7]` | 打印文件清单与 `summary.json` | 否 | — |

**关键缺失**：脚本**没有**任何一步调用 `scripts/eval_lerf_relags.py`（认证评测器），也**没有**一个「原样 ReLaGS（单证据 + 认证读出）」的对照臂。因此 `prototype_eval/summary.json` 里的数字**唯一可比的只有它自己的三个臂**，不能与 `results/ramen_summary.json` 的 0.5115 相减。

### 1.1 NAG 被重算，且重算结果不可复现（已确认）

`merge_proj.py` 在每个 stage 3 迭代里重新做图聚类（`assign_label`，merge_proj.py:324-407）。同一条命令（`--thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3`）在不同时间跑出的 level-3 分区数量不同：

| 文件 | L0 | L1 | L2 | **L3** |
|------|----|----|----|--------|
| `work/multievidence/ramen/sai_nag_3.0.pt`（本次多证据运行） | 7936 | 2517 | 587 | **304** |
| `ref/relags_hf/ramen/sai_nag_3.0.pt`（参考副本） | 7936 | 1860 | 490 | **245** |
| `ref/relags_hf/ramen/sai_nag.pt`（`run_hf_only.sh:11` 使用 → 31.46 那条） | 7936 | 1853 | 507 | **249** |

L0 三者完全相同（7936），L1–L3 互不相同；四层 label 长度均为 372270。联合单元数分别为 3952 / — / 18949，说明差别不是「同一分区的重新编号」，而是**真实的不同划分**。

**后果**：0.0067（304 区）、0.3146（249 区）、0.5115（第三份，`work/lerf` 已被删除，无法确认）**分别建立在三个不同的三维区域划分上**。这在物理上就不允许直接相减。这是本次审计发现的**第一个、也是时间上最早的**偏差点：它发生在 shell 脚本第 3 步，而不是任何新算法模块里。

---

## 2. 逐阶段核对（预期流程 vs 实际代码）

### 2.1 原始 RGB / SAM mask → CLIP 二维特征
- 特征来源：`/root/rivermind-data/lerf_oseg/build/ramen_run/language_features/*_s.npy`，共 131 个视角，每视角 `(n_masks,512)`，由 `merge_proj.py` 的 `extract_gaussian_features(..., level)` 读出。
- **未重新生成**，复用原始缓存；shell 脚本只做文件名一致性校验（step 1）。
- 二维特征归一化：`merge_proj.py` 中 `view_level_feature` 在使用前由 `proj_gaussian_features` 内部处理，与认证路径同一函数、同一代码 → **一致，无偏差**。

### 2.2 2D→3D 区域对应
- `proj_gaussian_features`（merge_proj.py:145-232）：`weight = render_pkg["weight"]`；`gau_mask = weight > WEIGHT_THRESHOLD`（`WEIGHT_THRESHOLD = 0.0001`）。
- `RATIO_THRESHOLD = 0.3` **只**用于 merge_proj.py:183 `sp_mask_mat[sp_mask_mat < RATIO_THRESHOLD] = 0`（superpoint × SAM-mask 重叠比），**没有**被误用作 ROFA 阈值。→ 任务书里怀疑的该项 **排除**。
- `_mix` 默认 `soft`：`sp_feat = sp_mask_mat @ view_level_feature`（merge_proj.py:197），即按真实覆盖比例混合；`--mix` 未传 → soft。**同一视角内多 mask 按覆盖比例混合，正确**。

### 2.3 多视角观测生成 — `production_observation_adapter.py`
- `add()`（adapter:11-23）在 `merge_proj.py:209-214` 被调用，位置在**每个视角的循环内**，即在 `F.normalize(...) * visibility_portion`（merge_proj.py:215）**之前**、在 ROFA（merge_proj.py:220-228）**之前**。
- 因此导出的 `sp_feat` 是：**未 L2 归一化、未乘 visibility_portion** 的原始混合向量；`visibility_portion` 作为独立列导出。
- **权重是否被平方乘入？——否，已用数值反证**（见 §3）。生产侧 `w` 只乘一次（merge_proj.py:215），`multi_evidence.build_bank` 的 `a = w/Σw` 是唯一一次加权。cos 复算 0.9996 直接排除该假设。
- **静默丢弃**：`add()` 在 `not np.isfinite(x).all()` / `not np.isfinite(w)` / `w<=0` / `np.linalg.norm(x)==0` 时直接 `return`（adapter:12-13），**不计数、不告警**。这是本次审计确认的 silent fallback（P2），当前数据下未触发（观测数 11961 均为有限值），但下一次换场景时不可见。
- 实际导出规模（`observations_level3.npz`）：`features (11961,512) float32` 行范数 0.3025–1.0004（均值 0.8837，说明确实未归一化）；`weights (11961,)` 3.669e-05–1.0（**连续可见性比例，不是二值 1.0**，与任务书的猜测相反）；`region_ids (11961,)` 1–303；`view_ids` 0–130；`member_gaussian_ids` 31.7 M 项、`visible_gaussian_ids` 14.6 M 项。

### 2.4 ROFA
- `_filter_feature_outliers`（merge_proj.py:97-143）：**per region、跨全部视角**（`sp_feature_list[sp]` 在视角循环里累加，merge_proj.py:217），ROFA 循环在视角循环之后。
- `keep_mask = mean_sim > (mu - tau*sigma)`（merge_proj.py:125），`tau = self.args.tau = 3`。空集 → 保留 argmax（:127-128）。
- **实测：tau=3 只剔除 0.9 % 的观测**（每个多观测区域平均保留 40.69 / 41.1）。→ ROFA 在本场景近乎无效，这解释了为什么「K=1 复现的是 ROFA 前聚合」这件事影响极小（cos 0.998997）。
- 新方法**保留的是 ROFA 删除前**的观测（导出于 merge_proj.py:209，ROFA 在 :226）→ 任务书要求确认的「保留删除前还是删除后」= **删除前**，已确认。

### 2.5 共享方向估计与去偏
- `u = l2n(l2n(f).mean(0))`（`prototype_b3_ramen.py:14`、`eval_ramen_evidence_prototype.py:30`）：**来自当前场景全部 11961 个观测**，不是三维区域特征、不是二维 CLIP 特征、更不是跨场景预计算。
- 空间一致性：`u` 由同一批特征算出，与特征同空间 ✔（`u_norm = 0.99999994`）。
- **但它与生产侧的「场景均值方向」几乎重合**：`cos(u, mean(nag_feat[2])) = 0.990734`。
- 去偏公式（`multi_evidence.py:66-67`）：`nf = l2n(f)`；`h = l2n(nf - (nf·u)u)`。即在**已单位化**特征上做一次投影再归一化，没有重复 L2 中心化、没有 shape 错误、没有把 `u` 当均值用。退化行（`|h| < 1e-6`）回落到 `nf`（:69-71）——合理的防御。
- **致命实事**：`h`（残差）只用于**决定怎么分组**，分组后的证据向量 `p` 用**原始 `ff`** 加权平均，`residuals` 被存进 npz 后**从未被任何评分函数读取**（`score_bank` / `score_bank_certified_style` 只读 `bank.features`）。→ 「原始特征保留完整语义、残差仅组织模式」的设计**在半途被丢弃**：残差既没进最终特征，也没进权重。

### 2.6 多语义证据构建 — `multi_evidence.py:52-111`
- 每区 `idx = where(r==rid & w>0)`；`a = w/Σw`；`base = l2n(Σ a·ff)`。
- `K=1` 或 `len(idx)==1` → 直接返回 `base`（:84-86）。
- `K>1`：在**残差空间** `hh` 做加权 medoid 播种 + 最远优先贪心切组，`sim = hh@hh.T`，`assign = argmax(sim[:, chosen])`（:88-95）；组质量 `mass = Σ a`，`mass < 0.05` 丢弃（:100-101）。
- **实测分布**：`group_count` = {1:67, 2:120, 3:113}；组内两两 cos 均值 **0.7540**；`cos(group, base)` 均值 0.9030。
  → **不是伪多证据**（K 条向量确实不同，相差约 41°），任务书怀疑的「多个证据完全相同」**排除**。
  → 但也**不是**「每组只存一个噪声观测」：组值都是加权平均。
- **K=1 是否严格退化为原始加权聚合？** 在 `build_bank` 内部**是**（`analyze_ramen_observations.py` 的 `k1_max_abs_error=0.0` 证明了这一点）。但那个断言只是**自洽检查**（用同一个 `l2n`、同一套权重、同一个循环顺序重算一遍），**不是**与生产聚合的等价性证明，因为生产多了 ROFA 一步。真正的等价性由本次审计补做，见 §3。

### 2.7 查询条件评分 — 这是问题真正所在
生产路径（`utils/vlm_utils.py`，`ClipSimMeasure`）：
```
encode_text(prompt)   →  tokenize([prompt, "object","things","stuff","texture"]) → L2 归一化 → (5,512)
compute_similarity(f) →  logit = f @ T^T                          # (R,5)
                         pos = logit[:,0:1], neg = logit[:,1:]
                         sims = stack((pos.repeat(1,4), neg), -1)  # (R,4,2)
                         p = softmax(10.0 * sims, -1)
                         best = p[...,0].argmin(1)                 # 与该区域最"像"的 canon 词
                         score = p[best][:,0,0]                    # ∈ (0,1)
```
即：**region 相对「最易混淆的通用概念」的胜出概率**，温度 10。这本身**已经是一次规范方向去偏**。

新路径（`eval_ramen_evidence_prototype.py`）：
```
line 44  text = vlm.get_text_encoding([prompt])          # 纯 prompt 向量，无 canon、无 softmax
line 46  score,_ = score_bank_certified_style(bank, text, alpha=0.0)
line 47  full = fallback @ l2n(text)[0]                  # 生产特征的原始余弦
line 48  for i, rid in enumerate(bank.region_ids): full[int(rid)] = score[i,0]
line 49  chosen = argpartition(full, -5)[-5:]
```
四重偏离：
1. **换了评分函数**（`get_text_encoding` 替代 `compute_similarity`）：丢掉 canon 对比，分数从「胜出概率」退化为**无界原始余弦**。
2. **`alpha=0.0`**（`multi_evidence.py:161-166`）：`ev = ev - alpha*center` 直接消失 → 方法自带的读端去中心化**被关闭**。
3. **换了候选规则**（`argpartition(-5)`）：硬编码 top-5、无 `filter_small=50`、无 level-2 下探、无相似度间隙截断。
4. **混用两个来源**（line 47/48）：4 个从未被观测的区域（id 0/199/286/302）用 fallback 的**原始余弦 0.0** 参与排序，其余 300 个用 bank 分数。数值上无害（见 §4），但两个分数空间的拼接是设计缺陷。

`score_bank_certified_style` 内部数学本身没有广播错误：`raw = x @ t.T` 为 `[R,K,T]`；`logits = raw/0.08 + log(prior)`；softmax 沿 **axis=1（K 轴，正确）**；无效槽置 `-1e9`；`score = (prob*raw).sum(1)`。**不同区域的证据权重是各自独立的**（`raw` 每行独立），不存在「所有区域共用一套权重」的广播错误。

### 2.8 三维区域 → Gaussian → 渲染
- `labels = nag["nag"][-1]`（304 个 level-3 id，372270 个 gaussian）；`rel = isin(labels, chosen)`；`gaussians._semantics = rel.expand(-1,20)`；`sem = render(...)["semantics"]`；`pred = sem > 0.5`。
- 与认证评测器 `scripts/eval_lerf_relags.py:85-95` 的渲染段**逐行等价**（同一 `Scene`/`render`/阈值 0.5，同一 `cam.image_height/width`）。
- `cam` 取自 `cams = {c.image_name: c for c in scene.getTrainCameras()}`（eval:35，与认证 :69 相同），`name` 取自 GT `.jpg` 词干 → **视角 id 与图像 id 对齐正确**。
- `bank.region_ids` 是**稀疏原始 id**（`[1..303] \ {0,199,286,302]`），`full[int(rid)]` 因此正确对齐，**不存在区域索引错位**（这条已被数值确认，见 §4）。

### 2.9 LERF-OVS GT 与 IoU
- GT：`cv2.fillPoly` 联合 `obj['category']==prompt` 的所有多边形（eval:54-56，与认证 :96-99 一致）。
- IoU = `inter/union`，`union==0` 时 NaN（eval:57，与认证 :100-102 一致）。71 条查询无 NaN。
- **mIoU 聚合方式一致**：先每图取均值，再对 7 张图取均值（eval:59-60，认证 :113-117）。
- **但没有套用 ScanNet 的 `aggregate.py`**，也没有套用任何错误脚本 ✔。
- **口径缺失**：新评测器**完全没有** `filter_small` 与 `remove_small`。53 % 的 level-3 区域（161/304）「不是最小 5 个」……准确说：304 个 level-3 区中有 70 个 gaussian 数 ≤ 50，认证路径会把它们从 top-10 里剔掉，新评测器不会。这不是「阈值口径改变」，而是**规则缺失**。

---

## 3. 数据来源与「是否使用原始缓存」

| 项 | 路径 | 是否重新生成 |
|----|------|--------------|
| 图像/COLMAP | `VALA/dataset/3dgs/lerf_ovs/ramen/{images,sparse}` | 复用 |
| CLIP 2D 特征 | `lerf_oseg/build/ramen_run/language_features/*_s.npy` | 复用 |
| 3DGS checkpoint | `ref/relags_hf/ramen/`（作者权重，`point_cloud/iteration_0`） | 复制，未重训 |
| NAG | `$RUN/ramen/sai_nag_3.0.pt` | **重新生成**（非确定性，见 §1.1） |
| GT | `ref/hf_lerf/lerf_ovs/label/ramen` | 复用 |
| 观测 | `observations/ramen/observations_level{1,2,3}.npz`（271/126/106 MB） | 生成 |

**没有**使用 ScanNet 的文本库、类别设置或 IoU 口径；**没有**改变 mask id / 视角 id / gaussian 映射（`member_gaussian_ids` / `visible_gaussian_ids` 的 max 均为 372269，与 `labels` 长度一致）。

---

## 4. 已排除的怀疑项（有数值证据）

1. **权重平方乘入 — 排除**。用 npz 按 `x_i = l2n(f_i)·w_i` 重算生产聚合，再与 `nag_feat[2]` 比：
   `cos = 0.999591 (mean) / 0.938409 (min)`，**296/300 区 > 0.9999，0 区 < 0.9**。若权重被平方，这个数会显著下降。
2. **`weight.npy` 二值 1.0 冒充连续可见性 — 排除**。导出权重实测 3.669e-05 … 1.0 连续分布。
3. **`RATIO_THRESHOLD=0.3` 被误用作 ROFA 阈值 — 排除**（merge_proj.py:183 只用于重叠比）。
4. **区域索引错位 — 排除**。`b2["region_ids"]` 是稀疏原始 id 且与生产可见 id 集合完全相等。
5. **共享方向让排序与 prompt 无关 — 排除**。两个随机文本方向之间的排序 Spearman 仅 0.004（生产特征）/0.029（k1）→ 排序对 prompt 高度敏感；且 `var(f·m̂)=0.001844` ≈ 随机方向 0.001947，说明共享方向是**近似常数偏移**，不主导排序。
6. **多证据是伪多证据（K 条相同）— 排除**。组内两两 cos 0.7540。
7. **`__semantics` 渲染/阈值/GT/IoU 与认证不一致 — 排除**（§2.8–2.9 逐行等价）。

---

## 5. 未决 / 被阻塞

- **T5 的最终确认**需要一次 GPU 渲染：在同一份 `sai_nag` 上同时跑「认证读出」与「原型读出」两个臂。`/tmp/diag_arms.py` 已写好，等 GPU。
- 51.15 那一次用的 NAG 已被删除（`work/lerf/` 不存在），无法确认它对应 245 还是 249 个 level-3 区。
- `production_observation_adapter.add()` 的静默丢弃没有计数，无法从产物反推被丢掉的观测条数。