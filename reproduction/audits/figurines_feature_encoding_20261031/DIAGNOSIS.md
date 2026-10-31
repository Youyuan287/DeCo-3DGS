# 诊断报告：LERF-OVS 语义场重建为什么只有 58.11（figurines）

日期：2026-10-31
范围：LERF-OVS / figurines 的 `sai_nag_3.0.pt` 自建重建
结论一句话：**不是算法问题、不是种子问题、不是 DeCo 补丁问题，是"读到的特征文件根本不是我们用 ReLaGS 编码器编出来的那一份"**。而让它三轮排查都看不见的原因，是一个符号链接造成的路径穿透。

---

## 0. 一页结论

| 项 | 事实 |
|---|---|
| 现象 | 作者发布 `sai_nag.pt` 评测 63.29，我们自建只有 58.11，差 5.18 pt |
| 直接原因 | 重建时 `scene/dataset_readers.py:117` 实际读的是 **LangSplat 的 language_features（OpenAI ViT-B/16）**，不是 ReLaGS 的 **laion2b_s34b_b88k** 特征 |
| 为什么藏了三轮 | `src/images` 是**符号链接**，`os.path.join(images_folder, '../language_features/…')` 经 realpath 逃到原来那个目录；我们新编的特征放在旁边一直没被读到 |
| 决定性证据 | 同一帧：`cos(读到, 新编)=0.911`、`cos(读到, LangSplat)=1.00000` |
| 修复后 | 0.6610（+2.81 vs 作者 0.6329），簇级特征与作者 `nag_feat` 余弦≈0.9999 |
| 影响面 | 主结果（用作者权重，均值 64.01）**不受影响**；受影响的只有"自建重建"这条支线，以及报告中"编码参数不一致"的错误归因 |
| 附带发现 | `merge_proj.py:207` 的 `torch.normal` 无种子 → 同参数重跑分区不同（本次已用 `tools/run_seeded.py` 固定） |

---

## 1. 症状与时间线

1. 复现 4 场景 LERF-OVS 主结果：用作者发布的 `sai_nag.pt`，均值 **64.01** vs 论文 64.40（Δ −0.39），这条线从一开始就是对的。
2. 为了在"只有几何、没有语义"的场景（以及后续 DeCo 算子替换实验）上能重算，我们尝试用仓库自带的 `scripts/image_encoding.py` 自己编特征、自己跑五步管线重建 `sai_nag_3.0.pt`。
3. 第一次重建结果：**58.11**（figurines），比作者低 5.18 pt。
4. 于是开始归因，连续排了三轮都不动分（见 §2 矩阵）。
5. 最终用"换特征 / 换标签"的交叉实验 + 逐张量比对定位到特征源，修好路径后到 **0.6610**。

---

## 2. 排查矩阵：14 次重跑（全部同一评测脚本、同 56 条 query、同 4 张图）

数据文件：`reproduction/audits/figurines_feature_encoding_20261031/results/rerun_matrix.csv`，逐次日志在 `results/reruns/<arm>__eval.log` 与 `<arm>__q.csv`。

| 臂 | 改了什么 | mIoU | flat |
|---|---|---:|---:|
| `fig_relags2_s7` | 官方编码器 + **真 images 目录** + seed 7 | **0.6610** | 0.6556 |
| `swapA` | 我们的分区 + **作者的 nag_feat**（最大重叠映射） | 0.6468 | 0.6407 |
| `official` | 作者发布的 `sai_nag.pt` | 0.6329 | 0.6278 |
| `fig_up_s7` | 上游 `merge_proj.py`（不带 DeCo 补丁）+ seed 7 | 0.5811 | 0.5755 |
| `fig_relags_s7` | 我们的 merge_proj + seed 7 | 0.5811 | 0.5755 |
| `figurines_s0 / s7 / s42 / s1234 / s2024` | 5 个种子 | 全 0.5811 | 0.5755 |
| `fig_off_s0 / fig_off_s42` | DeCo 算子关闭（`mode=off`） | 0.5811 | 0.5755 |
| `mp_off` | merge_proj 内算子也关 | 0.5811 | 0.5755 |
| `swapB` | 作者的标签 + 我们的特征 | OOM，未跑完（日志保留） | — |

从这张表能直接读出三条，互相独立：

- **换种子不动分**：5 个种子到小数点后四位完全相同 → `torch.normal` 的随机性不是偏差源（虽然它确实让重建不可复现，见 §6）。
- **换实现不动分**：上游版 / 我们的补丁版 / 把 DeCo 算子全关，三种都是 58.11 → 补丁与算子无辜。
- **只换特征就动 6.6 分**：`swapA` 只把特征换成作者的（标签仍是我们自己分割出来的），立刻从 58.11 到 **64.68**，甚至高于作者自带的 63.29 → 我们跑出的分区是好的，问题 100% 在特征。

`swapB` 本来是最干净的对照（作者标签 + 我们特征），但 234k 高斯下 max-overlap 映射把显存打爆，没跑完；用 `swapA` 已足以定案。

---

## 3. 根因一（主因）：特征源错位 —— OpenAI ViT-B/16 vs laion2b_s34b_b88k

### 3.1 两个编码器的差别

| | LangSplat 那套（我们误读到的） | ReLaGS / DeCo 要求的那套 |
|---|---|---|
| SAM | ViT-H，`points_per_side=32` | ViT-H，`points_per_side=32`（同一份权重） |
| CLIP | **OpenAI ViT-B/16**（`/root/.cache/clip/ViT-B-16.pt`） | **laion2b_s34b_b88k**（`open_clip`，HF 缓存） |
| 输出 | `_s.npy`(int32 掩码) + `_f.npy`(float16, 512-d) | 同左，**文件名格式完全一样** |

两者输出维度相同（512）、文件名相同、掩码数量相同，所以**从文件层面看不出任何异常**。

### 3.2 证据一：掩码一致、特征不一致（3 帧重编码）

用官方 `scripts/image_encoding.py` 重编 figurines 前 3 帧，与磁盘上已有的那份对比：

| 帧 | 掩码数（重编 / 已有） | 逐掩码最佳匹配余弦 |
|---|---|---|
| frame 0 | 274 / 274 | 0.913 |
| frame 1 | 246 / 246 | 0.918 |
| frame 2 | 276 / 276 | 0.913 |

掩码数完全相同 → SAM 权重与 `points_per_side` 本来就一致；而特征只有 0.91 的相似度 → **差异完全来自 CLIP 权重**。整帧方向的余弦是 0.9708 / 0.9666 / 0.9714。

### 3.3 证据二：路径穿透探针（决定性）

在重建用的 source 目录里直接打印 reader 那行会打开的文件：

```
cos(读到的 sem, 我们新编的)      = 0.91110 / 0.91477 / 0.91170   （3 帧）
cos(读到的 sem, LangSplat 那套)  = 1.00000 / 1.00000 / 1.00000
```

读到的是 LangSplat 那份，逐位相同。这一条把"编码参数不同""CLIP 权重不同"这类猜测全部排除：**问题不在编码，在读取**。

### 3.4 簇级特征对比（也是当初骗过我的一组数）

| 重建所用的特征 | L1/L2/L3 簇数 | 与作者 `nag_feat` 的余弦 |
|---|---|---|
| LangSplat（错） | 2110 / 468 / 178 | 0.9642 / 0.9680 / 0.9721 |
| laion2b（对） | 2113 / 468 / 178 | ≈0.9999 |

**0.964 这个数看起来很"健康"，是我当时放过它的原因**。簇级特征是大块区域的加权平均，两个 CLIP 空间在这里高度相关；但检索是逐 query 的 top-5 排序，0.96 的相关度足以让部分 query 的 top-1 从"正确物体"滑到"相邻物体"（机制见 §5）。

---

## 4. 根因二（为什么藏了三轮）：符号链接路径穿透

### 4.1 代码

`scene/dataset_readers.py:117`（ReLaGS 与 DeCo 两份完全一致）：

```python
seg_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_s.npy'))).long()
feature_map = torch.from_numpy(np.load(os.path.join(images_folder, f'../language_features/{image_name}_f.npy')))
semantic = {'seg_map': seg_map, 'fg_mask': seg_map != -1, 'sem': feature_map}
```

读取路径是**相对 `images_folder` 拼出来的**，而不是相对场景根目录。拼出来的 `…/images/../language_features/` 一旦 `images` 是符号链接，内核按 realpath 解析：`src/images -> /root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines/images`，于是 `../language_features` 落到 **`VALA/.../figurines/language_features`**，也就是 LangSplat 早就放在那里的那套。

### 4.2 为什么会用符号链接

figurines 有 303 帧、约 1.7 GB。第一次搭 source 时为了省磁盘、省时间，`src/images` 做成了指向原始数据集的软链。这个做法在当时看是纯优化，实际把整条数据通路劫持了。

### 4.3 后果（这是最坑的一点）

即使我们**把新编好的 `laion2b` 特征正确放进 `src/language_features/`**，重跑出来的 mIoU 依然一模一样是 58.11，`nag_feat` 与上一次**逐位相同**（L1/L3 `torch.equal=True`，L2 maxdiff=0）。"编了新特征 → 分数一位不动 → 特征逐位相同"这三件事同时成立，只能有一个解释：**新特征根本没被读到**。这一步本来是极强的自证信号，但我当时把它读成了"编码器是确定性的，所以结果稳定"，方向彻底错了。

### 4.4 修法

- source 目录用**真 `images/` 目录**，里面 299 个（figurines 全量）逐文件软链（`ln -s` 到文件而不是目录）；
- 重建前自检：`realpath(src/images/../language_features)` 必须等于我们放特征的目录，否则脚本 `exit 9 PATH_ESCAPE`。

---

## 5. 为什么 0.964 的余弦不足以报警（机制，标为假设）

检索分数是 `s(i,c) = τ · logmeanexp_{w∈L(c)} ⟨f_i, w⟩`，用的是余弦排序而非绝对阈值。

- 两个 CLIP 空间共享大量语义方向（都在图文对比学习下训练），所以粗粒度"这个簇像什么"仍然对得上 → 簇级余弦 0.96。
- 但 top-5 是在少数几个近邻簇之间做选择，不同空间里"第二、第三名"的顺序会翻转。偏差表现为**少数 query 掉到相邻物体上**，而不是整体崩掉。
- 与逐 query 对照一致：0.6610 vs 0.6329 的差异是 **5 条变好 / 49 条相同 / 2 条变差**，集中在几个相似的 infographic 物体上。

即：错误特征是"近似但错位"的，指标只掉 5 pt 而不是掉到随机水平，这恰好是最难被发现的那一档。

---

## 6. 附带发现：`merge_proj.py` 的随机性没有种子

`merge_proj.py:203-208`：

```python
seg_num = seg_max - seg_min + 2
seg_map -= seg_min - 1
seg_map[~img_mask] = 0
enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
enc = torch.nn.functional.normalize(enc, p=2, dim=-1)
feature_map = enc[seg_map]
```

这行随机张量被射线追踪渲染成"标签图"，再用反向传播把高斯分配到簇 —— 于是**同场景、同参数、同数据重跑，簇划分不同**（实测 L3 在 177/178 之间浮动；ScanNet 上 380/384/387）。

影响分两面：

- 对我们这次的定性结论**不影响**：5 个种子全 58.11，说明在本场景上种子不是偏差源。
- 但对"能不能复现"是硬伤：任何重建都不可 bit 复现，也就没法用"再跑一次看数字变不变"来排除随机性。本次统一用 `tools/run_seeded.py 7` 固定。

这行代码**不是** 58.11 的原因，但它是当初"怀疑种子"这条路走不通、白白多花两轮的根源。

---

## 7. 修复与验证

修复动作（三件，缺一不可）：

1. `src/images` 改为真目录 + 逐文件软链；
2. 用 `scripts/image_encoding.py`（laion2b_s34b_b88k + SAM ViT-H，`--resolution` 与原图一致）重编全部 299 帧；
3. 五步管线参数对齐 `scripts/run_lerf.sh`：`max_weight_pruning --contribution_threshold 0.0005` → `sp_partition -a` → `graph_weight --config ./configs/lerf.yml --level 1` → `sp_partition -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a` → `merge_proj --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3`（经 `run_seeded.py 7`）。

验证（三条独立判据，全部通过）：

| 判据 | 结果 |
|---|---|
| 簇级特征 vs 作者 `nag_feat` | L1/L2/L3 余弦 ≈ 0.9999（错的那版是 0.964/0.968/0.972） |
| 簇数 vs 作者 | 2113/468/178 vs 2103/463/177（同量级，各层 +10/+5/+1） |
| mIoU | **0.6610** vs 作者 0.6329（+2.81），flat 0.6556 vs 0.6278 |

---

## 8. 影响面审计：哪些数字作废、哪些不受影响

| 资产 | 是否受此 bug 影响 | 处理 |
|---|---|---|
| LERF-OVS 主结果均值 **64.01**（用作者 `sai_nag.pt`） | 否 —— 用的就是作者产物 | 保留 |
| 自建重建 **58.11** | 是 —— 作废 | 以 0.6610 取代 |
| 报告中"自建语义场与官方产物不等值 / 编码参数不一致"的因果判断 | 是 —— 错误归因 | 已在 `reproduction/FINAL_REPORT.md` §8.3 与审计 README 中更正 |
| `swapA` 64.68 | 否（对照实验本身有效） | 保留，作为"分区没问题、特征才是变量"的证据 |
| ScanNet-10 数字 | 否 —— `datasets/scannet/*/language_features` 是真目录、由 TFDS zip 解出 | 保留 |
| 3D-OVS 全流程 | 否 —— 数据用 tar 解包成真目录，未用软链 | 保留 |
| ramen / teatime / waldo 新重建（进行中） | 已在启动脚本里内置 realpath 自检 | 见 §9 |

---

## 9. 以后如何在 10 秒内自检（下次别再踩）

搭好 source 目录、跑任何五步管线之前，跑这三条：

```bash
# 1) 读到的到底是谁的特征？（最有效的一条）
python - <<'PY'
import os, numpy as np
src="/path/to/src"; f=sorted(os.listdir(src+"/images"))[0].rsplit('.',1)[0]
p=os.path.join(src,"images",f"../language_features/{f}_f.npy")
print("resolved:", os.path.realpath(p))
print("under src?", os.path.realpath(p).startswith(os.path.realpath(src)))
PY

# 2) 目录本身不能是软链
[ -L "$SRC/images" ] && echo "FAIL: images is a symlink" || echo "ok: real dir"

# 3) 特征维度/帧数对齐
ls "$SRC"/language_features/*_f.npy | wc -l   # 应等于 2 × 帧数
```

再加一条规矩：

- **比指标之前先比张量**。两个候选产物先用 `cos(逐层 nag_feat)` 比对，余弦不是 0.999 级别就不要去解释 mIoU 差异 —— 那是在解释噪声。
- 出现"换了输入、分数一位不变"时，第一反应必须是**读取路径错了**，而不是"这个输入不影响结果"。
- 建 source 目录一律用真目录 + 逐文件软链，`images/` 永不软链（因为上游代码用 `images/../` 拼路径）。

---

## 10. 证据文件索引

仓库 `Youyuan287/DeCo-3DGS`（main = `499bb82`）：

- `reproduction/audits/figurines_feature_encoding_20261031/README.md` —— 审计正文（结论、结果表、§3 穿透、§4 特征源、§5 +2.81 的逐 query 归因、§6 复现步骤、§8 不能主张的）
- `.../results/rerun_matrix.csv` —— 14 次重跑矩阵
- `.../results/reruns/<arm>__eval.log`、`<arm>__q.csv` —— 每次的日志与逐 query 分数（14 组）
- `.../results/off_mine.csv`、`off_official.csv`、`ref_theirfield.csv` —— 逐 query 三方对照
- `reproduction/FINAL_REPORT.md` §8.3 —— 更正块
- `reproduction/3dovs/README.md` —— 另一条线（3D-OVS），与本诊断无关但同属本仓库

服务器端（hn01 `/root/rivermind-data/ReLaGS_Reproduction`）：
`work_feat/`（重编码特征）、`src_fig_relags2/`（修好路径的 source）、`work_seed/fig_relags2_s7/`（重建产物 + 评测）、`tools/run_seeded.py`（种子包装）。