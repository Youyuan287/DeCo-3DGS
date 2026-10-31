# 协议问题：文本能否进入 2D→3D 特征聚合？

## 1. 标准 3D-OVS 协议（query-agnostic field）

以往方法（ReLaGS/SAI3D/Dr.Splat 等）的分工是：

1. **建场**：逐视角 CLIP 特征 + 2D mask → 聚合/提升到 3D 超点/区域；
   这一步**不使用文本**，得到固定的语义高斯场。
2. **查询**：给定文本 t，只对固定场做打分与选择（`<f_c, t>`）。

即 `f_c` 与 t 无关；同一个场可以被任意 query 复用。

## 2. 我们的 `id_top3` 违反了这一点

我们的聚合是：

```
f_c(t) = mean of top-3 { f_o : <f_o, t> 最大 }     # 文本参与聚合
score  = < f_c(t), t >
```

`f_c(t)` **依赖 query**：每个 query 的场不同，不能预先建好一次复用。
这不是“更好的场”，而是**query-conditioned readout**，是另一个任务/协议。
把它和 ReLaGS 的固定场直接比 mIoU，不公平。

### 哪些信号是 query-agnostic（协议内）

| 信号 | 聚合是否用文本 | 协议 |
|---|---|---|
| `id_mean` | 否（可见性/覆盖率加权） | 标准 |
| `id_mean_c` | 否（中心化） | 标准 |
| `agree` / `agree_c` | 否（跨视角一致性） | 标准 |
| `id_top3` | **是（文本 top-3）** | **query-conditioned** |
| `id_top1` | 是 | query-conditioned |
| mask rerank / gateE（用 id_top3） | 是 | query-conditioned |

## 3. “主信号”和“互补信号”是什么意思

- **主信号（main）**：released NAG 分数与选择。它是固定场（query-agnostic）+ 查询时文本打分，
  是标准协议下的基线。
- **互补信号**：逐视角 mask/观测证据。它本身可以有两种用法：
  - query-agnostic（id_mean/agree）→ 仍在标准协议内；
  - query-conditioned（id_top3）→ 变成 query-conditioned readout，**不能与主信号直接同协议比较**。

之前 gating 里的“主信号 vs mask 信号”就是这个区别；我没有把协议差异讲清楚，是表述失误。

## 4. 公平比较需要什么

1. **同协议消融**：固定场（id_mean/id_mean_c/agree）+ 查询时文本打分，对比 main。
2. 如果要用 query-conditioned 聚合，必须：
   - 明确声明这是 query-conditioned readout；
   - 给 baseline 同样的机制（例如 baseline 也用文本 top-k 聚合），否则不是同输入；
   - 或在 novel view / 固定场评测下比较。
3. 报告两个数字：标准协议下的增益，和 query-conditioned 协议下的增益，不能混为一谈。

## 5. 已补的实验（B0，同一 released 读权、同一候选池）

| 版本 | 聚合是否用文本 | macro | Δ vs main | 95% CI |
|---|---|---:|---:|---|
| main | 否 | 0.6393 | — | — |
| id_mean rerank | **否（query-agnostic）** | 0.6448 | **+0.0056** | [+0.0000,+0.0134] |
| id_mean_c rerank | **否（query-agnostic）** | 0.6421 | **+0.0028** | [+0.0000,+0.0085] |
| id_top3 rerank | 是（query-conditioned） | 0.6619 | +0.0226 | [+0.0138,+0.0319] |
| gateE（id_top3） | 是（query-conditioned） | 0.6760 | +0.0366 | [+0.0243,+0.0470] |

**结论**：
1. **标准协议下（query-agnostic field），增益只有 +0.28~+0.56，未达 +0.01 门控。**
2. 我们之前报告的 +2.3~+3.7 主要来自 **query-conditioned readout**（文本进入聚合），
   不是“更好的语义场”。
3. 因此在标准 3D-OVS 协议下，**目前不能声称优于 ReLaGS/DeCo**；0.676 必须标注为
   query-conditioned 结果。

## 6. 接下来必须做的选择

- **路线 A（标准协议）**：在 query-agnostic 场里做真正的改进（更好的 mask 关联/覆盖加权/固定场聚合），
  目标是让 id_mean 类信号从 +0.5 提到 +1 以上；否则不能作为 3D-OVS 方法投稿。
- **路线 B（query-conditioned）**：明确把它定义成新任务/新 readout，给 baseline 同样的机制
  （文本 top-k 聚合）再比较；不能与固定场方法直接比。
- 不再把“主信号 vs 互补信号”当作同协议比较；主信号是固定场，id_top3 是 query-conditioned。

## 7. 术语

- **主信号（main）**：released NAG 分数与选择；固定场 + 查询时文本打分（标准协议）。
- **互补信号**：逐视角 mask 证据。query-agnostic 用法（id_mean/agree）仍在标准协议内；
  query-conditioned 用法（id_top3）是另一个协议。之前把两者混着说，是表述错误。