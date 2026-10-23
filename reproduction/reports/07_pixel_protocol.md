# 07 · 逐像素相似度阈值协议（论文 §15 + COS3D relevance）

## 实现

`scripts/eval_lerf_pixel.py`：

1. 每个高斯携带其所属簇的 CLIP 特征（`nag_feat[i]` ↔ `nag_label[i+1]`），
   渲染 512 维语言特征图（光栅化器语义通道编译期固定为 20，故按 20 通道分 26 块渲染后拼接）。
2. 逐像素 L2 归一化后与文本嵌入点乘。
3. relevance 用仓库自带的 `ClipSimMeasure` 公式（= COS3D Eq.）：
   `R = min_i softmax(10·[s_query, s_canon_i])[pos]`，canonical = {"object","things","stuff","texture"}。
4. `pred = (R > τ=0.5) & (alpha_coverage > 0.5)`，与 GT 多边形算 IoU；逐图平均再对图平均。

## 结果（waldo_kitchen，HF 模型）

| 口径 | mIoU | mean(pred/gt) |
|---|---:|---:|
| 簇级掩膜（THGS `test_lerf.py` 约定，当前主口径） | 0.5762 | 0.72 |
| 逐像素，level 3（root/物体级） | 0.4995 | — |
| 逐像素，level 2 | 0.4693 | — |
| 逐像素，level 1（最细） | 0.3323 | 8.75 |
| 逐像素，max over levels | 0.4283 | 15.65 |
| 论文 ReLaGS | 0.606 | — |

## 诊断

- 逐像素口径**系统性过选**：`mean(pred/gt)` 达 8.75–15.65（簇级为 0.72）。
  例如 "spatula" 预测 197,070 px 而 GT 仅 2,462 px。
- 根因：`τ=0.5` 在该公式下等价于「query 相似度 > 所有 canonical 短语」。
  LERF/LangSplat 的特征是**训练得到的**判别性特征，Δ 较大；
  ReLaGS 的簇特征是 SAM 掩膜的平均 CLIP 特征，Δ 很小，于是大半画面都能"跑赢 texture/object"。
- 已排除的两个原因：特征未归一化（补归一化后 IoU 逐位不变）、背景像素（补 alpha 覆盖率掩膜后 IoU 逐位不变）。

## 结论

在 ReLaGS 这种**簇级语言场**上，逐像素 + 全局阈值 0.5 会显著劣化（waldo 0.5762 → 0.4995）。
论文 §15 的表述与 §12 Algorithm 1（簇级检索 + 最大相似度落差）在实现层面不一致；
现有证据更支持论文数字来自簇级路径。
逐像素口径作为**并列口径**保留，不用于替代主结果。
