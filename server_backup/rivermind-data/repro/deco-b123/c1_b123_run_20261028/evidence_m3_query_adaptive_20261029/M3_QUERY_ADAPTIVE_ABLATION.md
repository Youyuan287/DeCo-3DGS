# M3 查询自适应权重独立消融

固定官方 NAG、Gaussian、observation、Debiased M1 bank、原始 ReLaGS ROFA + DeCo 主路径、C1=image、canon、全区域绝对增益、temperature=0.08、coefficient=0.25。M2 在两组中均关闭（semantic 配置）。唯一变化是模式组合权重：Fixed 使用区域内 normalized priors；Adaptive 使用现有 V2 Softmax(query score / 0.08 + log prior)。两者都使用同一当前 query 的非负 mode advantage。DeCo off、Adaptive 直接复用已认证 semantic 结果，Fixed 为本轮唯一新增 GPU cell。

|scene|DeCo off|Fixed prior|Query-adaptive Softmax|Fixed−off|Adaptive−off|Adaptive−Fixed|
|---|---:|---:|---:|---:|---:|---:|
|figurines|0.585869|0.593810|0.593810|+0.007942|+0.007942|+0.000000|
|ramen|0.518076|0.517928|0.529587|-0.000147|+0.011512|+0.011659|
|teatime|0.793896|0.793896|0.793896|+0.000000|+0.000000|+0.000000|
|waldo_kitchen|0.659209|0.659209|0.659209|+0.000000|+0.000000|+0.000000|
|macro|0.639262|0.641211|0.644126|+0.001949|+0.004863|+0.002915|


scene mIoU 为每幅图内 query IoU 先平均，再跨图平均；四场景 macro 为四个 scene mIoU 等权平均。off 来自原始 off summary.miou，且四组原始 CSV 重算均与对应 summary 在 1e-6 内一致。旧报告错误使用 main_miou（实际是 flat）已修正。

|scene|DeCo flat|Fixed flat|Adaptive flat|Fixed−DeCo flat|Adaptive−DeCo flat|Adaptive−Fixed flat|
|---|---:|---:|---:|---:|---:|---:|
|figurines|0.580394|0.587927|0.587927|+0.007533|+0.007533|+0.000000|
|ramen|0.511650|0.511447|0.522873|-0.000203|+0.011223|+0.011426|
|teatime|0.785209|0.785209|0.785209|+0.000000|+0.000000|+0.000000|
|waldo_kitchen|0.681240|0.681240|0.681240|+0.000000|+0.000000|+0.000000|

|scene等权macro|0.639623|0.641456|0.644312|+0.001832|+0.004689|+0.002857|

flat 行的 macro 是每场 flat 均值再等权平均；它不等于 208 query 合并平均。

208 条逐查询相对 DeCo 的改善/退化/不变：Fixed=(4, 2, 202)，Adaptive=(10, 2, 196)；Adaptive 相对 Fixed=(0, 6, 202)（该 tuple 为 Fixed 优于 Adaptive、Fixed 劣于 Adaptive、不变）。

## 机制案例

### ramen 六个 nori
- frame_00006 / nori: fixed Δ=+0.000000, adaptive Δ=+0.146172, fixed−adaptive=-0.146172, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=+0/-3498/+0
- frame_00024 / nori: fixed Δ=+0.000000, adaptive Δ=+0.162716, fixed−adaptive=-0.162716, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=-119/-20180/+119
- frame_00065 / nori: fixed Δ=+0.000000, adaptive Δ=+0.230626, fixed−adaptive=-0.230626, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=-32/-11242/+32
- frame_00081 / nori: fixed Δ=+0.000000, adaptive Δ=+0.084551, fixed−adaptive=-0.084551, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=+0/-2282/+0
- frame_00119 / nori: fixed Δ=+0.000000, adaptive Δ=+0.098460, fixed−adaptive=-0.098460, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=-24/-10854/+24
- frame_00128 / nori: fixed Δ=+0.000000, adaptive Δ=+0.088739, fixed−adaptive=-0.088739, candidate_changed=True, mask_equal=False, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=-9/-4868/+9

### ramen/frame_00128/corn
- frame_00128 / corn: fixed Δ=-0.014447, adaptive Δ=-0.014447, fixed−adaptive=+0.000000, candidate_changed=False, mask_equal=True, fixed TP/FP/FN=+0/+57472/+0, adaptive TP/FP/FN=+0/+57472/+0

### figurines tesla door handle
- frame_00041 / tesla door handle: fixed Δ=+0.067802, adaptive Δ=+0.067802, fixed−adaptive=+0.000000, candidate_changed=False, mask_equal=True, fixed TP/FP/FN=+466/+1/-466, adaptive TP/FP/FN=+466/+1/-466
- frame_00105 / tesla door handle: fixed Δ=+0.305141, adaptive Δ=+0.305141, fixed−adaptive=+0.000000, candidate_changed=False, mask_equal=True, fixed TP/FP/FN=+3229/-5826/-3229, adaptive TP/FP/FN=+3229/-5826/-3229

### waldo_kitchen/frame_00053/ottolenghi
- frame_00053 / ottolenghi: fixed Δ=+0.000000, adaptive Δ=+0.000000, fixed−adaptive=+0.000000, candidate_changed=False, mask_equal=True, fixed TP/FP/FN=+0/+0/+0, adaptive TP/FP/FN=+0/+0/+0

`M3_FIXED_VS_ADAPTIVE_PER_QUERY.csv` 逐条记录真实候选三元组、候选集合是否变化、两组相对主路径的 Gaussian 增删、render mask XOR、mask SHA、以及 TP/FP/FN。`M3_WEIGHT_AND_CANDIDATE_TRACE.csv` 记录实际被触发模式的 fixed/adaptive 权重、优势、修正量和候选变化标记。未用 GT 选择模式或改变候选集合。

## 结论

Adaptive macro mIoU 为 0.644126，Fixed 为 0.641211，DeCo 为 0.639262。Fixed 相对 DeCo 的净收益为 +0.001949，Adaptive 为 +0.004863；Adaptive 相对 Fixed 为 +0.002915。

因此，V2 正向净收益的**符号不依赖查询自适应 Softmax 权重**：Fixed prior 已保留 +0.001949 的 macro 净收益。但 Softmax 确实提供了独立的增量贡献：Adaptive 比 Fixed 高 +0.002915，且增量集中在六个 nori 的候选/mask 切换；corn、tesla door handle 和 ottolenghi 在两种权重下不变。因此 M3 Softmax 可以作为一个幅度有限但有独立证据的性能贡献报告，同时不能声称它是 V2 保持正收益的必要条件。后续不启动其它实验。
