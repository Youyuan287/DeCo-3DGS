# M1-SAS + M2 + M3 四场景消融

## 结论

在同一批固定 seed=1234 的新 observation、同一 M1-SAS bank、同一官方 NAG/Gaussian 和同一评测协议下，叠加 M2 reliable 与 M3 query-adaptive 后，四场景结果与 M1-SAS（M2 关闭、M3 开启）逐查询完全一致。当前数据上 M2 没有可观察性能贡献；M3 已包含在两种辅助配置中。

本轮没有重建 bank、没有运行 CLIP 特征生成、没有修改 NAG/Gaussian/ROFA/C1/C2，也没有参数扫描。

## 统一协议

`center=image`, `text=canon`, `renorm=0`, `topk=5`, `level_until=3`, `filter_small=50`, `sel=gap`, `temperature=0.08`, correction coefficient `0.25`。

- M1-SAS：semantic weighted-Jaccard bank，M2 关闭（reliability=1），M3 query-adaptive Softmax 开启。
- M1-SAS + M2 + M3：同一 bank 和同一 M3，M2 使用现有多视角语义一致性可靠性；仅将每个有效 mode 的 reliability 乘入辅助增益。
- DeCo off：此前同协议的原始主路径认证结果，仅作跨报告基线；其 observation/NAG provenance 与本轮 M1-SAS 新 observation 不同，不能解释为同一 bank 的逐字节对照。

## Scene mIoU（每图 query 均值，再跨图平均）

| 场景 | DeCo off | M1-SAS (M2 off, M3 on) | M1-SAS + M2 + M3 | M1-SAS+M2+M3 − DeCo | M2 − M1-SAS |
|---|---:|---:|---:|---:|---:|
| figurines | 0.585869 | 0.593810 | 0.593810 | +0.007942 | +0.000000 |
| ramen | 0.518076 | 0.517910 | 0.517910 | −0.000165 | +0.000000 |
| teatime | 0.793896 | 0.793896 | 0.793896 | +0.000000 | +0.000000 |
| waldo_kitchen | 0.659209 | 0.659209 | 0.659209 | +0.000000 | +0.000000 |
| 四场景等权宏平均 | **0.639262** | **0.641207** | **0.641207** | **+0.001944** | **+0.000000** |

## Flat mIoU（所有 query 直接平均）

| 场景 | DeCo off | M1-SAS (M2 off, M3 on) | M1-SAS + M2 + M3 | M1-SAS+M2+M3 − DeCo | M2 − M1-SAS |
|---|---:|---:|---:|---:|---:|
| figurines | 0.580394 | 0.587927 | 0.587927 | +0.007533 | +0.000000 |
| ramen | 0.511650 | 0.511426 | 0.511426 | −0.000224 | +0.000000 |
| teatime | 0.785209 | 0.785209 | 0.785209 | +0.000000 | +0.000000 |
| waldo_kitchen | 0.681240 | 0.681240 | 0.681240 | +0.000000 | +0.000000 |
| 四场景等权宏平均 | **0.639623** | **0.641450** | **0.641450** | **+0.001827** | **+0.000000** |

查询数为 56/71/59/22，总计 208 条。

## M2 独立差分核验

逐场景比较了 M1-SAS 与 M1-SAS+M2+M3 的逐查询 CSV：

| 场景 | query IoU 改变 | 候选 Gaussian/区域 ID 改变 | 预测像素数改变 |
|---|---:|---:|---:|
| figurines | 0/56 | 0/56 | 0/56 |
| ramen | 0/71 | 0/71 | 0/71 |
| teatime | 0/59 | 0/59 | 0/59 |
| waldo_kitchen | 0/22 | 0/22 | 0/22 |
| 合计 | **0/208** | **0/208** | **0/208** |

因此 M2 的可靠性权重虽然被计算并记录在 `auxiliary_details` 中，但在本轮固定系数和候选 gap 选择下没有改变候选排序/阈值，也没有改变渲染 mask。

## 输入与输出

- M2+M3 远程输出：`/root/rivermind-data/repro/deco-b123/m1_sas_m2m3_20261029/`
- 本地逐查询结果：本目录下的 `figurines_per_query.csv`、`ramen_per_query.csv`、`teatime_per_query.csv`、`waldo_kitchen_per_query.csv`
- M1-SAS 对照：`../m1_sas_20261029/eval/`
- M2+M3 日志：本目录下四个 `*.log`

四个 SAS bank 均通过 evaluator 的 NAG、observation SHA256 校验，未混用旧 V2 bank。

## 解释

本轮最可靠的结论是：M1-SAS+M3 的宏平均相对原始 DeCo 提升约 `+0.001944` scene mIoU（flat 约 `+0.001827`），但提升主要来自 figurines；ramen 仍略低于原始 DeCo。M2 在当前 bank/观测和固定 M3 读出中没有独立可观察收益，不能宣称 M2 带来性能提升。
