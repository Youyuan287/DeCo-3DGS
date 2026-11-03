# V3 两因素正交消融

所有四格都固定使用同一份已认证 B1 debiased bank、冻结 observation/NAG、C1=image、C2=canon、temperature=0.08、correction coefficient=0.25、原始层级检索和 M2 reliable。没有参数搜索。已有 V2 reliable（全区域+绝对）及已有 V3（原始主路径已选 root+相对截断）直接复用；本轮只运行 selected+absolute 与 all+relative 两个缺格。`PROVENANCE_CHECK.json` 的 16 个四格比较均通过（每场景 bank/observation/NAG/members SHA-256 完全一致）。

|scene|all+absolute (V2)|selected+absolute|all+relative|selected+relative (V3)|
|---|---:|---:|---:|---:|
|figurines|0.593810|0.586945|0.592734|0.585869|
|ramen|0.529587|0.518076|0.517928|0.518076|
|teatime|0.793896|0.793896|0.793896|0.793896|
|waldo_kitchen|0.659209|0.659209|0.659209|0.626687|
|macro|0.644126|0.639532|0.640942|0.631132|


相对 all+absolute 的 208-query 改善/退化/不变分别为：all+absolute (10, 2, 196)；selected+absolute (2, 1, 205)；all+relative (2, 1, 205)；selected+relative (0, 1, 207)。

宏平均主效应：范围（selected−all，跨 gain 平均）=-0.007202；增益（relative−absolute，跨 scope 平均）=-0.005792；交互（selected-relative − selected-absolute − all-relative + all-absolute）=-0.005216。因此不能只用原始 V3 与 V2 的差分归因；三个正交对比共同给出范围、相对截断及其交互的证据。

全区域+相对的比较集合固定为该 scene bank 的所有有效 root ID，完全由 bank 决定，查询前固定、不用 GT、不会由修正后候选更新。selected 两格的集合则由同一查询未修正 DeCo 主路径在读出前的 root_id 固定，并且不会回写；随后仍执行原始检索。因此相对操作改变的是 root 间的 pairwise 修正，而不是对所有分数减去同一常数。

## 指定机制案例

ramen 六个 nori（V2 absolute-all 删除背景；其余三格未复现该选择）：

|image|query|arm|ΔIoU|ΔTP|ΔFP|ΔFN|mask XOR|
|---|---|---|---:|---:|---:|---:|---:|
|frame_00006|nori|all_absolute_reliable|+0.146172|+0|-3498|+0|3498|
|frame_00024|nori|all_absolute_reliable|+0.162716|-119|-20180|+119|20299|
|frame_00065|nori|all_absolute_reliable|+0.230626|-32|-11242|+32|11274|
|frame_00081|nori|all_absolute_reliable|+0.084551|+0|-2282|+0|2282|
|frame_00119|nori|all_absolute_reliable|+0.098460|-24|-10854|+24|10878|
|frame_00128|nori|all_absolute_reliable|+0.088739|-9|-4868|+9|4877|
|frame_00006|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00024|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00065|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00081|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00119|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00128|nori|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00006|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00024|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00065|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00081|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00119|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00128|nori|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00006|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00024|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00065|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00081|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00119|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00128|nori|selected_relative_reliable|+0.000000|+0|+0|+0|0|


ramen/frame_00128/corn（absolute-all 与 all-relative 都引入 root 12 背景；selected 范围两格没有）：

|image|query|arm|ΔIoU|ΔTP|ΔFP|ΔFN|mask XOR|
|---|---|---|---:|---:|---:|---:|---:|
|frame_00128|corn|all_absolute_reliable|-0.014447|+0|+57472|+0|57472|
|frame_00128|corn|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00128|corn|all_relative_reliable|-0.014447|+0|+57472|+0|57472|
|frame_00128|corn|selected_relative_reliable|+0.000000|+0|+0|+0|0|


figurines tesla door handle（absolute-all 与 all-relative 都恢复目标；selected 范围两格没有）：

|image|query|arm|ΔIoU|ΔTP|ΔFP|ΔFN|mask XOR|
|---|---|---|---:|---:|---:|---:|---:|
|frame_00041|tesla door handle|all_absolute_reliable|+0.067802|+466|+1|-466|467|
|frame_00105|tesla door handle|all_absolute_reliable|+0.305141|+3229|-5826|-3229|9055|
|frame_00041|tesla door handle|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00105|tesla door handle|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00041|tesla door handle|all_relative_reliable|+0.067802|+466|+1|-466|467|
|frame_00105|tesla door handle|all_relative_reliable|+0.305141|+3229|-5826|-3229|9055|
|frame_00041|tesla door handle|selected_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00105|tesla door handle|selected_relative_reliable|+0.000000|+0|+0|+0|0|


waldo_kitchen/frame_00053/ottolenghi（仅 selected-relative 发生灾难性删选）：

|image|query|arm|ΔIoU|ΔTP|ΔFP|ΔFN|mask XOR|
|---|---|---|---:|---:|---:|---:|---:|
|frame_00053|ottolenghi|all_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00053|ottolenghi|selected_absolute_reliable|+0.000000|+0|+0|+0|0|
|frame_00053|ottolenghi|all_relative_reliable|+0.000000|+0|+0|+0|0|
|frame_00053|ottolenghi|selected_relative_reliable|-0.975669|-16762|-271|+16762|17033|


`V3_FACTORIAL_PER_QUERY.csv` 对 4×208 个条目分别记录原始与新候选三元组 `(level,root_id,candidate_id)`、Gaussian 并集增删、逐像素 mask XOR、以及相对主路径的 TP/FP/FN 变化。mask 或 Gaussian 是否改变由各自集合/逐像素比较直接计算；不以 IoU 或像素总数代替。
