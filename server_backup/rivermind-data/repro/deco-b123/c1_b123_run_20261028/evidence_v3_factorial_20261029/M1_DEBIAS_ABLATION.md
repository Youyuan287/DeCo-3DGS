# M1 raw 与场景去偏 grouping 的最小消融

固定同一冻结 observations、同一 NAG、同一几何 overlap、K=3、geo_weight=0.25、同一 all+absolute M3 读出与未更改的 DeCo 主路径。两侧均 `evidence-v2=semantic`，即 M2 reliability=1；唯一变化是发现模式的特征：raw 为 `h=f`，debiased 为 `h=normalize(f-(f·u)u)`。最终 canon 评分的 `features` 在两边均是原始 normalized CLIP `f`，未重新训练或扫描。

|scene|debiased|raw|debiased−raw|
|---|---:|---:|---:|
|figurines|0.593811|0.592734|+0.001076|
|ramen|0.529587|0.529587|+0.000000|
|teatime|0.793896|0.793896|+0.000000|
|waldo_kitchen|0.659209|0.659209|+0.000000|
|macro|0.644126|0.643857|+0.000269|


所有 208 条主路径 IoU 均逐条相同，因而对照有效。逐条差异保存在 `M1_DEBIAS_PER_QUERY.csv`。raw bank 由隔离 builder 生成，metadata 与 control JSON 保存了 observation/NAG SHA256、region-id 一致性、形状一致性及仅改变 grouping-space 的声明；`PROVENANCE_CHECK.json` 的另 4 个 M1 比较也通过。未覆盖任何生产 bank。
