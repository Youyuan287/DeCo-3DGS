# DeCo-LI 结果

固定 M1 bank、M2 OFF、C1=image、canon、renorm=0、topk=5、level_until=3、filter_small=50、sel=gap、coefficient=0.25。E0/E1/E2/E3 均使用真实 NAG 搜索和 Gaussian renderer。

## 结果表

| 组别 | figurines | ramen | teatime | waldo_kitchen | macro | 相对 E1 | dev | holdout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.585869 | 0.518076 | 0.793896 | 0.659209 | 0.639262 | -0.004863 | 0.579244 | 0.698414 |
| E1 | 0.593810 | 0.529587 | 0.793896 | 0.659209 | 0.644126 | +0.000000 | 0.583078 | 0.701194 |
| E2 | 0.586945 | 0.529584 | 0.793896 | 0.626687 | 0.634278 | -0.009848 | 0.562750 | 0.698749 |
| E3 | 0.593810 | 0.517928 | 0.793896 | 0.659209 | 0.641211 | -0.002915 | 0.579192 | 0.701194 |

## 逐查询与 Gaussian 变化
- E2: improved=0, degraded=4, unchanged=204; candidate changed vs E1=107; max degradation=-0.975669; Gaussian +17,510/-46,872 relative to E0.
- E3: improved=0, degraded=6, unchanged=202; candidate changed vs E1=98; max degradation=-0.230626; Gaussian +70,302/-14,542 relative to E0.

## 重点查询
- `DECO_LI_PER_QUERY.csv` 保留 nori、corn、tesla door handle、ottolenghi 的逐查询 IoU、候选变化、Gaussian 增删和 mask XOR。

## 判断
- E2/E3 是否值得保留取决于其相对 E1 的 scene macro、开发/保留组和退化数量；本轮不依据单一案例调参。
- 若新算子没有稳定净收益，则不进行 coefficient 第二阶段。