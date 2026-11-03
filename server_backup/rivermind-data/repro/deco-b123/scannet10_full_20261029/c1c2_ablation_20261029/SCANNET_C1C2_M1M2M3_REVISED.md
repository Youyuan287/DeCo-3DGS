# ScanNet C1/C2 off baseline 与 M1/M2/M3 消融（重算）

输入目录：`/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/c1c2_ablation_20261029`

## 十场 scene macro mIoU

| 组合 | DeCo off | M1 fixed | M1+M2 fixed | M1+M3 adaptive | M1+M2+M3 adaptive |
|---|---:|---:|---:|---:|---:|
| off_off | 0.363801 | 0.365972 | 0.365887 | 0.366168 | 0.366352 |
| c1_on_c2_off | 0.422272 | 0.419569 | 0.419433 | 0.419190 | 0.419526 |
| c1_off_c2_on | 0.473284 | 0.473644 | 0.473438 | 0.476688 | 0.476012 |
| c1_on_c2_on | 0.498910 | 0.498006 | 0.497990 | 0.494266 | 0.497740 |

## 严格 C1/C2 主路径消融（只看 off arm）

| 条件 | scene macro mIoU | 相对 C1 off/C2 off |
|---|---:|---:|
| C1 off / C2 off | 0.363801 | +0.000000 |
| C1 on / C2 off | 0.422272 | +0.058471 |
| C1 off / C2 on | 0.473284 | +0.109482 |
| C1 on / C2 on | 0.498910 | +0.135108 |

## 解释

- `off_off` 是本轮新的严格 baseline：raw Gaussian feature + 单一类别文本分数。
- C1/C2 四格使用完全相同的 Gaussian、NAG、GT 与最近点映射。
- M1/M2/M3 只在 `off_off` 这一新 baseline 上解释；旧 `both_lse + renorm=0` 结果不混入本表。
- 若某一组合缺失，报告将保留缺失而不会用旧结果填充。