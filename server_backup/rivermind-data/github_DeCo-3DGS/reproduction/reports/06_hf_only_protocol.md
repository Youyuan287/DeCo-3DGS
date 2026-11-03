# 06 · 纯 HF 输入口径的 LERF-OVS 复现结果

用户要求：严格使用 HF 上发布的四场景预训练模型，不得混入本地副本。
本报告给出该口径下的完整输入清单、可追溯证据与结果。

## 1. 输入清单（全部来自 HF）

| 角色 | 文件 | 来源 |
|---|---|---|
| 2DGS 底座（MWP 后） | `point_cloud/iteration_0/point_cloud.ply` | `hf:dfki-av/ReLaGS/scenes/LeRF/<scene>/` |
| 作者最终语义场 | `sai_nag.pt` | 同上 |
| 相机/参数 | `cameras.json`, `input.ply`, `cfg_args` | 同上 |
| 图像 | `<scene>/images/*.jpg` | `hf:MinghanQin/lerf_ovs/lerf_ovs.zip` |
| GT 掩膜与查询 | `label/<scene>/*.json` | 同上 |

逐文件 sha256 见 `results/hf_only_manifest.csv`。

## 2. 可追溯性证据

1. **GT**：HF `lerf_ovs.zip` 内 22 个 `label/*.json` 与本地副本 **22/22 md5 完全一致**；本地副本与 THGS 内另一份独立副本也逐场景 md5 一致。
2. **图像**：抽查 figurines/ramen/teatime 三张，HF 与本地 **md5 完全一致**。
3. **模型**：`iteration_0`、`sai_nag.pt` 均直接下载自 HF，sha256 记录在 manifest。
4. 评测过程不读取任何 `langsplat/language_features`（`grep` 确认评测脚本无该依赖）。

## 3. 评测协议（这是唯一非 HF、非作者的环节）

作者**未发布评测代码**，因此协议由我实现，并在报告中固定声明：

- 渲染：`semantics` 第 0 通道 `> 0.5` 视为前景（与 THGS `test_lerf.py` 同约定）
- 检索：`search_matched_superpoint_in_mhtree(topk=3, level_until=3, filter_small=50, remove_small=True)`
- 指标：逐图对 query 求 IoU 平均，再对图求平均（`mIoU`）；另给全体 query 池化的 `flat`
- GT：多边形按 `fillPoly` 栅格化，同 query 多个实例取 `maximum` 合并

## 4. 结果

| 场景 | mIoU | flat | 论文 Table 3 | 差 |
|---|---:|---:|---:|---:|
| figurines | 58.11 | 57.85 | 64.7 | −6.59 |
| ramen | 40.40 | 39.33 | 51.2 | −10.80 |
| teatime | 82.91 | 82.25 | 81.0 | +1.91 |
| waldo_kitchen | 57.62 | 57.91 | 60.6 | −2.98 |
| **Mean** | **59.76** | 59.34 | **64.4** | **−4.64** |

208 个 query、22 张评测图。

## 5. 一致性检验

同一批 HF 模型，先前用本地副本（VALA 图像 + VALA label）跑出的数字为
58.11 / 40.40 / 82.91 / 57.62，**与本次逐位相同**。
说明此前的本地副本与 HF 内容一致，历史数字未被污染。

## 6. 边界声明

本表能支持的结论只有一条：**作者发布的语义场 + 我实现的评测协议 = 59.76**。
论文 64.4 的差距中，无法区分"协议差异"与"作者内部实现差异"，因为评测代码未公开。
不得据此表声称"ReLaGS 方法本身复现失败"，也不得声称"已复现到 64.4"。
