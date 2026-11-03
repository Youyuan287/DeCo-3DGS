# ReLaGS · LERF-OVS 四场景复现

仓库：`github.com/dfki-av/ReLaGS` @ `52cc0c9`
完整交付说明见 **[`reports/00_FINAL_DELIVERY.md`](reports/00_FINAL_DELIVERY.md)**。

## 结果速览

纯 HuggingFace 输入，簇级协议，`topk=5 / level_until=3 / filter_small=50`（前三者为仓库函数默认值）。

| Scene | 复现 | 论文 | Δ | queries |
|---|---:|---:|---:|---:|
| figurines | 63.29 | 64.7 | −1.41 | 56 |
| ramen | 51.15 | 51.2 | −0.05 | 71 |
| teatime | 80.97 | 81.0 | −0.03 | 59 |
| waldo_kitchen | 60.62 | 60.6 | +0.02 | 22 |
| **Mean** | **64.01** | **64.40** | **−0.39** | 208 |

## 目录

```
ReLaGS/           官方仓库（3 处兼容性补丁，见 patches/）
ref/relags_hf/    HF 发布的 2DGS 场景 + sai_nag.pt
ref/hf_lerf/      HF lerf_ovs.zip 解包（images + label）
ref/hf_only/      评测用源目录（images/sparse 软链至 HF）
scripts/          eval_lerf_relags.py（主评测）、run_hf_only.sh、sweep_lerf.py 等
results/          逐 query/逐场景指标、掩膜、可视化、manifest
reports/          00 最终交付稿，01–08 分阶段记录
patches/          3 处兼容性 diff
```

## 环境

Python 3.10 + torch 2.1 + CUDA 11.8，`venv/`；CUDA 扩展需 `CUDA_HOME` 指向 conda 前缀并把
`$CUDA_HOME/targets/x86_64-linux/include` 加入 `CPATH`。

## 运行

```bash
# 四场景完整评测（约 3 分钟/场景）
bash scripts/run_hf_only.sh
```

## 重要说明

- 评测协议**不是**论文公开内容（仓库未发布 mIoU 代码），采用与 THGS `test_lerf.py` 同构的簇级协议；
  依据与敏感性分析见 `reports/04`、`reports/08`。
- 逐像素阈值口径（论文 §15 字面）实测严重过选，仅并列记录于 `reports/07`，不作主指标。
- `topk=5` 为仓库默认值；此前误用 THGS 的 `topk=3` 得到 59.76，已作废。
- 未从零训练 2DGS，复用官方发布场景（已逐点核验与 THGS 30k 同源）。

## 已知第三方写入

`THGS/` 下与本任务无关的文件在同期被其他会话修改；本任务对 `THGS/` 只读，未写入。
