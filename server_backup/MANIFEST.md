# 服务器备份清单

**来源**：`fj01:/root/rivermind-data`
**备份时间**：2026-11-03
**规模**：4 435 个文件 / 107 MB（打包后 24 MB）

本目录是**文本产物的快照**，用于防止服务器丢失导致脚本、预注册文档与结果表无法追溯。原始数据、模型权重与点云不在备份范围内。

## 包含内容

| 目录 | 文件数 | 大小 | 内容 |
|---|---|---|---|
| `repro/deco-b123/` | 2 999 | 90 MB | DeCo 复现与全部归因/消融实验的脚本、逐场景 CSV/JSON 结果 |
| `ReLaGS_Reproduction/` | 562 | 6.5 MB | ReLaGS 复现管线、ROFA 实验、`reports/` 全部分析报告、`PREREG_relags_rofa.md` |
| `github_DeCo-3DGS/` | 587 | 5.0 MB | 服务器侧的 DeCo 代码仓工作副本（脚本与配置） |
| `ARIS-ReLaGS/` | 163 | 3.4 MB | 技能与工作流脚本 |
| `results/` | 124 | 3.2 MB | 汇总指标表 |

## 筛选规则

**包含**扩展名：`.py` `.md` `.sh` `.csv` `.json` `.yml` `.yaml` `.txt`

**限制**：单文件 `< 2 MB`

**排除路径**：`*/venv/*` `*/site-packages/*` `*/__pycache__/*` `*/.git/*`

## 未包含

| 类别 | 说明 |
|---|---|
| 模型权重 | `*.pt` `*.pth` `*.safetensors`，含 `sai_nag_*.pt`（可由 `merge_proj.py` 重生成） |
| 三维资产 | `*.ply` `*.npz` `*.npy`，含点云、`observations_level3.npz`、NeRF 重建产物 |
| 数据集 | `VALA/` `datasets/`（30 GB / 11 GB） |
| 训练环境 | `thgs_mamba_root/` `venv/` |
| 超限结果 | `repro/deco-b123` 中 6 个 > 30 MB 的逐查询 trace CSV（最大 93 MB） |

> 超过 2 MB 的结果表被刻意排除，它们多为逐查询全量 trace，可由同目录下的脚本与汇总 CSV 重建。

## 恢复方式

列表文件与打包命令可复现：

```bash
cd /root/rivermind-data
find ReLaGS_Reproduction repro/deco-b123 github_DeCo-3DGS results ARIS-ReLaGS \
  -type f -size -2M \( -name '*.py' -o -name '*.md' -o -name '*.sh' \
  -o -name '*.csv' -o -name '*.json' -o -name '*.yml' -o -name '*.yaml' -o -name '*.txt' \) \
  -not -path '*/venv/*' -not -path '*/site-packages/*' \
  -not -path '*/__pycache__/*' -not -path '*/.git/*' \
  > /tmp/backup_list.txt
tar czf /tmp/rmb_backup.tgz -T /tmp/backup_list.txt
```

## 注意

- 这是**快照**，不是同步。服务器后续产生的实验不会自动进入本仓库，需重新打包。
- `github_DeCo-3DGS/` 只是服务器侧工作副本，正式代码仓是 `Youyuan287/DeCo-3DGS` 的 `main` 分支。
- 备份中不含任何凭据、token 或私钥。