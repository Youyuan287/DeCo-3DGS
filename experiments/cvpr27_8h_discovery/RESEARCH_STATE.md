# RESEARCH STATE

- 开始：2026-11-03T01:27:15+08:00
- 硬截止：2026-11-03T09:27:15+08:00
- 分支：`research/cvpr27-autonomous-8h`

## 当前结论（一句话）
审计器 0.6711 已用掉其选项集天花板的 88.6%；所有分数层/几何层校正路线均已否证；
要让 LERF 明显突破 0.675，必须新增与 released/C/mask 不相关的候选，而现有可测信号产不出。

## 已完成实验
E0 基线复现 / E1 VCLS / E2 局部对比 / E3–E3b 行偏置 / E5–E5b 审计叠加 /
E6 簇 bootstrap / E7 选项 oracle / E8 单候选 oracle / E9 非文本信号 probe /
E10 支配模式 / E11 排序器 probe。

## 当前最佳可信结果
- LERF：审计器 **0.6711**（training-free，CI 正，簇 bootstrap 稳健）
- ScanNet：zR+E0 **48.77**（zR 本身 +5.62）

## 正在执行的进程
无（服务器上无遗留任务；LERF 四场景与 axis 四场景均已结束）。

## Git
- 本地分支 `research/cvpr27-autonomous-8h`
- 本地提交 `791332c`（`research: two-axis bias decomposition, audit ceiling, and signal probes`）
- **推送阻塞，原因有三，均已核实：**
  1. `git remote -v` 为空，本仓库从未配置 remote；`gh` 未安装。
  2. 远程 `https://github.com/Youyuan287/DeCo-3DGS` 可达，但其 HEAD 为 `1aa0bf1`，
     **本地仓库不包含该提交**（`git cat-file` 失败），即两者是无关历史。
     本目录是研究笔记仓库，不是 DeCo-3DGS 的克隆。
  3. 因此 `git push -u origin HEAD` 会在用户的公开仓库里建立一个历史无关的分支。
     这属于对共享状态的不可逆改动，且很可能不是用户本意 —— 按要求记录而不猜测执行。
- 恢复方式（需用户执行一次）：
  `git remote add origin <正确仓库> && git push -u origin research/cvpr27-autonomous-8h`
  或者把 `experiments/cvpr27_8h_discovery/` 拷进真正的 DeCo-3DGS 克隆后提交。

## 未解决 bug
无（probe_rank 的 rank1/oracle 索引错误已修复）。

## 文献检查（固定流程）
- 工具：`tools/litcheck.py`（arXiv API，经 curl）；协议：`LITERATURE_PROTOCOL.md`；账本：`NOVELTY_LOG.md`
- 已查实：粒度诊断被 QAGaussian 占；zR/原型校准被 PTC、Target-Aware 占；实例发现被 DIGS 占。
- 唯一未占（弱证据）：对返回的 3D 区域集合做保形覆盖保证。

## 下一步最优先
若获得远程权限：①push 本分支；②补 CSLS/hubness 公平对照；③在未参与设计的场景复现。

## 已否证、不可重复
见 `NEGATIVE_RESULTS.md`。
