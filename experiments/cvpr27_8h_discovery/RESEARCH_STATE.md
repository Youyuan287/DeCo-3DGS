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
- **阻塞：本仓库没有配置任何 remote（`git remote -v` 为空），`gh` 未安装。
  无法 push。需要用户提供远程地址与凭证。** 已按要求仅做本地提交，未尝试绕过。

## 未解决 bug
无（probe_rank 的 rank1/oracle 索引错误已修复）。

## 下一步最优先
若获得远程权限：①push 本分支；②补 CSLS/hubness 公平对照；③在未参与设计的场景复现。

## 已否证、不可重复
见 `NEGATIVE_RESULTS.md`。
