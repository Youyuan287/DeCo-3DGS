# 执行账本：docs/research_decision_20261024/RESEARCH_DECISION.md

用户2026-10-24明确批准“可以，进行”：只补figurines固定候选排名/基数缺失两格，不扩场景、不扫参、不重跑baseline。

使用run-experiment和executing-plans的测试/预检要求；项目独立实验目录及不覆盖上游约束优先。当前工作区master上的资产几乎全为未跟踪历史数据；不搬移、不提交整库、不创建空历史worktree。新增文件仅在本独立目录，源代码/模型/旧结果不动。

Pre-flight: 两格共用已保存pool、scores及原render/evaluator；冻结记录五字段需在renderer边界还原为原七字段tuple。GT仅在全部预测后用于评价。

Ruling: 直接重放已有21query分数，不再加载CLIP/重投影全部节点；所有原输入哈希逐项核验。两旧格仅CPU验证原节点和Gaussian并集哈希，不重跑GPU。代价：若旧缓存来源或哈希不一致立即中止。

资源：FJ GPU0 4090，预检15MiB空闲；数据盘约16GiB、系统盘9.3GiB，满足10/5GiB预留。只允许一个GPU进程，screen=aris-relags。

待办：selector测试RED→GREEN；独立renderer与完整性断言；部署两格；下载结果与逐query因果分解；更新裁决。

Task 1 complete：新增selector4项测试先因模块缺失报错，补接口桩后4项均出现真实断言FAIL；实现后4/4通过。项目原测试15/15通过；run_rank_count.py语法编译通过。21条真实缓存均重放原两格selected完全一致；red apple两排名同序、pirate hat仅成员替换。

Final code review：独立只读review_rank_count子代理返回No findings；仍保留运行时原输入SHA、42个旧union及同union逐像素渲染重放检查，不用模型评审代替验证。

部署首次因远程experiments父目录不存在失败；创建项目内目录后显式上传五个小文件成功，无产物删除。已启动screen=aris-relags，输出runs/ovs_rank_count_20261024；不包含旧baseline GPU重跑。

Task 2 complete：两格最终mIoU 59.463316/59.264033%，各56query-view。42个旧Gaussian并集检查、157次同并集预测逐像素检查通过；16.89秒、峰值torch分配849.97MiB。screen自然退出，GPU恢复15MiB/0%，磁盘预留仍满足。

Task 3 complete：完整产物已下载review-stage/ovs_rank_count_20261024。CPU独立读取四格224张mask，验证像素数、旧文件哈希、同pool、正式图像权重及因子加和。主效应−3.823824/−4.023107pp，交互−2.733410pp；合计−10.580341pp。无重跑baseline，无追加扫描。

最终裁决：两格均未恢复原63.287140%，停止当前空间乘法及gap挽救，不扩场景。pirate hat投影原因仍未核明；停止条件已满足，不为挽救该实现追加GPU。代码和成果保留，不删除实验目录。
