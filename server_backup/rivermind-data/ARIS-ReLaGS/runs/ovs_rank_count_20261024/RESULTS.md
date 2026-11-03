# LERF-OVS figurines：排名与数量的两格补全

日期2026-10-24，Asia/Shanghai。用户在研究审查后明确批准实施。**两组已完成，结论是停止当前空间乘法与gap挽救路线，不扩场景、不扫权重。** 此否决针对实际实现，不否定所有空间证据或整个领域的集合选择问题。

## 四格最终结果

同一作者PT、pre-gap池、原语义分数、相机及原renderer；正式口径为图内query平均→四张图像等权平均。56个query-view，21个不同query，每图17/13/11/15条。不是LERF-Mask、不是自生成PT、不是四场景均值。

| 配置 | 排名来源 | 选择数量来源 | mIoU % | 相对baseline pp | 空mask |
|---|---|---|---:|---:|---:|
| D00 原baseline，复用 | 原语义 | 原gap | 63.2871 | 0 | 2 |
| D10 本轮新增 | 空间分数 | 原gap数量 | 59.4633 | −3.8238 | 1 |
| D01 本轮新增 | 原语义 | 空间gap数量 | 59.2640 | −4.0231 | 3 |
| D11 旧空间组，复用 | 空间分数 | 空间gap | 52.7068 | −10.5803 | 2 |

**直接裁决**：恢复原选择数量仍低于baseline，恢复原排序也仍低于baseline；“只是gap改坏数量，空间信号其实有效”的挽救解释不成立。没有任何一格形成净提分，不能按query事后挑选四格最好值。

以原baseline为参考的加性分解：

\[
-10.58034 = \underbrace{-3.82382}_{D10-D00}+\underbrace{-4.02311}_{D01-D00}+\underbrace{-2.73341}_{D11-D10-D01+D00}\quad\mathrm{pp}.
\]

最后一项是确定性干预的交互项，不是置信区间或独立因果路径占比；不把绝对值归一化成“原因解释百分比”。只能针对固定P及当前r1进行解释。

## 哪些样本改变

| 对照 | 改善/退化/不变 query-view | Gaussian并集改变 | 救回空/新增空 | 双方非空对总体贡献 pp |
|---|---|---:|---|---:|
| D10−D00 | 4 / 6 / 46 | 15 | 1 / 0 | −4.9378 |
| D01−D00 | 3 / 7 / 46 | 21 | 0 / 1 | −4.0231 |

- **pirate hat**：D10保留错误L2/70，仍损失−6.5953pp；D01恢复原L2/75，四图与baseline一致。是成员更换，不是数量变化。
- **red apple**：D10仅保留原正确L3/43，恢复baseline；D01仍选L3/43、L2/69、L3/24，损失−3.5657pp。是范围扩大，而非首名不正确。
- **jake**：D10无收益，D01贡献+1.1761pp。缩短原选择在部分样本有用，但不足以抵消其他退化。
- **bag**：D10救回空预测、贡献+1.1140pp；D01没有该收益。不能将救回空输出泛称均匀语义增强。

收益与损失依旧集中在少数query；不能由负的场景均值宣称所有query都退化，也不能删去pirate hat后宣布方法有效。逐query贡献见analysis/query_contributions.csv，逐query-view四格见analysis/per_query_factorial.csv。

敏感性检查：删除pirate hat并在各图重新归一化，D10−D00变为+3.0194pp，说明其负增益高度集中；删除red apple后，D01−D00仍为−0.5203pp。它们是事后leave-one-query-out诊断，不是可部署规则，不能按GT识别失败query并回退，也不作为全量提分证据。

## 实际执行与完整性

- 只渲染两组新组合，共112张PNG；没有重跑baseline或D11，没有重提特征/SAM/训练/优化NAG。
- 原manifest内全部源码、模型及缓存SHA逐项验证；本地旧per_query和diagnostic文件与本轮manifest引用的远端哈希一致。
- 21query两旧格的42个Gaussian并集哈希完全相同。新组与任一旧组并集相同的157次比较，预测mask逐像素完全一致；是渲染重放检查，不是157个独立样本。
- 本地重新读取四格224张PNG，pred_px逐一匹配。56条的pool哈希及GT像素数四格一致；独立重算正式加权均值和交互加和通过。
- 新selector测试4/4、项目测试15/15在本地和服务器通过；独立只读代码review无发现。
- 两组合计16.89秒，torch峰值已分配显存849.97MiB；包含输入校验/加载在内。不代表完整特征构建成本，也不是GPU实际保留显存峰值。
- screen=aris-relags正常退出；末次GPU15MiB/0%空闲；数据盘16GiB、系统盘9.3GiB。旧模型/结果未删除或覆盖。

## 代码、日志与手动复现

FJ：`ssh root@fj01-ssh.gpuhome.cc -p 30253`。

代码：`/root/rivermind-data/ARIS-ReLaGS/experiments/rank_count_20261024/run_rank_count.py`。

日志和结果：`/root/rivermind-data/ARIS-ReLaGS/runs/ovs_rank_count_20261024/{run.log,full/FINAL_STATUS.json,full/per_query.csv}`。本地本目录保存完整副本及112张新mask。

```bash
tail -n 30 /root/rivermind-data/ARIS-ReLaGS/runs/ovs_rank_count_20261024/run.log
cat /root/rivermind-data/ARIS-ReLaGS/runs/ovs_rank_count_20261024/full/FINAL_STATUS.json
```

实际运行入口是`bash /root/rivermind-data/ARIS-ReLaGS/experiments/rank_count_20261024/run.sh`，已存在输出会拒绝覆盖。若手动重放，先确保GPU空闲，并使用新目录：

```bash
R=/root/rivermind-data
A=$R/ARIS-ReLaGS
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$R/thgs_mamba_root/envs/thgs HF_HUB_OFFLINE=1
export PATH=$CUDA_HOME/bin:$PATH
export RELAGS_REPO=$R/ReLaGS_Reproduction/ReLaGS
export PYTHONPATH=$RELAGS_REPO/ext:$RELAGS_REPO:$R/ReLaGS_Reproduction/ext:$R/ReLaGS_Reproduction:$A/scripts
cd "$A"
$R/ReLaGS_Reproduction/venv/bin/python -u experiments/rank_count_20261024/run_rank_count.py \
  --prior "$A/runs/ovs_support_20261024/full" \
  --out "$A/runs/ovs_rank_count_manual_$(date +%Y%m%d_%H%M%S)"
```

脚本仅消费既有分数和pool，按原规则确定k0/k1，不读取GT决定选择；全部预测冻结后才用polygon评价。原并列gap、全并列/单节点空行为未修复。不是重新编码query或重新搜索候选。

## 停止与未解决事项

当前空间重评分没有作为下一轮方法的有效起点，不迁入其他场景。若未来有真正不同、无GT且独立于原CLIP偏差的新证据，须另立假设，不从此处继续扫融合权重。

pirate hat零支持IoU究竟来自语义误认、视角差异还是投影问题，独立投影可视化仍未完成。本次两格已满足停止条件，因此不为挽救实现自动追加GPU诊断。这个未定原因保留，不编造因果结论。
