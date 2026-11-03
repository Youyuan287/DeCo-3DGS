# 执行说明与结果解释

2026-11-01（Asia/Shanghai）。最终运行02:21:49–02:23:24，约95秒，四场景208条完整输出，无主指标缺失。RTX4090结束后显存15MiB、利用率0%。新文件约388KiB（汇总上传前），全部远端写入数据盘 `/root/rivermind-data/repro/deco-b123/evidence_attribution_20261101`。

## 结果

A原gateE .676011；G几何Hungarian绑定 .623639；I独立文本mask读出 .638746；J近最优几何竞争约束 .623639。J相对A -.052372，95%CI[-.080569,-.031447]；J相对I -.015107，95%CI[-.022422,-.006762]。J/G的最终IoU在208条全部相同，单候选渲染结果也全部相同。预注册门槛FAIL，未修改约束、容忍度或进行事后sweep。

## 实现和有效性检查

- A 208/208重渲染完全匹配归档，最大误差0。
- 纯几何最优的边应可参与J的可行匹配，J可选边属于I的子集；全部候选检查G_score<=J_score<=I_score。这是分数关系，不保证IoU关系。
- 同scene-prompt跨图的candidate IDs和原id_top3（包括查询内NaN填充）离线一致，故按prompt缓存未跨图错用fallback。
- 分族124/36/19/29与冻结值一致。
- 独立代码review未发现匹配、同root处理、NaN或render接口错误。
- 每视角root-mask一对一匹配，相互独立；先检查强制边是否保留至少90%最优总几何分，再选3个不同视角的最高文本证据。这可精确求解当前可行性模型，不是MILP近似，也没有额外跨视角pairwise约束。

## 约束实际做了什么

- 按query平均，可行mask中有多个root竞争的比例2.5596%；114/208条query该比例为0。
- 55/208条query的I/J top3 token集合存在变化，但包括重复提示词跨图；不当作55个独立实例。
- 按逐query-candidate计，共同fallback比例48.9517%。这些候选的G/I/J共同使用原id_top3，没有被删除。
- 去除重复评测图后，共366个prompt-candidate记录，182个三路共同有效。28个G/I分数不同，18个J/I分数不同，11个G/J分数不同；最终J/G选择结果仍全部相同。因此不能写成“J算法完全空操作”，应写“有局部分数变化，但没有产生相对G的最终结果收益”。
- 独立prompt计算的匹配累计约4.54秒，CSV中的solver_seconds按多视图重复记录，不能把它直接求和当真实总耗时。

## 能解释到哪一步

新证据场比原信号更稀疏（最多24个非评测视角、只用level3mask），且有较多fallback；G/I/J低于A的全部差值不能都归因于竞争约束。严格相同输入下可比较的新增约束效应是J-I=-.015107，方向为负。

当前querypool root竞争较少，且近最优匹配约束倾向于回到几何绑定，未出现超出G的有效改进。这个结果不支持继续把当前root-mask竞争作为论文贡献；也不证明所有潜在归属模型均无效。尤其不能由image-image余弦重叠直接推断“跨root mask归属竞争是主要瓶颈”。

实验只测试当前pool与冻结Gaussian支撑；无法解决池外recall，也不是EviSplat、DIGS或LaGa完整系统复现。源视角排除了评测图，但原gateE和fallback字段继承既有设置，所以不宣称整条管线完全无评测视角证据。

具体配置见PREREG.md，完整指标和CI见REPORT.md/summary.json，原始CSV、候选证据与选中mask ID见results/，代码hash见SHA256SUMS。
