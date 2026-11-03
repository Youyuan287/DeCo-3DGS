# 结果到主张：冻结模型的覆盖/回退直接贡献消融

日期：2026-11-02。fresh native Codex reviewer；`review_independence: same-family`，`acceptance_status: provisional`。实际模型与推理档位未由宿主暴露，记为 null。未发现本轮或项目根目录的 `EXPERIMENT_AUDIT.json`，`integrity_status: unavailable`；未运行完整实验审计，不影响本次有边界的结果判断。

**结论：`claim_supported: partial`，只支持具体冻结模型错误的解释和修复；“统一取消两列直接贡献带来稳健整体改善”的主张为 `no`。预注册 diagnostic benefit 门槛 FAIL，语义增量主张仍为 `no`。** 不将 partial 当作继续调参或方法成功的许可。

| 主口径：图内 query 均值→场景内图均值→四场景等权 | 结果 |
|---|---:|
| A：原 main/gap | 0.639262456 |
| S = J：原冻结标量/语义头 | 0.632376118 |
| S0 = J0：取消指定两列直接贡献 | 0.642884248 |

| 固定比较 | 差值 | 95% 图像 cluster 配对 CI | view 胜/负/平 |
|---|---:|---|---:|
| J0−J：唯一主比较 | +0.010508130 | [0, +0.031524390] | 1/0/207 |
| J0−A：必报原基线比较 | +0.003621792 | [−0.000073711, +0.010939086] | 1/1/206 |
| S0−S：机制控制 | +0.010508130 | [0, +0.031524390] | 1/0/207 |
| J0−S0：语义增量诊断 | 0 | [0, 0] | 0/0/208 |

`J0−J>0` 与无场景下降超过 .01 均满足，但 CI 下界为 0，不满足严格大于 0 的冻结判据。全部主比较收益来自 waldo_kitchen 的一个 query-view：plastic ladle 的 IoU 从 0 恢复到 0.840650407，与 A 相同。J0−J 只在 waldo_kitchen 有非零场景变化（+0.042032520），其他三场为 0。本轮没有观察到相对 J 的 IoU 损失，这不构成新场景无害性的保证。

S0/J0 各改变 4/67 个 scene-prompt 的集合，涉及 5/208 个 query-view：ramen/hand、teatime/hooves（两视图）、waldo_kitchen/plastic ladle、waldo_kitchen/spoon。除 plastic ladle 外，其余改变选择的视图干预前后 IoU 都是 0。S0 与 J0 的最终集合 67/67 相同、IoU 208/208 相同。故不能把修复归功于额外语义交互，也不能从分数改变推断有效的语义判别增量。

J0 对 A 的两个非零差是既有 bag 收益（figurines/frame_00195，+0.878075366）和 corn 损失（ramen/frame_00128，−0.014447413）；plastic ladle 不是超过 A 的来源。J0−A 的 CI 跨 0，不能声称超过原 A。历史 gateE=.6760107 仍是单列保留的最好结果；A 不称 gateE，本轮没有对 gateE 做配对显著性检验，也不与 THGS 的 156 条 oracle 或官方重建外数字混比。

本次实际支持的窄主张是：**在已探索四场、固定 A 候选池和 g_A 的冻结 S/J 模型中，取消变换后 log1p_cover_views 与 fallback_seg 两列的联合直接贡献，足以修复 plastic ladle 的一个既有错误；这一局部恢复没有产生超出标量控制的语义输出增量，也未达到预设稳健收益门槛。**

该结论不能分辨两列各自的独立作用，不能外推所有 metadata 有害、完整去除 missingness 通路、缺列重训的效果、CLIP 无信息或必须新增 mask。cover_frac、mwp_conf 与插补等路径仍保留。假说来自同一批已探索样本，原模型使用 LOSO 不会自动使此次事后提出的假说成为独立确认。图像 cluster bootstrap 条件于这四场，也不支持新场景泛化结论。

独立复核读取了 PREREG、make_predictions.py、predictions.jsonl、PREDICTION_CHECK、四份原始 CSV、SUMMARY、REPORT、CASE_RESULTS、full_driver.log，以及研究 contract。另检查了预处理、渲染和统计实现；独立从 CSV 重算五分支主均值与四组 CI（B=2000，seed=20261102），均与汇总一致。208 条为 56/71/59/22，覆盖 67 个 scene-prompt；A/S/J 渲染复现误差均为 0，无重复 query-view，各分支数量等于 g_A 且身份均在原池。只读加载原 S/J 八个折模型，按“原得分减两列直接贡献”重算全部预测，最大差 8.88e−16；最终选择一致。实现没有调用拟合、改预处理、改候选池或按 GT/family 分流。本次复核没有重训、增加候选组合、启动服务器作业或按结果调整列。

确定性 pre-check `.aris/evidence_precheck_metadata.json` 的 4 项数值均 verified；这只确认数值存在。`PREDICTION_CHECK` 的 PREDICTIONS_ONLY_NOT_RENDERED 是预测阶段快照，完整 CSV 与 full_driver.log 的四场 SCENE_DONE/ALL_DONE 已支持本轮渲染完成。

下一步：关闭这条固定 A 池、固定 g_A 的线性头及其两列修补支线，不继续删除列、扫阈值或重选 head。现有结果缺的是广泛收益和语义增量，不是再补同一单例的证据。若继续当前研究，回到此前已有互补候选证据、已固定但未执行的 A∪B × single/gap probe，以相同评分对照候选保留与输出规则；本轮并未检验该备选，也不预测其成功。若未来要提出通用 metadata 修复主张，需在不再用于选机制的新场景获得分散收益与语义对照证据，当前不据此自动追加实验。

`confidence: high`，限于上述固定实验事实及“不支持更强主张”的判断。
