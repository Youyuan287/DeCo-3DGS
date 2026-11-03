# Ablation causal summary

1. **V3下降的来源。** 两个因素均为负：范围主效应 -0.007202，相对增益主效应 -0.005792，且负交互为 -0.005216（scene-mIoU macro）。absolute 下 selected−all=-0.004594；all 范围下 relative−absolute=-0.003184；selected 范围下 relative−absolute=-0.008400。所以 V3 的下降不是单一原因：范围限制是较大的主效应，relative 截断也独立有害，二者合用最差。
2. **M1 场景去偏 grouping。** 在严格的 raw/debiased 对照中，debiased−raw 的四场景 macro 为 +0.000269，仅 figurines 为 +0.001076，其余三场为 0。这是一个独立但极弱、没有跨场景稳定性的正向信号；不能据此做强性能主张。
3. **M2。** 既有 semantic 与 reliable 的逐查询 IoU 完全一致；在此前审计中 M2 未产生可观察的候选或 mask 改变。因此当前数据没有 M2 的独立可观察贡献证据。
4. **M3 查询自适应。** 没有独立支持：all+relative 比 all+absolute -0.003184，selected+relative 比 selected+absolute -0.008400，均为负。它还未复现 nori 的背景删除或 tesla 恢复，并在 ottolenghi 造成漏检。
5. **保留/删除。** 暂时保留 all+absolute 的原 V2 辅助读出作为唯一在本协议下有净提升的 M3 形式，但同时保留 corn 的背景扩张反例；删除 selected 范围限制与相对均值截断作为性能模块。M2 关闭/删除为性能主张；M1 debias 保留为待进一步证实的微弱工程选择，而非已证实收益。此处停止，不启动 V4 或空间路由。
