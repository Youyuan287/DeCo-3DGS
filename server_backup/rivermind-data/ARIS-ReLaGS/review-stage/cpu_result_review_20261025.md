你是ARIS的研究裁决者。请基于以下CPU-only诊断结果和已核验文献，判断是否批准一次新的语义 bank 采集；不要把代理统计当作最终mIoU。

固定资产：LERF-OVS figurines；R0自生成NAG；299相机；已有2D seg_map；没有GT、没有新特征、没有GPU。诊断代码已修复A/B split泄漏，当前parent映射为nag[li+1]到nag[li]。

结果（20次随机A/B split）：
- level 1: TV中位数0.5000，ICC 0.3740；无足够parent pair。
- level 2: TV中位数0.48485，ICC 0.57790，M1相对M0 delta-R2 0.08133，delta-AUC 0.25054，precision lift 1.7893。
- level 3: TV中位数0.43119，ICC 0.41584，delta-R2 0.02174，delta-AUC 0.25140，precision lift 2.1245。
- 当前尚未实现D置换FDR、置信区间和leave-one-camera稳定性。

门槛原案：H/D ICC>=0.5；TV<0.3；D高于置换零模型FDR<0.05；delta-R2或AUC>=0.05且CI下限>0；precision lift>=1.5；leave-one-camera同号>=80%。

相关文献事实：OpenScene CVPR2023做多视角2D特征回投和3D蒸馏；econSG ICLR2025使用confidence-region regularization和融合后降维；CUA-O3D CVPR2025用多基础模型及确定性不确定性；VALA 3DV2026使用可见性门控和球面几何中值；CAGS 2025/2026强调跨视角粒度不一致和上下文传播。

请回答：
1. 当前结果是否足以批准一次bank采集？
2. 如果不能，最小下一步CPU诊断是什么，如何针对TV高但ICC中等的矛盾？
3. support bucketing是否已被上述工作覆盖，什么实质差异可能仍有价值？
4. 给出一个明确的continue/stop判决；不要建议无约束参数扫描或直接宣称CVPR。
