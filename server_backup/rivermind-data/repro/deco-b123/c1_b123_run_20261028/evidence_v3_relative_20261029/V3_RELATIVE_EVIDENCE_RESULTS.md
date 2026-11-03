# M3 相对证据 V3 固定四场景结果

|scene|queries|DeCo off|V2 semantic|V3 relative|V3−off|V3−V2|
|---|---:|---:|---:|---:|---:|---:|
|figurines|56|0.585869|0.593810|0.585869|+0.000000|-0.007942|
|ramen|71|0.518076|0.529587|0.518076|+0.000000|-0.011512|
|teatime|59|0.793896|0.793896|0.793896|+0.000000|+0.000000|
|waldo_kitchen|22|0.659209|0.659209|0.626687|-0.032522|-0.032522|
|scene macro|208|0.639262|0.644126|0.631132|-0.008131|-0.012994|


V3相对off的改善/退化/不变查询数：(0, 1, 207)；相对V2：(2, 11, 195)。scene mIoU按图像内查询平均后再跨图像平均，flat指标与其分开；CSV有208行，不混淆两种口径。

表格使用原始summary中的全精度scene mIoU；逐query CSV沿用evaluator的六位小数，反算均值可能在最后一位有舍入差异。

V3相对off的查询IoU正向总量 0.000000，负向总量 0.975669。背景FP的查询正增量合计 0，负增量合计 -271；对应V2为 577330 和 -58750。这些是每查询净FP变化的正负汇总，不是逐像素背景集合新增/删除总量。

## 唯一预注册规则

同一查询先完整运行原始DeCo，固定 S = 原始最终候选的root_id去重集合（这些root处于同一level3；即便输出为level2 child，也归于其父root进行竞争）。M2使用现有固定consistency可靠性；计算原版V2的 a_r。S内 correction_r=0.25*max(a_r-mean_S(a),0)，S外 correction=0。单root或空S回退原始分数；其他层辅助修正恒为0，保留原始canon分数。随后仍运行原始完整top-root/root-child/gap检索。S只由原始主路径定义，绝不随新分数或GT更新。

截断意味着不同root获得不同修正，pair差变化为0.25*(positive_residual_i-positive_residual_j)，可改变root排序、相邻gap及root-child比较；若二者均未过均值则无变化。这不是全体减同一常数。缓存预检75条query在S内产生非零pair变化后才启动评测。S外候选仍可能因gap改变而间接进入/退出，算法没有人为固定最终输出集合。

V2对照采用已认证semantic（M1+M3）；V3按要求固定使用当前M2。因此V3−V2是读出方案的总体对比，不能把全部差值单独归因于相对运算。审计同时运行原有reliable分支，候选/IoU与semantic核对，相关输出保留。

## 核心案例与空间证据

见 `V2_KEY_MECHANISMS.csv` 与 `V3_CANDIDATE_CHANGES.csv`：逐查询列出nori/corn/tesla的三组IoU、root/child候选、分数、辅助修正、真实Gaussian坐标范围、Gaussian增删及完整mask XOR。新增/减少背景FP是对GT的离线评测统计，从未参与S、a或门控规则。

缓存的具体竞争量说明了该固定规则的限制：nori 的 S=[5,16,92]，固定M2后的 a 分别约0.0480131、0.00197694、0.0270140，V3修正分别0.00558627、0、0.000336506；分差确实改变，但不足以复现V2的最终gap切换。corn 的 S=[5,187,190]，root12不在S，V3不能直接提高它的分数，因而避免V2的root12空间扩张。tesla door handle 的 S=[29,74]，root26不在S，所以也失去V2对正确child194的恢复。这些都是同一集合规则的后果，未使用类别特定判断。

这份负结果只否定当前“原始已选root集合 + 相对均值截断”的具体M3方案；不能据此否定M1辅助模式存在有用信息，也不能把对corn的改善当作总体收益。原始候选集合限制同时屏蔽了有害扩张和有益替换。

唯一相对off的IoU退化是waldo_kitchen/frame_00053/ottolenghi：0.975669→0。原始root集合为269、218、264、4；V3将root218提高至0.618745685，root4提高至0.603403871，最大gap截断最终只保留218与269，移除root264和root4，共12198个Gaussians。17033个预测像素全部消失，其中16762为GT交集、271为FP。S内相对排序/间隔变化是真实的，却不反映当前视角上的目标支持；仅有候选竞争并不能保证偏离主路径更可靠。

## 结论与停止

V3宏平均低于V2，保留负结果，不扫描参数。 当前实验完成后停止，不进入V4，不改变M1/M2/ROFA/C1/C2或生产输入。

## 复现

远程启动入口 `run_relative_v3.py`，读出 `evidence_readout_v3.py`，独立评测器 `eval_lerf_ovs_v3.py`。输出远程目录 `/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/evidence_v3_relative_20261029`。每组manifest记录完整argv及NAG/bank/observation/members SHA256，运行前后校验一致。离线环境变量HF_HUB_OFFLINE=1。未重建特征、NAG或bank。
