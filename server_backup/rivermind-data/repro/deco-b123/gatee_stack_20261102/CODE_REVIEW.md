# 叠加实验实现检查

2026-11-02。fresh reviewer在完成评议前因服务额度失败；**外部评议REVIEW_UNAVAILABLE，以下为执行者local-only检查**，不冒充独立PASS。按ARIS experiment-bridge reviewer不可用时的本地检查fallback继续已授权实验。

已读原lerf_c_mask_gate.py、冻结J/J0预测、新生成器、renderer、launcher、汇总器。当前未发现阻止sanity的具体实现问题；是否复现原IoU必须以GPU sanity/full为准。

- 原gateE按归档NAG排序、GBM pred、area/mean_vis、id_top3逐view重构。全部208条main/C/M/E长度与真实gatee归档一致；e_c/e_m绝对误差均0。复用原缺失填补、zscore和lambda=1，不新增中心化评分。
- 生成器只读取pred、area、mean_vis、id_top3和冻结选择作决策。GT IoU不进入选择；family在路由固定后仅复制供汇总。
- 原C有9个、E有8个scene/prompt随view变化，保留208条预测而非压成67个。所有候选身份/顺序与原J/J0池一致，J/J0仍用其OOF选择。
- 独立按预测中的原filled id_top3与selection重算严格max证据判据，208条均一致：EJ切换12条，EJ0切换11条。二者所选集合严格等于相应父分支；不把E选择数量强制改成g_A。
- 修复唯一改变路由的query-view为plastic ladle：未修复J会触发替换，J0保留E。这只是预测事实，效果须真实渲染评价。此前人工初步消息的0次路由未被正式复算支持，不用于本轮结论。
- 接口修正：生成器route字段统一为0/1，并重新生成全部预测，与renderer/CSV/summarizer一致，未改路由公式。
- renderer逐条验证A/E/J/J0的旧选择或cardinality与归档，最终以旧IoU复现误差≤1e-6为条件；GT在所有选择和渲染固定后读取。
- 汇总器主比较EJ0−E，另报EJ0−EJ、EJ−E和J0−E；图内query→scene内image→四scene等权，图像cluster配对bootstrap B2000 seed20261102，不选择最优分支。
- 三个Python脚本AST解析通过。旧模型和主框架只读，launcher中所有新增输出、cache、TMP均在指定数据盘新目录，不安装依赖。需在服务器bash -n后先sanity再full。

限制：fresh代码评议未完成，不能记为same-family PASS或cross-family accepted；本地检查和baseline复现不证明方法有效或新颖。
