# 同一底座下与历史最好gateE的补充配对

主实验完成后补充的历史参照，不替代PREREG的J0−J主比较，也不用于选择干预列或分支。

直接读取服务器 `/root/rivermind-data/repro/deco-b123/gatee_20261031/*_cmaskgate_image.csv` 的明确字段 **gatee_iou**，保存副本于anchor_gatee/。旧本地cmaskgate2目录的gate_iou不是gateE，未将其混用。

四场208条按scene/image/prompt逐条对齐，历史文件的main_iou与本次A_iou最大误差0。历史gatee_iou按相同图→scene宏平均重算为 **.6760106935634101**。

J0=.6428842481395194；J0−gateE=−.03312644542389078，配对图级bootstrap95%CI[−.04546356386041218,−.019991497213018]，9胜29负170平，B2000 seed20261102。完整数值和逐条非零差见ANCHOR_COMPARISON.json。

该补充进一步说明本轮局部恢复未达到当前最好方法。它只比较相同released底座208条，不与THGS另批oracle或重建/官方64.01、63.93混比。fresh result-to-claim先完成了本轮主协议的判断；此补充统计由执行者在其后计算，未冒充该评议的原始范围。
