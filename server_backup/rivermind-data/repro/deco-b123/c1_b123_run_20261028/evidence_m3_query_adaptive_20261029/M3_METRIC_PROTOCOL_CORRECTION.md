# M3 评测统计口径修正

错误根因：旧汇总把 auxiliary evaluator 的 main_miou 当作 scene mIoU；该字段是对所有 query 的 main_iou 直接平均。Fixed/Adaptive 的 miou 则先按图像平均，二者不可直接相减。此前 DeCo 列及其净增益撤回，以本文件和修正后的 M3_QUERY_ADAPTIVE_ABLATION.md 为准。

定义：scene = mean_image(mean_query_in_image IoU)；flat = mean_all_queries IoU；四场景 macro 对各场 scene（或 flat）等权平均。另列208条合并 flat，避免混淆。

## scene mIoU

|scene|DeCo|Fixed|Adaptive|Fixed−DeCo|Adaptive−DeCo|Adaptive−Fixed|
|---|---:|---:|---:|---:|---:|---:|
|figurines|0.585868956|0.593810482|0.593810482|+0.007941526|+0.007941526|+0.000000000|
|ramen|0.518075607|0.517928185|0.529587432|-0.000147423|+0.011511825|+0.011659247|
|teatime|0.793896128|0.793896128|0.793896128|+0.000000000|+0.000000000|+0.000000000|
|waldo_kitchen|0.659209135|0.659209135|0.659209135|+0.000000000|+0.000000000|+0.000000000|
|scene等权macro|0.639262456|0.641210982|0.644125794|+0.001948526|+0.004863338|+0.002914812|

## flat mIoU

|scene|DeCo|Fixed|Adaptive|Fixed−DeCo|Adaptive−DeCo|Adaptive−Fixed|
|---|---:|---:|---:|---:|---:|---:|
|figurines|0.580394321|0.587926929|0.587926929|+0.007532607|+0.007532607|+0.000000000|
|ramen|0.511650366|0.511446887|0.522873141|-0.000203479|+0.011222775|+0.011426254|
|teatime|0.785208966|0.785208966|0.785208966|+0.000000000|+0.000000000|+0.000000000|
|waldo_kitchen|0.681239773|0.681239773|0.681239773|+0.000000000|+0.000000000|+0.000000000|
|scene等权macro|0.639623357|0.641455639|0.644312202|+0.001832282|+0.004688845|+0.002856563|

## 复算核验

四场景三臂的原始CSV均复算scene和flat，并与各自原始summary比对，12/12通过，误差小于1e-6（CSV保留六位IoU造成舍入）。最终表保留summary的全精度均值。每臂均208条，56/71/59/22；来源与复算明细见 M3_METRIC_RECOMPUTE.json。

- off 208-query pooled flat：0.625691644。
- fixed 208-query pooled flat：0.627650197。
- adaptive 208-query pooled flat：0.631550505。

修正后的scene宏平均：DeCo 0.639262456、Fixed 0.641210982、Adaptive 0.644125794。Fixed−DeCo +0.001948526；Adaptive−DeCo +0.004863338；Adaptive−Fixed +0.002914812。用户估计经原始文件独立复算成立。Adaptive增量集中于ramen的六个nori实例，属于单场景同类多视角证据，不能解释为跨场景普遍收益或六个独立语义类别。

未重新运行性能实验；仅修复统计脚本与报告。此前逐query IoU及paired counts不受宏平均错误影响。
