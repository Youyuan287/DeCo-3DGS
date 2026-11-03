# M1-SAS 四场景结果

状态：四个场景均已完成固定 seed=1234 的新 observation/sidecar、semantic weighted-Jaccard SAS bank 和统一 V2 评测。

| 场景 | scene mIoU | flat mIoU | 查询数 | 图像数 |
|---|---:|---:|---:|---:|
| figurines | 0.5938 | 0.5879 | 56 | 4 |
| ramen | 0.5179 | 0.5114 | 71 | 7 |
| teatime | 0.7939 | 0.7852 | 59 | 6 |
| waldo_kitchen | 0.6592 | 0.6812 | 22 | 5 |
| 四场景宏平均 | **0.6412** | **0.6414** | **208** | 22 |

固定协议：`center=image`, `text=canon`, `renorm=0`, `topk=5`, `level_until=3`, `filter_small=50`, `sel=gap`, `K=3`, `geo_weight=0.25`, `temperature=0.08`, M2 关闭，V2 query-adaptive readout。

重要限定：这些结果基于重新导出的固定 seed=1234 observation/sidecar，不是历史 V2 observation 的逐字节复原；旧 V2 bank 未混用，生产 NAG、Gaussian 和 ReLaGS 代码未修改。

逐查询文件保存在服务器：

`/root/rivermind-data/repro/deco-b123/m1_sas_20261029/eval/<scene>_sas/per_query.csv`
