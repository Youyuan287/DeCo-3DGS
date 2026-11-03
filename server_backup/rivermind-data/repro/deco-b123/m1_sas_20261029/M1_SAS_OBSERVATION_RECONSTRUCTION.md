# M1-SAS 原始 observation 重建验收

状态：**NOT RUN / BLOCKED**

阻断原因：历史 observation 没有二维 trace label、trace confidence、mask beta 或 feature-level/seed；当前 `merge_proj.py` 也未保存随机编码矩阵。任何重新 trace 的结果都无法证明对应历史 observation，因此不能通过 `reconstructed_features ≈ original_features` 的验收。

已保留原始 observation 哈希与输入清单于 `M1_SAS_SOURCE_SNAPSHOT.json`。没有伪造重建误差、CSR 语义支持或 SAS 对比结果。
