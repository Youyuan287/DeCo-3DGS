# ScanNet-10 DeCo V2 消融设计与运行命令

本文档供 Claude 审查本次重新认证的 ScanNet V2 消融。此前使用不同 NAG 的 0.498910 结果已作废。

## 1. 固定输入与基线

认证输入根目录：

    /root/verify58_20261029/full10

重新导出的 observation、M1 bank 和 V2 结果全部写入：

    /root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029

认证基线验收：

    cd /root/verify58_20261029
    /opt/conda/envs/vala/bin/python verify_expected.py

必须输出 ALL CHECKS PASSED；认证 K19 baseline 是 pooled 43.88%、scene macro 40.36%。旧链 scene0000 使用 386 个 root，认证链为 380 个；旧 NAG SHA 为 f94fd500...，认证 NAG SHA 为 7d08933d...，所以旧链全部撤回。

## 2. Staging 与模型一致性

认证 NAG 链接到隔离 staging，Gaussian 使用同一 XYZ 顺序的 PLY：

    BASE=/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029
    for s in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
      M="$BASE/$s/model"; mkdir -p "$M/point_cloud/iteration_0"
      ln -sfn "/root/rivermind-data/output/scannet/$s/point_cloud/iteration_0/point_cloud.ply" "$M/point_cloud/iteration_0/point_cloud.ply"
      ln -sfn "/root/verify58_20261029/full10/$s/model/sai_nag_3.0.pt" "$M/sai_nag_3.0.pt"
    done

实际运行还链接 cameras.json、cfg_args、input.ply、neighbor.pt、neighbor_new.pt 和 nag-l1.pt。审查时必须同时核对 Gaussian 数量、XYZ 顺序、NAG SHA 和 L3 root 数；不能只比较点数。

## 3. Frozen observation 与 M1 bank

每个场景的 observation 导出命令：

    PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
    REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
    S=scene0000_00
    "$PY" /root/rivermind-data/repro/deco-b123/m1_sas_export.py       --repo "$REPO" --source "/root/rivermind-data/datasets/scannet/$S"       --model "$BASE/$S/model" --nag "$BASE/$S/model/sai_nag_3.0.pt"       --outdir "$BASE/$S/export" --seed 1234 --iteration 0

导出必须保留原始 mask→Gaussian/ray tracing 关联，不调用 assign_label、everything_seg 或随机 NAG 重建。输出 observations_level3.npz、observation_semantic_sidecar.npz 和 EXPORT_PROVENANCE.json。

M1 bank 固定为 semantic grouping、max_modes=3、geo_weight=0.25：

    "$PY" /root/rivermind-data/repro/deco-b123/build_m1_sas_bank.py       --observations "$BASE/$S/export/observations_level3.npz"       --sidecar "$BASE/$S/export/observation_semantic_sidecar.npz"       --out "$BASE/$S/export/B1.npz" --mode semantic --max-modes 3 --geo-weight .25
    touch "$BASE/$S/export/BANK_READY"

导出门槛：features/weights 无 NaN/Inf；CSR offsets 单调且尾 offset 等于 ID 数；Gaussian ID 和 region ID 不越界；provenance 的 NAG SHA 与认证模型一致。五组必须共享同一 observation、B1.npz、B1.members.json 和 SHA。

## 4. 五组受控消融

固定官方 ScanNet K19、四模板文本平均、renorm=0、temperature=0.08、coefficient=0.25、max_modes=3、geo_weight=0.25、相同 NAG/Gaussian/GT：

| 组别 | M1 | M2 | M3 |
|---|---|---|---|
| DeCo off | 关闭 | 关闭 | 无辅助修正 |
| M1 Fixed | 开启 | 关闭 | 查询无关固定权重 |
| M1+M2 Fixed | 开启 | 开启 | 固定权重 × M2 reliability |
| M1+M3 Adaptive | 开启 | 关闭 | query-adaptive Softmax |
| M1+M2+M3 Adaptive | 开启 | 开启 | Softmax × M2 reliability |

共同读出为：

    advantage = max(mode_score - main_score, 0)
    delta = aggregate(advantage, mode_weight, reliability)
    new_score = main_score + 0.25 * delta

五组不改变 NAG、Gaussian、候选范围、gap 搜索、CLIP 特征、GT 或参数。M1 Fixed 与 M1+M2 Fixed 只改变 reliability；Adaptive 两组的唯一额外差异也是 M2 reliability。

## 5. 评测命令

单场景：

    SCANNET_SCENE="$S" SCANNET_C1=off SCANNET_C2=off     SCANNET_INPUT_ROOT="$BASE/$S/export"     SCANNET_ABLATION_ROOT="$BASE/ablation/v2_off_off/$S"     SCANNET_CODE=/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code     SCANNET_MODEL="$BASE/$S/model"       "$PY" "$BASE/scannet_c1c2_m1m2m3_evaluate_certified.py"

一次执行生成五组 SUMMARY.json、SCANNET_M3_PER_CLASS.csv、SCANNET_M3_WEIGHT_TRACE.csv、SCANNET_M3_CANDIDATE_CHANGE.csv 和 predictions_and_scores.npz。十场景后台命令：

    nohup /root/rivermind-data/repro/deco-b123/run_certified_v2_eval.sh       > "$BASE/batch_eval.log" 2>&1 &

每个场景必须先打印 baseline parity，再打印 METRICS；baseline 不一致时停止，不能解释 V2 差异。

## 6. 聚合口径

scene macro 是十个场景 mIoU 的等权平均：

    macro = mean(scene0000_mIoU, ..., scene0645_mIoU)

pooled 先对同名类别累加十场景 intersection/union，再计算每类 IoU 后平均：

    pooled = mean_c(sum_s inter[s,c] / sum_s union[s,c])

两者不能混用，也不能把不同 NAG 的逐场结果混合。

## 7. 认证结果

| 组别 | pooled mIoU | scene macro | pooled 相对 off | macro 相对 off |
|---|---:|---:|---:|---:|
| DeCo off | 43.8759% | 40.3575% | — | — |
| M1 Fixed | 43.8869% | 40.3847% | +0.0110 | +0.0272 |
| M1+M2 Fixed | 43.8874% | 40.3852% | +0.0115 | +0.0277 |
| M1+M3 Adaptive | 43.9457% | 40.4102% | +0.0582 | +0.0547 |
| M1+M2+M3 Adaptive | 43.8757% | 40.3783% | -0.0002 | +0.0208 |

M1+M3 Adaptive 是最高组，但提升仅约 0.058 pooled 百分点和 0.055 macro 百分点；M2 没有稳定独立贡献，不能宣传为显著改进。

## 8. Claude 审查清单

1. DeCo off 是否严格为 43.88/40.36，而非 0.498910。
2. 每场 NAG SHA、root 数、Gaussian 顺序是否一致。
3. 五组 observation、bank、GT 是否完全相同。
4. M1 Fixed 到 M1+M2 Fixed 是否只打开 M2 reliability。
5. Adaptive 两组是否只增加 M2 reliability。
6. 是否没有 GT 参与模式、候选或参数选择。
7. 是否没有温度、系数、候选范围或 C1/C2 扫描。
8. pooled 与 scene macro 是否按本文定义计算。
9. baseline parity 失败时是否停止报告 V2 改进。

本地数值报告：
C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro\SCANNET_V2_CERTIFIED_ABLATION_REPORT.md

服务器结果目录：
/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029/

