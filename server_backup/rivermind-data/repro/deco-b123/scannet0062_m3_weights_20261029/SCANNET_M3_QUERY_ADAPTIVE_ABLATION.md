# ScanNet scene0062_00：M3 Fixed / Adaptive 最新消融

范围：一个ScanNet场景、19类，冻结当前重训并剪枝后的模型与NAG。三组为DeCo off、Fixed normalized priors、Adaptive V2 Softmax，M2关闭。未实施新M1语义支持方案。

|方法|3D mIoU|mAcc|OA|mIoU−off|
|---|---:|---:|---:|---:|
|off|0.823382392|0.934632716|0.920759114|+0.000000000|
|fixed|0.822478220|0.934911479|0.920712872|-0.000904172|
|adaptive|0.823582591|0.936047801|0.921460457|+0.000200199|

Adaptive−Fixed：+0.001104371。该scene中有GT的类别数：6。mIoU/mAcc严格沿用发布metrics函数的有GT类别均值；所有19类TP/FP/FN另外完整保留，含只有FP的缺席类。此处无LERF的逐图query宏平均，也不使用旧805980点NAG缓存。

## 固定协议与基线认证

DeCo ScanNet发布scorer为 `score_scannet.py` 的 C1 renorm=0 + C2 both_lse（模板和词库边缘化），19类root argmax，再映射至每个Gaussian，按最近GT mesh点取得真值。它与LERF-OVS的canon/gap检索不同；为保持原始ScanNet行为，本实验没有把LERF检索硬套到ScanNet。三个arm严格共享此C1/C2及类别集。

首先用未修改的发布scorer在新GT缓存上生成off，再检查新增评测器off的完整metrics字典完全相等，门槛通过才执行辅助分支。主路径用原NAG生产ROFA描述符；辅助分组固定debiased、K=3、geo_weight=.25、B2=off，M2=off。Fixed与Adaptive只改变有效模式组合权重，advantage/资格逐元素相等；temperature=.08，修正系数=.25，全root绝对非负补充。

原发布main会附带打印其它文本变体；本实验预先固定both_lse，从未根据这些打印值选择文本设置。ScanNet的C2边缘化tau=.02是原发布常数；辅助模式Softmax温度=.08，二者用途不同，均未扫描。

## 逐类变化与机制

|GT存在类|off IoU|Fixed IoU|Adaptive IoU|Adaptive−Fixed|
|---|---:|---:|---:|---:|
|wall|0.886486792|0.885980774|0.885980774|+0.000000000|
|floor|0.940982369|0.942316806|0.942316806|+0.000000000|
|door|0.738180880|0.738180880|0.738180880|+0.000000000|
|counter|0.834975808|0.832294997|0.838921223|+0.006626226|
|toilet|0.942526067|0.938953427|0.938953427|+0.000000000|
|sink|0.597142436|0.597142436|0.597142436|+0.000000000|

Fixed与Adaptive实际改变2个root的类别：root40（485个Gaussian）table→counter，root32（259个Gaussian）refrigerator→counter。root40的485个GT全部为counter，因此Adaptive相对Fixed恢复485 TP、减少485 FN；counter FP保持2058不变。root32的259个GT全部为otherprop(40)，不在原19类GT评价范围；这次变化影响真实Gaussian和mask，但不进入发布mIoU。它不能被称作正确语义恢复，说明仅看指标会遗漏无效类别上的响应扩张。GT仅用于事后解释，从未参与评分或模式选择。

相对off，Adaptive仍有wall与toilet的IoU退化；floor/counter改善，door/sink不变。整体只比off高0.000200199，是微弱净收益，不能表述为稳定或显著改进。

## 输入与来源核验

剪枝前Gaussian 1158506，剪枝后 789225；NAG区域数 [14715, 1729, 257, 92]；训练视角 219；新冻结观测 4579。

剪枝后XYZ逐元素匹配30k点云的有序子集，且与语义PLY坐标序列一致；nag-l1与首层标签一致，saifine_labels与末层一致。每个训练view的_s/_f文件存在、维度匹配且SHA留存。真实GT从现有VALA/dataset/scannet.zip精确提取对应labels.ply，仅写入隔离目录；没有利用旧Gaussian GT标签按索引重用。

冻结导出只调用原ray tracing和投影，不调用assign_label/everything_seg，不训练、不生成CLIP、不改变NAG。使用已审计sink保存features/weights/view/region/member/visible CSR，校验offset单调、尾偏移、ID范围及每条CSR的root归属。脚本在运行前后核验原始模型/NAG/相机等SHA保持不变。

## 候选、Gaussian与mask变化

- off vs fixed：Gaussian集合改变的类别 6/19；实际渲染mask XOR非零 43/475；XOR总像素 376246。
- off vs adaptive：Gaussian集合改变的类别 7/19；实际渲染mask XOR非零 50/475；XOR总像素 394524。
- fixed vs adaptive：Gaussian集合改变的类别 3/19；实际渲染mask XOR非零 20/475；XOR总像素 18278。

ScanNet没有逐query的gap候选列表；候选变化CSV记录每root的19类竞争得分、类别argmax及Gaussian数，per-class CSV记录该类选中root集合与Gaussian增删。相同IoU不作为mask相等依据。

对原transforms_test的25个相机逐类、逐arm用相同完整Gaussian模型和renderer渲染，阈值0.5，直接计算mask XOR与SHA。渲染mask仅为诊断；本场未制造二维GT，TP/FP/FN均为真实三维标签比较。跨类别XOR合计可重复计数同一像素，应只理解为class-view诊断量。

原Blender loader将test_cam_infos置空，且readCamerasFromTransforms(load_sem=False)未初始化semantic；隔离评测器显式读取原transforms_test，并在函数局部副本仅初始化semantic=None，保留相机数学不变。前两次render-only适配的traceback保存在日志；三维评分未重跑，最后一次render replay完成。生产loader未修改。

独立复核：三臂mIoU/mAcc/OA从完整预测重新计算，误差<1e-12；57条类-方法TP/FP/FN记录全部一致；1425条mask XOR/SHA相等性比较一致；438个训练特征文件及冻结输入在最后一轮SHA复核中保持不变。详见VERIFICATION.json和FINAL_INPUT_CHECK.json。

## 文件及复现

服务器隔离目录：`/root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/`。

`SCANNET_M3_PER_CLASS.csv`：19类IoU/TP/FP/FN、root集合及Gaussian XOR；`SCANNET_M3_WEIGHT_TRACE.csv`：每类每root每模式权重/优势/修正；`SCANNET_M3_CANDIDATE_CHANGE.csv`：原始与两组竞争评分；`SCANNET_M3_MASK_XOR.csv`：25×19的真实渲染差异；`predictions_and_scores.npz`保留完整三维预测。`INPUT_MANIFEST.json`、`EXPORT_MANIFEST.json`、`SUMMARY.json`绑定输入SHA。

使用服务器 `/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3`。准备脚本已保存至隔离目录 `code/scannet_m3_prepare.py`；完整检查、冻结投影、bank构建顺序见脚本。准备输出已存在时拒绝覆盖；重复导出须新隔离目录。评测入口：

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3 /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/scannet_m3_evaluate.py
```

本场Adaptive优于Fixed，构成该场景下的独立权重增量；不能由单场结论宣称ScanNet-10整体收益。本次完成后停止，不扩展场景或新方案。
