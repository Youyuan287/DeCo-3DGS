# 跨领域迁移二次质疑：现在有实时原文/代码证据

本次仅研究决策，无GPU授权，不要回到pirate hat个案诊断作为主线。你没有浏览工具，以下由Codex实时核对。当前2026-10-24，全部成绩不跨论文直接比较。

## 新核对的源领域与重叠

- CAT-Seg（CVPR2024）：在图像×文本代价体上空间/类别聚合，训练并微调CLIP；不是把一个相似度做缩放。作者github.com/cvlab-kaist/CAT-Seg。
- PCA-Seg（CVPR2026），arxiv2603.17520：并行空间/类别聚合及EPL/FOD。实际作者代码cat_seg/modeling/transformer/model.py的AggregatorLayer.forward分别算x_spatial_first和x_class_first再DualFeatureMoE；有cosine_disentanglement_loss。需训练，不可称免费即插即用。单靠并行聚合/正交已经被覆盖。
- LPOSS（CVPR2025），arxiv2503.19777：CLIP提供跨模态语义，DINO提供关联，再传播patch/pixel预测；训练free但额外DINO/稠密特征不是free。models/lposs/lposs.py.forward确实输出dino_feats、clip_feats、clf，不是先把DINO向量与CLIP向量相加。
- PEARL（CVPR2026），arxiv2603.21528：修CLIP末层Q/K几何＋text-aware Laplacian预测传播。pearl/prop.py:TLP.forward构造(Cw+L)F=Cw*logits并CG求解。但当前源码三处einsum('cc,bcij->bcij',S,p)直接只用S对角项；因此不能未经检查就声称代码实现完整类间传播，也不能修一处实现当新研究。
- Mask-Adapter（CVPR2025），arxiv2412.04533：mask特征好不等于识别好；用mask-conditioned semantic activation map替代binary mask pooling，需训练。mask_adapter_head.py:MASKAdapterHead.forward把dense feature与mask编码相加，经ConvNeXt块输出多个pooling map；head仅匹配base/large ConvNeXt640/768，不能直接接当前ViT-B/16的512维缓存。
- CorrCLIPv2，arxiv2610.24757（2026-10-17；作者仓库称IJCV）：用VFM索引从无标注图像bank检索对象、用CLIP特征作值。corrclip_segmentor.py:forward_slide/_faiss_search/get_support_clip_logits可读；已有跨图抗context bias，不可把多视角memory bank当新。
- Gaussian Grouping（ECCV2024）已用DEVA追踪mask；MaskClustering已有view consensus graph；PairGS（arxiv2607.01140）已有稀疏Gaussian relation graph+TreeDBSCAN层级，图结构不是新。
- AG2aussian（ICCV2025）已有anchor graph Laplacian feature propagation；Splat Feature Solver（ICLR2026，arxiv2508.12216）已有稀疏线性逆问题lifting+正则，不能把rendering matrix拟合当新。
- Multi-View Foundation Models（arxiv2512.15708）明确给CLIP/DINO/SAM加入3D-aware intermediate attention以匹配跨视角特征。因此“3D几何约束CLIP token attention”本身也有直接近邻。
- OpenInsGaussian（ICCV2025 Findings）已有mask context feature extraction和attention跨视角融合，作者2026代码还有mix_ctx新预处理；不能只据原文否定新版本。

## 纠正上轮建议

固定r1失败不说明候选生成丢失，也不能推定复合query塌缩；正确候选在部分反例已存在。FCE/BSHRINK改变特征/聚合，不是全部都是候选后处理。

DeepSeek上轮循环一致式的数据项不含优化变量y，不能拿它作可求解方法。已知相机几何的同Gaussian投影循环本身是构造事实，不能当新增语义证据；图割/ICM不可不检查目标就套用。DETR Hungarian主要是训练匹配，且文字queries可同义、包含、多个实例，不能强加文本与实例一对一。

GLM上轮将SAN/ODISE等笼统称查询逻辑分解的来源无原文依据，不能使用；语义相近不等于能推复杂逻辑，也不能把51次零支持解释成生成失败。

## 二轮任务

现在最多比较这三类，不预设任何一类创新成立：
A. 在语义特征生成/读取阶段改变patch交互或mask-conditioned readout，而非更换骨干（最近邻：Mask-Adapter/OpenInsGaussian/MVFM）。
B. 将2D dense cost/logit空间的联合推理迁成可见性约束的3D查询场，而不是先平均embedding再Top-k（最近邻：LPOSS/PEARL/SFS/AG2aussian/PairGS/EviSplat）。
C. 处理开放文本查询的粒度与组合语义，必须有准确来源和真实query分布，不依靠“属性embedding塌缩”推断（最近邻：PCA-Seg/Search3D/ReLaGS关系任务）。

请选择唯一值得进一步研究的**问题**（不是已成立的方法），或明确全部新颖性不足；写出具体尚未被近邻覆盖的假设、为什么不是给旧方法换名、一项最小直接验证及最可能反例。若给公式，变量必须真正参与目标，数据/正则角色清楚，不写一个普通Laplacian就宣称创新。不用任意pp门槛，不执行或授权实验。

正文≤1500字。顾问认可不能替代原文查新或最终指标验证。
