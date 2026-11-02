# 13 — ScanNet 评测协议逐行比对（对着开源代码）

方法：`codeload.github.com` 下 tarball（GitHub API 额度被共享 IP 耗尽），本地只读不改。

| 仓库 | 状态 | 与本仓库的关系 |
|---|---|---|
| [yanmin-wu/OpenGaussian](https://github.com/yanmin-wu/OpenGaussian) | 有完整 ScanNet 评测 `scripts/eval_scannet.py` | **协议相同** |
| [dfki-av/ReLaGS](https://github.com/dfki-av/ReLaGS) | 只有推理管线，**仓库内无 mIoU 评测脚本** | 表 4 数字来源不明 |
| [Atrovast/THGS](https://github.com/Atrovast/THGS) | 只有 LERF/3D-OVS 的 2D mask 评测，**无 ScanNet 评测** | 无法核对 |
| VALA | GitHub 搜索未定位到官方仓库 | 不猜测 |

## 1. 逐项比对（OpenGaussian vs 我们）

| 项 | OpenGaussian `eval_scannet.py` | 我们 `eval_scannet_3d.py` | 一致 |
|---|---|---|---|
| 场景 | 10 个：0000/0062/0070/0097/0140/0200/0347/0400/0590/0645 | 完全相同 | ✅ |
| 19 类 id | `[1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36]` | 完全相同 | ✅ |
| 15 / 10 类子集 | 脚本内注释给出 | `SUBSETS[15]`/`[10]` 完全相同 | ✅ |
| 聚合 | 仅统计 GT 中出现过的类，`valid_gt_classes` 求均值 | `present = [i for i in 1..k if tot[i] > 0]` | ✅ |
| unlabeled | `pred[gt == 0] = 0` | `p[g == 0] = 0` | ✅ |
| mAcc | 仅 present 类的类别准确率均值 | 同 | ✅ |
| **低不透明度点** | `sigmoid(opacity) < 0.1` 的点 **GT 置 0（忽略）** | `load_gaussians(model, 0)`，`opacity_min=0.0`，**keep=全部** | ❌ |
| 预测粒度 | 每 cluster（k≈320）argmax | 每 Gaussian（NAG）argmax | 方法差异，非评测差异 |
| 文本编码器 | CLIP ViT-B/16 512（LangSplat 提取） | OpenCLIP ViT-B-16 laion2b，自定义 prompt | ≈ 同架构，提示词不同 |

**唯一实质评测差异：低不透明度过滤。** OpenGaussian 把 `σ(opacity)<0.1` 的高斯从 GT 里剔除，
我们保留全部。这批点多是边界浮点，影响可正可负，但**必须对齐**，否则数字不能并列。

## 2. 一个重要纠正：ReLaGS 表 4 的数字不可比，原因不是"协议细节"

本仓库旧文档（§3.4.1）把 ReLaGS 32.35 与我们 50.18 的 18 分差距归因为"协议细节差异"。**这个归因不准确。**

证据：`dfki-av/ReLaGS/scripts/run_scannet.sh` 里两处关键参数——

```
python merge_proj.py ... --vlm_type openseg
python predict_3d_scene_graph.py ... --vlm_type openseg
```

ReLaGS 的 ScanNet 管线用的是 **OpenSeg** 文本/视觉主干，**不是 CLIP**。
而我们全部 ScanNet 实验用 `ViT-B-16 / laion2b_s34b_b88k`（CLIP）。

→ **32.35 与 50.18 是不同主干下的数字，不能同表。** 这比"协议细节"严重得多。

## 3. 结论：谁能比、谁不能比

| 对象 | 能否同表 | 前提 |
|---|---|---|
| OpenGaussian（24.73 @19类） | **可以**（同协议、同 CLIP 家族） | 先加 `σ(opacity)<0.1` 过滤重跑；表头写明提示词模板 |
| VALA / THGS | **待定** | 需先确认它们的评测脚本与主干（THGS 仓库内没有 ScanNet 评测） |
| ReLaGS 表 4 | **不可以** | 主干不同（OpenSeg vs CLIP）；除非我们在 OpenSeg 场上复现它的数 |

## 4. 必须做的三步（按顺序）

1. **加 opacity 过滤**：把 `load_gaussians(..., opacity_min=0.1)` 接进评测，重跑 raw / zR / zR+E0 三行。
   预期变化幅度未知；不变说明该过滤无影响，变了就说明旧数字需要重报。
2. **核对提示词**：OpenGaussian 的 `text_features.json` 来自 LangSplat 的模板；我们的 prompt 是自定义。
   同一类别名不同模板会带来系统性偏移，必须在表头注明。
3. **按主干分表**：CLIP 表与 OpenSeg 表分开列，绝不混排。

## 5. 对论文的直接含义

- 与 OpenGaussian 同表是**可行**的，这是目前唯一可核实的对手；
- 与 ReLaGS 同表**不可行**，除非补一个 OpenSeg 实验臂；
- 因此"我们在 ScanNet 上比 ReLaGS 高 18 分"这句话**不能写**。
