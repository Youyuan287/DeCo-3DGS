# 边界损失：ReLaGS 2D-3D-2D 中被忽视的关键难点

日期：2026-11-01
结论：**边界/掩膜保真度是一个独立于“身份/候选”的瓶颈；即使 GT 最好的 2D mask，
BIoU 也远不到 95%，经过 2D→3D→2D 后进一步下降。**

---

## 1. ReLaGS 的 2D-3D-2D 到底做了什么

### 1.1 2D → 3D（merge_proj）

- 每个视角渲染 `weight`，取 `seg_map[level]` 的 SAM mask 标签与 mask 特征；
- 对每个 superpoint，计算它落在各 mask 内的比例 `ratio`；
- `RATIO_THRESHOLD = 0.3`：比例低于 0.3 的 mask 贡献被丢弃；
- `sp_feat = sp_mask_mat @ view_level_feature`：**一个 superpoint 的逐视角特征是多张 mask 特征的加权混合**；
- 用可见性比例累积多视角特征；ROFA 去 outlier 后平均；
- 层级区域生长（`thres_connect 0.9,0.7,0.7`、`thres_merge 20`）把相邻 superpoint 合并成更大区域。

**边界在哪里丢的**：
1. **mask 特征是整张 mask 的池化向量**，没有边界/距离信息；
2. ratio < 0.3 的 mask 被丢弃 → **部分覆盖的薄结构、边界区域直接消失**；
3. superpoint 是几何/图划分的结果，边界不一定和 mask 边界对齐；
4. 多张 mask 加权混合 → 物体边界上的 superpoint 会混入邻居/背景语义；
5. 区域生长按特征相似度合并，边界会被进一步“抹平”。

### 1.2 3D → 2D（推理渲染）

- 文本打分选中区域后，把该区域的 Gaussian 语义设为 1，渲染 `semantics > 0.5`；
- 预测边界由 **Gaussian splat 的覆盖/不透明度/尺度**决定，而不是原始 2D mask 边界；
- Gaussian 是各向异性椭圆，边界会膨胀/收缩；薄结构、孔洞、细边界容易丢；
- 没有使用任何 2D mask 的边界先验做约束。

**结论**：ReLaGS 的 2D-3D-2D 是一个**语义+几何的双重有损过程**：
- 语义损失：mask 特征混合/ROFA 选共享语义；
- mask/边界损失：ratio 阈值、superpoint 边界、Gaussian splat 边界。

---

## 2. 用 GT 量化边界上限（Boundary IoU）

定义：`B(mask)` = 距 mask 边界 d 像素内的 band；`BIoU = |B(pred)∩B(gt)| / |B(pred)∪B(gt)|`。
d = 2% 和 1% 图像对角线，两种带宽。

### 2.1 GT 最好的 2D mask

| 带宽 | mean BIoU | median | max | ≥0.95 | ≥0.90 |
|---|---:|---:|---:|---:|---:|
| 2% diag | **0.678** | 0.870 | 0.995 | **41.8%** | 48.1% |
| 1% diag | **0.633** | 0.787 | 0.988 | **21.6%** | 41.3% |

**即使 GT 选出的最佳 2D SAM mask，也只有 42%（2% band）/ 22%（1% band）的查询 BIoU ≥0.95。**
“最好的 mask 能达到 95%”在 LERF 上不成立。

### 2.2 2D GT-best mask 提升到 3D 再渲染

| 带宽 | 2D BIoU | lifted BIoU | 损失 |
|---|---:|---:|---:|
| 2% diag | 0.678 | **0.636** | −0.042 |
| 1% diag | 0.633 | **0.566** | −0.067 |

lifted 的 ≥0.95 比例：2% band 12.5%，1% band **2.4%**。
**2D→3D→2D 还会再损失 4–7 个 BIoU 点；1% band 下几乎不可能到 95%。**

### 2.3 分族（2% band）

| 族 | 2D BIoU | lifted BIoU |
|---|---:|---:|
| found | 0.7595 | 0.7146 |
| rank2 | 0.5364 | 0.5132 |
| deep | 0.8714 | 0.8355 |
| recall | 0.3818 | 0.3229 |

IoU 与 BIoU 的相关性约 0.95–0.97，但**绝对边界质量远低于 IoU 给人的印象**。

---

## 3. 这为什么是“真正难点”，而不是补丁

之前我们把问题归为“候选/实例生成”（recall 0.027、deep 0.66），这只是一半。
另一条独立的损失是 **2D-3D-2D 的掩膜/边界保真度**：

1. **它限制了任何方法的 IoU 上限**：即使候选选对，渲染边界仍然受 Gaussian splat + mask 池化限制。
2. **它不被任何现有模块优化**：merge_proj 没有边界项；推理没有 2D 边界先验；评测只看 IoU。
3. **它解释了很多“选对了但分数不高”的查询**：found 已经 0.831，但 BIoU 只有 0.715；
   rank2/deep 的边界更差。
4. **它是可观测量、可优化量**：BIoU 有明确数学定义，可做损失/选择准则/精修目标。

一句话：**3D-OVS 有两条独立损失：身份/候选损失 + 边界/掩膜损失。**
之前只盯第一条；第二条同样大，而且更“基础”。

---

## 4. 可以怎么做（方法层面）

### 4.1 边界感知的 2D→3D 投影

- **边界带排除/降权**：mask 边界 d 像素内不参与特征池化（用距离变换）；
- **部分 mask 不直接丢弃**：ratio < 0.3 的 mask 用软权重保留，避免薄结构消失；
- **边界感知的 superpoint 分配**：按距离/边界概率加权，而不是整块 ratio；
- **多视角 mask 边界一致性**：用多视角 boundary 投票约束 superpoint 边界。

### 4.2 边界感知的 3D→2D 渲染/精修

- **2D mask 作为 trimap**：把选中候选的 3D 支撑投影到查询视角，与 2D mask 边界做
  graph cut / matting / level set，精修边界；
- **边界带高斯重加权**：边界附近的 Gaussian 用 2D mask 的 inside/outside 概率调整；
- **多视角边界共识**：多个视角的 mask 边界投射到 3D，投票决定边界 Gaussian；
- **边界感知的可见性/尺度**：边界处高斯尺度/不透明度对 splat 边界的影响。

### 4.3 边界感知的候选选择

- 候选分数加入 **2D-3D boundary consistency**：渲染 mask 与逐视角 2D mask 的边界一致性；
- 用 BIoU 作为 rerank/gating 的信号，而不是只用地 IoU；
- 评测同时报 IoU 和 BIoU，避免“面积对了、边界糊了”。

### 4.4 可借的方法/数学

- **Boundary IoU**（Cheng et al., CVPR 2021）：边界带定义与评测；
- **Boundary-aware segmentation**（Gated-SCNN、PointRend）：边界精修；
- **Graph cut / level set / matting**：trimap 边界精修；
- **Distance transform / signed distance field**：边界带权重、边界损失；
- **Optimal transport**：多视角边界匹配；
- **SAM/HQ-SAM**：更强的 2D mask 边界源。

---

## 5. 两个必须补的实验

1. **当前方法的 BIoU**：对 main / module C / gateE 的渲染 mask 算 BIoU（2%/1%），
   与 2D GT-best 上界（0.678/0.633）和 lifted 上界（0.636/0.566）比较。
   如果 gateE 的 BIoU 已经接近 lifted 上界，说明边界损失主要来自 2D→3D→2D；
   如果远低，说明候选/掩膜选择也在损失边界。
2. **2D 边界精修**：对 gateE 选中的候选，用查询视角 2D mask 做 trimap + graph cut / matting，
   看 IoU/BIoU 是否提升；这是最便宜、最直接的边界改进。

---

## 6. 与“objectness/候选”方向的关系

- **不是二选一**：身份/候选损失（recall/deep）和边界损失是两条独立轴。
- 候选修复能把 recall 0.02→0.2；边界修复能把 found/deep 的 BIoU 拉近 2D 上界。
- 论文如果只做候选，会被问“边界为什么这么糊”；只做边界，会被问“recall 为什么这么低”。
- 最稳的叙事：**先 objectness（候选从哪里来），再 boundary-preserving 2D-3D-2D（掩膜怎么保真）**，
  两条都用 GT 量化头寸。

---

## 7. 一句话

**ReLaGS 的 2D-3D-2D 同时损失语义和边界；GT 最好的 2D mask 也只有 42%/22% 的查询 BIoU≥0.95，
lift 后进一步掉到 12.5%/2.4%。边界是一个独立、可量化、未被优化的关键难点。**