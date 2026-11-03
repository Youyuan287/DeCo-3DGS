# 独立审查任务
请审查以下完整裁决稿及未改写的原始表/代码材料。你没有工具，只能审查附带内容，不能声称已浏览链接或访问路径。与Codex独立判断：唯一问题是否值得持续投入？哪些是已知工程补全、哪些证据还不足？是否应拒绝推荐任何新方法？最多三项核心反对，提供明确的最小取舍，不打分，不设随意pp门槛。若本轮检查不足，直接说RETHINK或DIAGNOSTIC_ONLY。保持原任务边界，不授权实验。

# 初始问题裁决稿：相关性排序何时不足以确定最终三维区域

## 问题锚点

目标：在开放词汇三维分割领域中，依据本项目真实历史证据、当前代码和2026-10-24实时文献，收敛到一个值得持续投入且可证伪的问题；不将领域名称与具体3D-OVS数据集混淆。

必须解决：把失败现象、竞争原因、方法干预与最终分割收益连接起来；允许最后不推荐任何具体新方法。

非目标：不复刻高oracle发现；不默认整体/局部失衡或单向量压缩为主因；不改骨干/堆模块/加损失充当创新；不追旧最高分、不把工程补洞当语义能力。

边界：本轮只读审查、必要CPU分析和方法设计。不训练、不提特征、不SAM、不建图、不参数扫描、不修改正式评测协议，不启动长GPU作业。已有代码/模型/结果保留。

成功条件：四类入口各≤2候选，合并后≤3条，选唯一问题（或证据不足）；明确原始证据、相近工作、新增算法对象、直接对照、最终指标、泛化限制、基于波动而非随意百分点的继续/停止条件。

## 候选问题（不预先认定领域根因）

1. 查询相关性与最终区域集合质量是否错配：在语义排名、候选边界、渲染固定时，是否缺乏决定完整性/排他性/停止的独立证据？red apple属于数量变化案例，pirate hat属于成员替换，二者不是一个原因。原始正确候选和high-oracle只是前提，不是本问题新发现。
2. 聚合是否删除少数但有用的视角证据：BSHRINK方差收缩和robust去离群可能损伤真part，但缺旧production映射和IRLS-no-shrink，尚不能证明；EviSplat/LaGa已有直接方法重叠。
3. 固定分区是否缺失查询需要的边界：DIGS自己报告SAM粒度上限；Search3D已有多层对象/part表示。若是此问题，必须改sp_partition/merge/association，不能只给固定NAG重排。目前本项目多次高3D候选oracle不支持把此项认定全局主因。

初选问题1作为唯一诊断优先级，但**不推荐任何已经成立的新方法**。原因是旧校准/扩k已失败，新空间乘法也失败，且“类别置信度不同于mask质量”已有Mask Scoring R-CNN等先例。

## 新增低成本核验与反证

对最新figurines21query的原pre-gap分数做affine、square、sqrt、logit温度2和.5，全部严格保持排名。五种变换均0/21改变最终节点集合。只用了CPU，没有新mIoU。这削弱“普通尺度变化普遍导致gap不稳定”的解释。不能据red apple把它称领域定律；空间重评分不是单调变换。

## 最小闭环（本轮只设计，不执行GPU）

修改对象是独立selector接口，不动特征、NAG、边界、root筛选、parent-child比较或renderer。令r0为原排序分数，r1为已经实现的空间重评分，k0和k1分别是两个分数经原gap后的数量。

```
decode(order_scores, count):
    ordered = stable_sort(frozen_pool, order_scores)
    return exact_original_render(gaussian_union(ordered[:count]))
```

四组：rank0/count0（原）、rank1/count1（原空间组）、rank1/count0、rank0/count1。前两组已有figurines结果，真正缺少后两项的同源最终渲染。若只用已有匹配节点组合mask，必须逐条验证覆盖；不能对未知组合bitwise OR后冒称正式渲染。

这个分解本身是**直接因果对照，不是论文新算法**。它用来决定何时继续成员选择、何时考虑集合停止，避免把一个标量同时承担两项决策产生的混杂再解释成语义能力。不能因oracle variableK≈fixed3就认定实际decoder完全不受数量影响，两者优化条件不同。

主指标仍为最终Gaussian选择→原renderer→原固定阈值→正式图内query/图像/场景mIoU。解释性指标只保留节点/并集、数量、FP/FN、空转换、按query聚类增益。没有新增GT触发，不按测试IoU决定query或k。

先在开发figurines完成缺失2组并复核pirate hat投影；冻结方案再在另外3场景回顾性配对，不称全新留出集。若存在简单固定K或单一开发温度可获得同样收益，停止复杂集合机制；若仍主要是错选unrelated节点，停止gap主线；若对应合法空间oracle也差才转候选构建。

继续门槛不采用ARIS planner提出的任意2pp或缩oracle50%。复用已验收baseline，必要短重复估计同输入数值噪声，以query/实体聚类的配对区间及逐场景符号、耗时和leave-one-query-out检验是否值得。实际净收益若未超baseline重放波动，或只集中单个query/空补洞，不升级为论文主线。正式跨数据集能力需新冻结独立评测，当前未证明。

## 当前最邻近原文（由Codex核查，顾问不能自行打开链接）

- THGS §3.5：多层region relevance→Top候选→Gaussian二值渲染。https://arxiv.org/html/2504.13153v1
- ReLaGS §12 Alg1：child-parent比分下降、最大score drop决定前缀；已覆盖查询自适应层级。https://arxiv.org/html/2603.17605v1
- COS3D §3.3：语言prompt→实例特征邻域→opacity加权relevance接受region；已覆盖语言/边界协同，不能把两种证据分开作为新贡献。https://arxiv.org/html/2510.20238v1
- VALA §4/A/B：可见性贡献gate和robust multi-view聚合；其2D特征图阈值与3D选择/输出阈值不同。https://arxiv.org/html/2509.05515v1
- DIGS §3、D.1、E.3：多标签Dirichlet/late-argmax已有；实际检索包含阈值和margin，不能把reported71.95全部归因分布表示；part缺slot是作者明确限制。https://arxiv.org/html/2610.16177v1
- EviSplat §4/A.2/B：保留观测bank/query聚合、Gaussian局部证据、固定后处理已有；其LERF-OVS按frame-class平坦均值，不可直接与本项目图像平均相减。https://arxiv.org/html/2609.34853v1
- Search3D §III：对象/part层级及SigLIP像素/对象特征已有。https://arxiv.org/html/2409.18431v2
- QAGaussian v1：query-conditioned slots和多粒度router，含训练，非同设定直接baseline。https://arxiv.org/abs/2608.16103
- Mask Scoring R-CNN（CVPR2019）：类别置信度与mask质量解耦并监督预测IoU已有，不可将同一问题重新称首次发现。https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html

## 新颖性裁决

上述文献没有为本项目提供一个已证明有效的training-free集合质量目标；这不是“无人研究”的证明。当前仅有值得优先证伪的问题，没有可直接投入完整论文方法的创新性证据。若最小闭环只有ReLaGS gap工程收益，归类工程补全，不升级为领域贡献。

dataset,scene,baseline_origin,mode,miou_percent,delta_vs_baseline_pp,n_query_views,empty_masks
LERF-OVS,figurines,author_PT,baseline,63.28714011730666,0,56,2
LERF-OVS,figurines,author_PT,visibility,49.49520142350826,-13.791938693798397,56,1
LERF-OVS,figurines,author_PT,spatial,52.70679899425017,-10.580341123056487,56,2
LERF-OVS,figurines,author_PT,permutation,16.60719676485241,-46.67994335245423,56,23

mode,improved,regressed,unchanged,gaussian_union_changed,rescued_empty,newly_empty,both_nonempty_n,both_nonempty_contribution_pp,other_contribution_pp,total_delta_pp
visibility,7,20,29,32,1,0,54,-15.223216592687633,1.431277898889235,-13.791938693798397
spatial,4,13,39,25,1,1,53,-12.011619021945721,1.431277898889235,-10.580341123056487
permutation,6,38,12,50,0,21,33,-23.160354035876864,-23.519589316577385,-46.67994335245423

{
  "dataset": "LERF-OVS",
  "scene": "figurines",
  "n_unique_queries": 21,
  "input_sha256": "8a9c7abb4adab40e3d90b280bdfd39c5b62d6aa7511e230ad8da6cd0b12471e6",
  "source_sha256": "9216d2c6cc4e465dcca435d957a473393ff5edc2414e60506fa8dda777261931",
  "official_miou": null,
  "diagnostic_only": true,
  "transforms": {
    "affine_control": {
      "changed": 0,
      "total": 21,
      "rank_unchanged_all": true
    },
    "square": {
      "changed": 0,
      "total": 21,
      "rank_unchanged_all": true
    },
    "sqrt": {
      "changed": 0,
      "total": 21,
      "rank_unchanged_all": true
    },
    "odds_temperature_2": {
      "changed": 0,
      "total": 21,
      "rank_unchanged_all": true
    },
    "odds_temperature_half": {
      "changed": 0,
      "total": 21,
      "rank_unchanged_all": true
    }
  }
}
"""
This module defines the SemanticNAG class, which constructs a hierarchical superpoint graph with semantic features.
It builds upon the Nested Adjacency Graph (NAG) structure from the SPT (Superpoint Transformer) library (https://arxiv.org/abs/2306.08045).
"""
import torch
from typing import List
import sys
sys.path.append('ext/')
from spt.data import Data, NAG, Cluster
# from ext.spt.data import Data, NAG, Cluster

class SemanticNAG():
    def __init__(self, labels, feat):
        labels = [label.cuda() for label in labels]
        self.labels = labels # [4, N]
        self.nag = self.build_nag_from_multilevel_labels(labels)
        self.feat = feat
        self.gaussian_num = labels[0].shape[0]

    # sim in, gaussian_num * 1 out, indicating the 0-1 mask
    def get_related_gaussian(self, sim: List[torch.Tensor], topk: int = 1, level: int = -1) -> torch.Tensor:
        """
        sim: List of Tensors, each shape like superpoint
        topk: int, number of top similar points to consider
        level: int, choose certain level to get the related gaussian, or -1 to get all levels
        """
        assert len(sim) == len(self.labels) - 1, "Number of similarity matrices must match number of levels"
        if level == -1:
            lvls = [i for i in range(1, len(sim))]
        elif isinstance(level, list):
            lvls = [i - 1 for i in level]
        else:
            lvls = [level - 1]
        
        # 1. find the topk similar points in sim list, get tuple (level, sim_val, index)
        related_sp_lvl = []
        for i in lvls:
            sim_array = sim[i]
            sim_val, indices = torch.topk(sim_array, topk)
            related_sp_lvl.extend(list(zip([i+1] * topk, sim_val, indices)))
        # sort the list based on sim_val, descending order
        related_sp_lvl.sort(key=lambda x: x[1], reverse=True)
        # get the topk similar points
        related_sp_lvl = related_sp_lvl[:topk]
        
        # 2. get the related gaussian
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for i, tup in enumerate(related_sp_lvl):
            level, _, index = tup
            lowest_idx = torch.where(self.labels[level] == index)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians
    
    def get_gs_by_mask(self, mask, level):
        """
        mask: N, boolean mask
        level: int, choose certain level to get the related gaussian
        """
        assert level > 0 and level <= len(self.labels), "Level out of range"
        # get the sub from NAG, iterative down to level 0
        label = self.labels[level]
        sps = label[mask]
        # get gaussian in the sps
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for sp in sps:
            lowest_idx = torch.where(label == sp)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians

    
    @staticmethod
    def build_nag_from_multilevel_labels(labels: List[torch.Tensor]) -> NAG:
        """
        labels: List of Tensors, each shape = (N,), level-0 point to higher level cluster IDs
            e.g. [label_lvl0, label_lvl1, label_lvl2]
        pos / feat: Optional, only used at level 0
        """
        assert len(labels) >= 2, "At least two levels required to construct NAG"
        device = labels[0].device
        N = labels[0].shape[0]
        num_levels = len(labels)

        data_list = [Data(num_nodes=N, super_index=labels[0])]
        prev_sub = None

        for i in range(num_levels):
            data = Data()
            # 求每层的sub和上一层的super_index
            if i == 0:
                upper_labels = labels[i]
                lower_labels = torch.arange(N, device=device)
                sorted_upper_labels, perm = torch.sort(upper_labels)
                sorted_lower_labels = lower_labels[perm]
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
            else:
                data = Data(num_nodes=labels[i-1].max().item() + 1)
                data.sub = prev_sub
                upper_labels = labels[i].long()
                lower_labels = labels[i-1].long()

                # 1. remove duplicates based on lower labels
                unique_lower_labels, inv = torch.unique(lower_labels, sorted=True, return_inverse=True)
                unique_upper_labels = torch.zeros_like(unique_lower_labels)
                unique_upper_labels[inv] = upper_labels
                # print('uni:', lower_labels, unique_lower_labels, unique_upper_labels, inv)

                data.super_index = unique_upper_labels
                # 2. sort based on upper labels, construct next sub
                sorted_upper_labels, perm = torch.sort(unique_upper_labels)
                sorted_lower_labels = unique_lower_labels[perm]
                # print("sort:", sorted_lower_labels, sorted_upper_labels)
                # upper label take down the change
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                # print('csr:', pointers, sorted_lower_labels)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
                data_list.append(data)
        data_list.append(Data(sub=prev_sub, num_nodes=labels[-1].max().item() + 1))
        return NAG(data_list)

if __name__ == '__main__':
    lvl1 = torch.tensor([4, 5, 2, 3, 1, 5, 0, 3])
    lvl2 = torch.tensor([3, 4, 3, 2, 1, 4, 0, 2])
    lvl3 = torch.tensor([1, 2, 1, 1, 0, 2, 0, 1])
    nag = SemanticNAG.build_nag_from_multilevel_labels([lvl1, lvl2, lvl3])
    print(nag, nag[1].sub, 777)
    print(nag[3].sub[1].points)
    pt = 1
    for i in range(3, 0, -1):
        pt = nag[i].sub[pt].points
        print(55, pt)
    print(nag.get_super_index(1, 0))
    print(nag.get_super_index(2, 0))
    print(nag.get_super_index(3, 0))
    print(nag.get_super_index(2, 1))
    # print(from_super_index(lvl2, lvl1))
    # cls = Cluster(pointers=torch.tensor([0, 3, 5, 8]), points=torch.tensor([2, 0, 1, 3, 4, 5, 6, 7]))
    # print(cls, cls.pointers)

"""Pure cache-only spatial support primitives. No GT, model loading or I/O."""
import numpy as np


def validate_run_inputs(cache_frames, query_by_image, historical_rows):
    cache_frames = set(cache_frames)
    if not set(query_by_image).issubset(cache_frames):
        raise ValueError('Evaluation frames cannot be matched to cache frames')
    rows = [r for r in historical_rows if r.get('scene') == 'figurines']
    required = {'scene','image','prompt','iou','pred_px'}
    if not rows or any(not required.issubset(r) for r in rows):
        raise ValueError('Missing historical fields')
    keys = [(r['image'],r['prompt']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate historical query keys')
    expected = {(im,q) for im,qs in query_by_image.items() for q in qs}
    if not expected.issubset(set(keys)):
        raise ValueError('Incomplete historical query coverage')
    return sorted(cache_frames - set(query_by_image))


def select_feature(scores, seg):
    scores, seg = np.asarray(scores), np.asarray(seg)
    ids = np.unique(seg[seg >= 0])
    if not len(ids) or ids.max() >= len(scores):
        raise ValueError('No valid proposal masks or invalid feature indices')
    return int(ids[np.argmax(scores[ids])])


def mask_iou(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError('Mask shape mismatch')
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def evidence_scores(ious, visible):
    ious, visible = np.asarray(ious, float), np.asarray(visible, bool)
    if ious.shape != visible.shape or ious.ndim != 1:
        raise ValueError('Evidence must be matched one-dimensional arrays')
    if not np.isfinite(ious).all() or ((ious < 0) | (ious > 1)).any():
        raise ValueError('Invalid IoU')
    if not len(ious):
        return 0.0, 0.0
    return float(visible.mean()), float(np.where(visible, ious, 0).mean())


def gap_select(candidates, scores):
    """Original ReLaGS prefix rule, including empty single/tied candidate cases."""
    if len(candidates) != len(scores):
        raise ValueError('Candidates and scores differ in length')
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    gaps = [0.0] + [ranked[i][1] - ranked[i+1][1] for i in range(len(ranked)-1)]
    cut = gaps.index(max(gaps))
    return [item[0] for item in ranked[:cut]]


def rescore(base_scores, evidence, all_unobserved=False):
    if len(base_scores) != len(evidence):
        raise ValueError('Evidence and candidates differ in length')
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in evidence):
        raise ValueError('Invalid evidence')
    if all_unobserved:
        return list(base_scores)
    return [float(s * e) for s, e in zip(base_scores, evidence)]


def proposal_permutation(seg, n_features, seed):
    """Feature ID -> mask ID derangement within view/level/log2(area) bins.

    Keeps the mask bank and score bank unchanged, NOT each selected mask's exact
    area. Singleton bins are unchanged and must be included in coverage reports.
    Feature IDs must be level-disjoint as in the actual LangSplat cache.
    """
    seg = np.asarray(seg)
    if seg.ndim != 3 or n_features < 0:
        raise ValueError('Expected level x height x width segmentation')
    if (seg < -1).any() or (seg >= n_features).any():
        raise ValueError('Segmentation ID outside feature table')
    rng = np.random.default_rng(seed)
    mapping = np.arange(n_features)
    seen = set()
    eligible = moved = singleton = 0
    areas = np.zeros(n_features, dtype=np.int64)
    for layer in seg:
        ids, counts = np.unique(layer[layer >= 0], return_counts=True)
        if seen.intersection(ids.tolist()):
            raise ValueError('Feature IDs shared across levels')
        seen.update(ids.tolist())
        areas[ids] = counts
        bins = np.floor(np.log2(counts)).astype(int)
        eligible += len(ids)
        for bin_id in np.unique(bins):
            group = ids[bins == bin_id].copy()
            if len(group) == 1:
                singleton += 1
                continue
            rng.shuffle(group)
            mapping[group] = np.roll(group, 1)
            moved += len(group)
    return mapping, {
        'eligible_features': eligible, 'moved_features': moved,
        'singleton_features': singleton,
        'total_mask_area_preserved': bool(areas.sum() == areas[mapping].sum()),
    }


