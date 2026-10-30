"""Cross-view 2D semantic consistency of the same object.

For each scene and each GT category present in >=2 annotated frames:
  - in each frame, pick the 2D mask (level-3 SAM segmentation) with best IoU vs GT;
  - take that mask's dataset CLIP feature (absolute-label indexing);
  - measure pairwise cosine between the same object's features across views;
  - cross-view retrieval: rank of view B's best mask among all B masks by cosine
    to view A's feature (A->B and B->A).
Baselines: cosine between features of different categories / random mask pairs.
GT is used only to identify each view's object mask.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
cams = {c.image_name: c for c in scene.getTrainCameras()}
gt_dir = os.path.join(args.gt_root, args.scene)

obj = defaultdict(list)   # category -> [(frame, feat, best_iou, n_masks, masks, feats)]
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".json")):
        name = im[:-5]
        if name not in cams:
            continue
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
        labels = [int(L) for L in np.unique(seg[fg]) if 0 <= int(L) < sem.shape[0]]
        if not labels:
            continue
        mask_arrs = {L: ((seg == L) & fg) for L in labels}
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        for prompt in list(dict.fromkeys(o["category"] for o in anno["objects"])):
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            ious = {}
            for L, m in mask_arrs.items():
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            bestL = max(ious, key=ious.get)
            obj[prompt].append(dict(frame=name, feat=sem[bestL], best_iou=ious[bestL],
                                    labels=labels, feats=sem))

rows = []
same_cos, diff_cos = [], []
retr = {1: 0, 3: 0, 5: 0}
retr_n = 0
cats = sorted(obj)
for c in cats:
    fs = obj[c]
    for i in range(len(fs)):
        for j in range(len(fs)):
            if i == j:
                continue
            a, bb = fs[i], fs[j]
            # retrieval A -> B
            sims = bb["feats"] @ a["feat"]
            order = np.argsort(-sims)
            best_idx = bb["labels"].index(max(bb["labels"], key=lambda L: 0)) if False else None
            # find index of B's best mask: recompute via stored labels ordering is ambiguous;
            # instead compare to the feature of B's best mask:
            sims_best = float(bb["feat"] @ a["feat"])
            rank = int(np.sum(sims > sims_best + 1e-12)) + 1
            retr_n += 1
            for k in (1, 3, 5):
                if rank <= k:
                    retr[k] += 1
            if i < j:
                same_cos.append(sims_best)
for c in cats:
    for d in cats:
        if c >= d:
            continue
        for a in obj[c][:2]:
            for bb in obj[d][:2]:
                diff_cos.append(float(bb["feat"] @ a["feat"]))

same_cos = np.array(same_cos)
diff_cos = np.array(diff_cos)
print("scene", args.scene)
print("same-object cross-view pairs:", len(same_cos),
      "cosine median %.3f p10 %.3f p90 %.3f" % (np.median(same_cos), np.percentile(same_cos, 10), np.percentile(same_cos, 90)))
print("different-object pairs:", len(diff_cos),
      "cosine median %.3f p10 %.3f p90 %.3f" % (np.median(diff_cos), np.percentile(diff_cos, 10), np.percentile(diff_cos, 90)))
if retr_n:
    print("cross-view retrieval of the same object: top1 %.1f%% top3 %.1f%% top5 %.1f%% (n=%d)" % (
        100 * retr[1] / retr_n, 100 * retr[3] / retr_n, 100 * retr[5] / retr_n, retr_n))
per_cat = []
for c in cats:
    fs = obj[c]
    if len(fs) < 2:
        continue
    cs = [float(fs[j]["feat"] @ fs[i]["feat"]) for i in range(len(fs)) for j in range(i + 1, len(fs))]
    per_cat.append((c, len(fs), float(np.median(cs)) if cs else float("nan")))
print("per-category (n_views, median same-cos):", per_cat)

with open(os.path.join(args.out), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["scene", "cat", "n_views", "same_cos_med", "n_same_pairs", "diff_cos_med",
                 "retr_top1", "retr_top3", "retr_top5", "retr_n"])
    wr.writerow([args.scene, "ALL", len(cats), float(np.median(same_cos)) if len(same_cos) else "",
                 len(same_cos), float(np.median(diff_cos)) if len(diff_cos) else "",
                 (retr[1] / retr_n) if retr_n else "", (retr[3] / retr_n) if retr_n else "",
                 (retr[5] / retr_n) if retr_n else "", retr_n])
    for c, n, m in per_cat:
        wr.writerow([args.scene, c, n, m, "", "", "", "", "", ""])
print("SCENE_DONE", args.scene)