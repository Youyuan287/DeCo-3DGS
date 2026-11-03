"""Decisive feasibility probe: WITHOUT GT, can the query's best 2D mask be ranked
highly by mask-level CLIP similarity?

For each query (query view): rank all SAM masks by cos(mask CLIP feature, text).
GT is used ONLY to identify which mask is the best match (max IoU with GT) and to
report the rank that mask obtains under the GT-free score.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
vlm = ClipSimMeasure()
vlm.load_model()


def sims_for(prompt):
    vlm.encode_text(prompt)
    t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
    return t / max(np.linalg.norm(t), 1e-8)


gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
rows = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
        labels = np.unique(seg[fg])
        if labels.size == 0:
            continue
        smin = int(labels.min())
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        mask_arrs = {int(L): ((seg == L) & fg) for L in labels}
        for prompt in prompts:
            nq += 1
            if args.limit_queries and nq > args.limit_queries:
                break
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            ious = {}
            for L, m in mask_arrs.items():
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            best_L = max(ious, key=ious.get)
            t = sims_for(prompt)
            rows_sim = {}
            for L in mask_arrs:
                idx = L  # features are indexed by absolute label id
                if 0 <= idx < sem.shape[0]:
                    rows_sim[L] = float(np.dot(sem[idx], t))
                else:
                    rows_sim[L] = float("nan")
            order = sorted(rows_sim, key=lambda L: -rows_sim[L])
            rank = order.index(best_L) + 1 if best_L in order else -1
            row = dict(scene=args.scene, image=name, prompt=prompt,
                       n_masks=len(mask_arrs), best_label=int(best_L),
                       best_mask_iou=ious[best_L], best_mask_sim=rows_sim.get(best_L, float("nan")),
                       best_rank=rank)
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_maskrank.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))