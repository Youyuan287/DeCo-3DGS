"""Decisive audit: do the 2D segmentation masks (the source of all 3D candidates)
contain the queried object/part? For each query view, compute the best IoU between
the GT mask and any 2D mask in the level-3 segmentation map.

If recall-fail parts are not covered by any 2D mask, no 3D lifting/aggregation can
recover them; if they are covered, instance-building work is justified.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
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

gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
rows = []
nq = 0
for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
    name = im[:-4]
    anno = json.load(open(os.path.join(gt_dir, name + ".json")))
    cam = cams[name]
    h, w = cam.image_height, cam.image_width
    seg = cam.semantic["seg_map"][3].detach().cpu().numpy().astype(np.int64)
    fg = cam.semantic["fg_mask"][3].detach().cpu().numpy().astype(bool)
    labels = np.unique(seg[fg])
    prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
    for prompt in prompts:
        nq += 1
        if args.limit_queries and nq > args.limit_queries:
            break
        gt = np.zeros((h, w), np.uint8)
        for o in anno["objects"]:
            if o["category"] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
        gt = gt.astype(bool)
        best = 0.0
        best_l = -1
        for L in labels:
            m = (seg == L) & fg
            u = np.logical_or(m, gt).sum()
            if u == 0:
                continue
            v = np.logical_and(m, gt).sum() / u
            if v > best:
                best, best_l = float(v), int(L)
        rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                         gt_area=int(gt.sum()), n_masks=int(len(labels)),
                         best_mask_iou=best, best_label=best_l))
    print("IMG", name, len(prompts), flush=True)
    if args.limit_queries and nq >= args.limit_queries:
        break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_2dmaskcov.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))