"""2D multi-granularity gate on LERF-OVS.

For every query, pools all SAM masks from seg_map levels 0..3 and measures:
  - best IoU at each level (what ReLaGS level-3 sees vs finer levels);
  - best single mask over all levels (granularity oracle);
  - greedy union of up to 10 masks from the top-20 candidates (set oracle).
No 3D lifting yet; this is the gate for building a part-whole DAG.
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


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def masks_for(cam):
    out = []
    for L in range(4):
        seg = cam.semantic["seg_map"][L].detach().cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][L].detach().cpu().numpy().astype(bool)
        labels = np.unique(seg[fg])
        out.append([((seg == l) & fg) for l in labels])
    return out


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
        levels = masks_for(cam)
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
            if not gt.any():
                continue
            best_l = []
            cand = []  # (iou, level, idx, mask)
            for L, ms in enumerate(levels):
                bl = 0.0
                for i, m in enumerate(ms):
                    v = iou(m, gt)
                    if v > bl:
                        bl = v
                    cand.append((v, L, i, m))
                best_l.append(bl)
            cand.sort(key=lambda x: x[0], reverse=True)
            best_all = cand[0][0] if cand else 0.0
            best_L = cand[0][1] if cand else -1
            best_i = cand[0][2] if cand else -1
            # greedy set oracle: top-20 by individual IoU, up to 10 additions
            pool = cand[:20]
            union = np.zeros((h, w), bool)
            used = set()
            best_union = 0.0
            n_union = 0
            union_levels = []
            for _ in range(10):
                best = None
                for j, (v, L, i, m) in enumerate(pool):
                    if j in used:
                        continue
                    u = np.logical_or(union, m)
                    val = iou(u, gt)
                    if best is None or val > best[0]:
                        best = (val, j, u)
                if best is None or best[0] <= best_union + 1e-6:
                    break
                best_union, j, union = best
                used.add(j)
                n_union += 1
                union_levels.append(pool[j][1])
            rows.append(dict(
                scene=args.scene, image=name, prompt=prompt,
                n0=len(levels[0]), n1=len(levels[1]), n2=len(levels[2]), n3=len(levels[3]),
                best0=best_l[0], best1=best_l[1], best2=best_l[2], best3=best_l[3],
                best_all=best_all, best_level=best_L, best_idx=best_i,
                best_union=best_union, n_union=n_union,
                union_levels=";".join(str(x) for x in union_levels),
                gt_area=int(gt.sum()),
            ))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_mg.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))