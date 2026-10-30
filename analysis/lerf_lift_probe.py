"""Lift probe: can an existing SAM 2D mask be lifted to a good 3D segment?

For each query: take the 2D mask in the query view with the best IoU vs GT (the SAM
mask, no switching), select visible Gaussians whose projected center falls inside it
(render_point weight > 1e-4), render that selection, and measure IoU vs GT.

This tests whether masklet-based 3D candidates built from the FIXED SAM masks are
viable, and quantifies the 2D-side upper bound.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, render_point
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
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                  dtype=torch.float32, device="cuda")
xyz = (gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz).detach()


def render_bool(bm, cam, h, w):
    """Render the union of Gaussians whose boolean mask is True."""
    rel = bm.float().view(-1, 1).cuda()
    gaussians._semantics = rel.expand(-1, 20)
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


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
        seg = cam.semantic["seg_map"][3].detach().cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].detach().cpu().numpy().astype(bool)
        labels = np.unique(seg[fg])
        wt = render_point(cam, gaussians, pipe, bg)["weight"].detach().flatten()
        vis = (wt > 1e-4)
        R = torch.from_numpy(np.asarray(cam.R)).float().cuda()
        T = torch.from_numpy(np.asarray(cam.T)).float().cuda()
        pc = xyz @ R + T
        z = torch.clamp(pc[:, 2], min=1e-6)
        fx = w / (2 * np.tan(cam.FoVx * 0.5))
        fy = h / (2 * np.tan(cam.FoVy * 0.5))
        ixn = ((pc[:, 0] / z) * fx + w / 2).round().long().cpu().numpy()
        iyn = ((pc[:, 1] / z) * fy + h / 2).round().long().cpu().numpy()
        frontn = (vis & (pc[:, 2] > 1e-4)).cpu().numpy()
        inb = frontn & (ixn >= 0) & (ixn < w) & (iyn >= 0) & (iyn < h)
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
            best, best_L = 0.0, -1
            for L in labels:
                m = (seg == L) & fg
                u = np.logical_or(m, gt).sum()
                if u == 0:
                    continue
                v = np.logical_and(m, gt).sum() / u
                if v > best:
                    best, best_L = float(v), int(L)
            bm = np.zeros(len(xyz), bool)
            if best_L >= 0:
                m = (seg == best_L) & fg
                bm[inb] = m[iyn[inb], ixn[inb]]
            n_sel = int(bm.sum())
            lifted = render_bool(torch.from_numpy(bm), cam, h, w) if n_sel else np.zeros((h, w), bool)
            rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                             cov2d=best, n_sel=n_sel,
                             lifted_iou=iou(lifted, gt), gt_area=int(gt.sum())))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_liftprobe.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))