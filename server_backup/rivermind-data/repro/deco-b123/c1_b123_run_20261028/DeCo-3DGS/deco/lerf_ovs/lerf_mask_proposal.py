"""Text-selected SAM mask lifted to a 3D proposal (mask expansion, training-free).

For each query, in the query view only:
  - score every level-3 SAM mask by text similarity of its dataset CLIP feature;
  - take the top-1 mask (and top-3), lift the visible Gaussians whose projected centers fall
    inside the mask (render_point weight > 1e-4), render the selection, measure IoU vs GT.
No NAG pool, no hierarchy, no GT in selection.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, render_point
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
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                  dtype=torch.float32, device="cuda")
xyz = (gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz).detach()
vlm = ClipSimMeasure()
vlm.load_model()


def render_bool(bm, cam, h, w):
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
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
        labels = [int(L) for L in np.unique(seg[fg]) if 0 <= int(L) < sem.shape[0]]
        if not labels:
            continue
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
            vlm.encode_text(prompt)
            t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
            t /= max(np.linalg.norm(t), 1e-8)
            sims = {L: float(sem[L] @ t) for L in labels}
            order = sorted(labels, key=lambda L: sims[L], reverse=True)
            ious = {}
            for L in labels:
                m = (seg == L) & fg
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            out = dict(scene=args.scene, image=name, prompt=prompt, n_masks=len(labels))
            for K, tag in ((1, "top1"), (3, "top3")):
                chosen = order[:K]
                bm = np.zeros(len(xyz), bool)
                for L in chosen:
                    m = (seg == L) & fg
                    bm[inb] = bm[inb] | m[iyn[inb], ixn[inb]]
                lifted = render_bool(torch.from_numpy(bm), cam, h, w) if bm.any() else np.zeros((h, w), bool)
                out["prop_%s_iou" % tag] = iou(lifted, gt)
                out["cov2d_%s" % tag] = max(ious[L] for L in chosen)
                out["area_%s" % tag] = float(sum(int(((seg == L) & fg).sum()) for L in chosen))
            out["best_label_iou_in_view"] = max(ious.values())
            out["sim_top1"] = sims[order[0]]
            out["sim_margin"] = sims[order[0]] - sims[order[1]] if len(order) > 1 else 0.0
            rows.append(out)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_maskprop.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))