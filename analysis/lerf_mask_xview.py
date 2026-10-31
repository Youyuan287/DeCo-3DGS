"""Cross-view mask SELECTION: choose the query-view SAM mask using cross-view
mask-feature support (plus text), then lift it to 3D. Training-free; GT only for IoU.

Variants per query:
  text   : top-1 by text similarity (baseline proposal)
  xview  : top-1 by text + lam * cross-view support (support z-scored within masks)
  supp   : top-1 by cross-view support only
Support of a mask = mean over other annotated frames of the max cosine to that frame's masks.
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


def zs(x):
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-12 else np.zeros_like(x)


gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
frames = []
for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".json")):
    name = im[:-5]
    cam = cams.get(name)
    if cam is None:
        continue
    seg = cam.semantic["seg_map"][3].detach().cpu().numpy().astype(np.int64)
    fg = cam.semantic["fg_mask"][3].detach().cpu().numpy().astype(bool)
    sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
    sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
    labels = [int(L) for L in np.unique(seg[fg]) if 0 <= int(L) < sem.shape[0]]
    if not labels:
        continue
    feats = sem[np.array(labels)]
    frames.append(dict(name=name, cam=cam, seg=seg, fg=fg, labels=labels, feats=feats))

rows = []
nq = 0
with torch.no_grad():
    for fr in frames:
        name, cam = fr["name"], fr["cam"]
        h, w = cam.image_height, cam.image_width
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        seg, fg = fr["seg"], fr["fg"]
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
            fq = fr["feats"]
            text = fq @ t
            supp = np.zeros(len(fq), np.float32)
            cnt = 0
            for other in frames:
                if other["name"] == name:
                    continue
                sims = fq @ other["feats"].T  # (nq, no)
                supp += sims.max(axis=1)
                cnt += 1
            if cnt:
                supp /= cnt
            zt, zs_ = zs(text), zs(supp)
            scores = {
                "text": zt,
                "xview05": zt + 0.5 * zs_,
                "xview10": zt + 1.0 * zs_,
                "supp": zs_,
            }
            out = dict(scene=args.scene, image=name, prompt=prompt, n_masks=len(fq))
            ious = {}
            for L in fr["labels"]:
                m = (seg == L) & fg
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            out["best_label_iou"] = max(ious.values())
            for tag, sc in scores.items():
                L = fr["labels"][int(np.argmax(sc))]
                m = (seg == L) & fg
                bm = np.zeros(len(xyz), bool)
                bm[inb] = m[iyn[inb], ixn[inb]]
                lifted = render_bool(torch.from_numpy(bm), cam, h, w) if bm.any() else np.zeros((h, w), bool)
                out[tag + "_iou"] = iou(lifted, gt)
                out[tag + "_cov2d"] = ious[L]
                out[tag + "_label"] = L
            rows.append(out)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_maskxview.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))