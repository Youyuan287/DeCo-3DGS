"""Does per-view normalization keep text-based mask selection?

Variants for the similarity (text similarity is transformed consistently with the
feature transform so the comparison is metric-fair):
  raw     : <unit(f), unit(t)>
  viewz   : z-score features per view (per dimension); <unit(z), unit(t)>
  white   : Mahalanobis: <f - mu, Sigma^-1 t>
  viewz+wh: z per view, then scene whitening of z; <z - mu_z, Sigma_z^-1 t>
Reports the rank of the GT-best mask per family (GT only for evaluation).
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
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
vlm = ClipSimMeasure()
vlm.load_model()
cams = {c.image_name: c for c in scene.getTrainCameras()}
gt_dir = os.path.join(args.gt_root, args.scene)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# collect all view features for scene statistics (raw and per-view z-scored)
all_f = []
all_z = []
Z_BY_FRAME = {}
LBL_BY_FRAME = {}
for name, cam in cams.items():
    sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
    segv = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
    fgv = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
    lbl = np.array([int(L) for L in np.unique(segv[fgv]) if 0 <= int(L) < sem.shape[0]])
    if lbl.size == 0:
        continue
    Fv = unit(sem[lbl])
    Zv = unit((Fv - Fv.mean(0)) / (Fv.std(0) + 1e-6))
    Z_BY_FRAME[name] = Zv
    LBL_BY_FRAME[name] = lbl
    all_f.append(Fv)
    all_z.append(Zv)
ALL = np.concatenate(all_f, 0)
ALLZ = np.concatenate(all_z, 0)
mu = ALL.mean(0)
C = np.cov(ALL.T)
evals, evecs = np.linalg.eigh(C)
evals = np.clip(evals, 1e-4, None)
W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
Cinv = evecs @ np.diag(1.0 / evals) @ evecs.T

t_cache = {}
rows = []
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
        if name not in Z_BY_FRAME:
            continue
        labels = LBL_BY_FRAME[name]
        F = unit(sem[labels])
        # per-view z-score features (cached) and scene-level whitening of Z
        Z = Z_BY_FRAME[name]
        mzz = ALLZ.mean(0)
        Czz = np.cov(ALLZ.T)
        ev2, evec2 = np.linalg.eigh(Czz)
        ev2 = np.clip(ev2, 1e-4, None)
        C2inv = evec2 @ np.diag(1.0 / ev2) @ evec2.T
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        ious = {}
        for prompt in prompts:
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            ious[prompt] = []
            for k, L in enumerate(labels):
                m = (seg == L) & fg
                u = np.logical_or(m, gt).sum()
                ious[prompt].append(float(np.logical_and(m, gt).sum() / u) if u else 0.0)
        for pi, prompt in enumerate(prompts):
            if prompt not in t_cache:
                vlm.encode_text(prompt)
                t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
                t_cache[prompt] = t / max(np.linalg.norm(t), 1e-8)
            t = t_cache[prompt]
            iou_arr = np.array(ious[prompt])
            best = int(np.argmax(iou_arr))
            variants = {
                "raw": F @ t,
                "viewz": Z @ t,
                "white": F @ (Cinv @ t),
                "viewz+wh": Z @ (C2inv @ t),
            }
            row = dict(scene=args.scene, image=name, prompt=prompt, n=len(labels),
                       iou_best=float(iou_arr[best]), best_label=int(labels[best]))
            for tag, sims in variants.items():
                order = np.argsort(-sims)
                row["rank_" + tag] = int(np.where(order == best)[0][0]) + 1
                row["iou_top1_" + tag] = float(iou_arr[order[0]])
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))