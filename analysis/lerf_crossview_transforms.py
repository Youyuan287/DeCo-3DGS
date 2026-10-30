"""Training-free feature transforms: can simple post-processing increase
cross-view consistency and class separability of mask features?

Transforms (no training):
  raw      : L2-normalized dataset features
  viewz    : per-view z-score across masks (per dimension), then L2
  white    : per-scene ZCA whitening over all masks, then L2
  viewz+wh : viewz followed by scene whitening
  power    : signed sqrt, then L2
Metrics: same-object / different-object cross-view cosine, Fisher ratio,
cross-view retrieval top1/3/5 (using each view's GT-best mask as the anchor).
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


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


views = []   # {frame, labels, feats}
obj = defaultdict(list)  # prompt -> [(view_idx, best_label)]
with torch.no_grad():
    for name, cam in cams.items():
        pass
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".json")):
        name = im[:-5]
        if name not in cams:
            continue
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        labels = np.array([int(L) for L in np.unique(seg[fg]) if 0 <= int(L) < sem.shape[0]])
        if labels.size == 0:
            continue
        feats = unit(sem[labels])
        vi = len(views)
        views.append(dict(frame=name, labels=labels, feats=feats))
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        for prompt in list(dict.fromkeys(o["category"] for o in anno["objects"])):
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            bestL, bestI = None, -1
            for k, L in enumerate(labels):
                m = (seg == L) & fg
                u = np.logical_or(m, gt).sum()
                v = float(np.logical_and(m, gt).sum() / u) if u else 0.0
                if v > bestI:
                    bestI, bestL = v, k
            obj[prompt].append((vi, bestL))

ALL = np.concatenate([v["feats"] for v in views], 0)
mu = ALL.mean(0)
C = np.cov(ALL.T)
evals, evecs = np.linalg.eigh(C)
W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-6, None))) @ evecs.T


def transform(tag, feats):
    if tag == "raw":
        return feats
    if tag == "viewz":
        return unit((feats - feats.mean(0)) / (feats.std(0) + 1e-6))
    if tag == "white":
        return unit((feats - mu) @ W)
    if tag == "viewz+wh":
        z = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
        return unit((z - z.mean(0)) @ W)
    if tag == "power":
        return unit(np.sign(feats) * np.sqrt(np.abs(feats)))
    return feats


rows = []
for tag in ["raw", "viewz", "white", "viewz+wh", "power"]:
    F = [transform(tag, v["feats"]) for v in views]
    same, diff = [], []
    retr = {1: 0, 3: 0, 5: 0}
    n = 0
    cats = sorted(obj)
    for c in cats:
        fs = obj[c]
        for i in range(len(fs)):
            for j in range(len(fs)):
                if i == j:
                    continue
                va, la = fs[i]
                vb, lb = fs[j]
                if va == vb:
                    continue
                sims = F[vb] @ F[va][la]
                sb = float(F[vb][lb] @ F[va][la])
                rank = int(np.sum(sims > sb + 1e-12)) + 1
                n += 1
                for k in (1, 3, 5):
                    if rank <= k:
                        retr[k] += 1
                if i < j:
                    same.append(sb)
    for c in cats:
        for d in cats:
            if c >= d:
                continue
            for (va, la) in obj[c][:2]:
                for (vb, lb) in obj[d][:2]:
                    diff.append(float(F[vb][lb] @ F[va][la]))
    same = np.array(same)
    diff = np.array(diff)
    fish = (same.mean() - diff.mean()) / np.sqrt((same.var() + diff.var()) / 2) if len(same) and len(diff) else float("nan")
    rows.append(dict(transform=tag, same_med=float(np.median(same)) if len(same) else float("nan"),
                     diff_med=float(np.median(diff)) if len(diff) else float("nan"),
                     fisher=float(fish),
                     retr1=retr[1] / max(n, 1), retr3=retr[3] / max(n, 1), retr5=retr[5] / max(n, 1),
                     n=n))
    print(tag, rows[-1], flush=True)

with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene)