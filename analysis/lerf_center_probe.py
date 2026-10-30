"""Centering-only transform: per-view mean subtraction without re-scaling.

Variants:
  raw       : unit(f) . t
  cdot      : (f - mu_v) . t            (ranking within a view is invariant by construction)
  ccos      : unit(f - mu_v) . unit(t)
Geometry: same/different cross-view cosine (after centering), retrieval top1/3/5.
Text: rank of the GT-best mask under each variant.
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


V = {}
for name, cam in cams.items():
    sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
    segv = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
    fgv = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
    lbl = np.array([int(L) for L in np.unique(segv[fgv]) if 0 <= int(L) < sem.shape[0]])
    if lbl.size == 0:
        continue
    F = unit(sem[lbl])
    V[name] = dict(labels=lbl, F=F, C=F - F.mean(0), Cc=unit(F - F.mean(0)))

obj = {}
for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".json")):
    name = im[:-5]
    if name not in V:
        continue
    cam = cams[name]
    h, w = cam.image_height, cam.image_width
    seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
    fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
    labels = V[name]["labels"]
    anno = json.load(open(os.path.join(gt_dir, name + ".json")))
    for prompt in list(dict.fromkeys(o["category"] for o in anno["objects"])):
        gt = np.zeros((h, w), np.uint8)
        for o in anno["objects"]:
            if o["category"] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
        gt = gt.astype(bool)
        best, bL = -1.0, None
        for k, L in enumerate(labels):
            m = (seg == L) & fg
            u = np.logical_or(m, gt).sum()
            v = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            if v > best:
                best, bL = v, k
        obj.setdefault(prompt, []).append((name, bL, best))

t_cache = {}
def text(prompt):
    if prompt not in t_cache:
        vlm.encode_text(prompt)
        t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
        t_cache[prompt] = t / max(np.linalg.norm(t), 1e-8)
    return t_cache[prompt]

# geometry: cross-view retrieval + same/diff cosine, using centered features
res = {}
for tag, key in (("raw_cos", "F"), ("center_cos", "Cc")):
    same, diff, retr = [], [], {1: 0, 3: 0, 5: 0}
    n = 0
    cats = sorted(obj)
    for c in cats:
        fs = obj[c]
        for i in range(len(fs)):
            for j in range(len(fs)):
                if i == j or fs[i][0] == fs[j][0]:
                    continue
                va, la, _ = fs[i]
                vb, lb, _ = fs[j]
                Fa, Fb = V[va][key], V[vb][key]
                sims = Fb @ Fa[la]
                sb = float(Fb[lb] @ Fa[la])
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
            for va, la, _ in obj[c][:2]:
                for vb, lb, _ in obj[d][:2]:
                    diff.append(float(V[vb][key][lb] @ V[va][key][la]))
    res[tag] = dict(same=float(np.median(same)) if same else float("nan"),
                    diff=float(np.median(diff)) if diff else float("nan"),
                    r1=retr[1] / max(n, 1), r3=retr[3] / max(n, 1), r5=retr[5] / max(n, 1), n=n)

# text ranking per variant
rows = []
for prompt, lst in obj.items():
    t = text(prompt)
    for name, bL, best in lst:
        F = V[name]["F"]
        C = V[name]["C"]
        Cc = V[name]["Cc"]
        for tag, sims in (("raw", F @ t), ("cdot", C @ t), ("ccos", Cc @ t)):
            order = np.argsort(-sims)
            rank = int(np.where(order == bL)[0][0]) + 1
            rows.append(dict(scene=args.scene, image=name, prompt=prompt, variant=tag,
                             rank=rank, iou_best=best))

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
with open(args.out.replace(".csv", "_geom.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["variant", "same_med", "diff_med", "retr1", "retr3", "retr5", "n"])
    for tag, d in res.items():
        wr.writerow([tag, d["same"], d["diff"], d["r1"], d["r3"], d["r5"], d["n"]])
print(args.scene, res)
print("SCENE_DONE", args.scene)