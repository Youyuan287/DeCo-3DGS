"""Module v2 grid: prune threshold alpha x learner-inclusion beta
(PREREG_module_v2.md). Nine frozen configurations; scale-gated inclusion always on.
GT only for IoU. No pandas (GPU env).
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "out", "preds", "evidence"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

ALPHAS = [0.3, 0.4, 0.5, 0.6]
BETAS = [0.8, 1.0]
CONFIGS = []
for a in ALPHAS:
    for b in BETAS:
        CONFIGS.append((a, b, "a%02d_b%02d" % (round(a * 100), round(b * 100))))
CONFIGS.append((0.5, None, "a05_off"))

preds = {}
with open(args.preds, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["scene"] == args.scene:
            preds[(r["image"], r["prompt"], int(r["rank"]))] = float(r["pred"])
ev = {}
with open(os.path.join(args.evidence, "%s_evidence.csv" % args.scene), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ev[(r["image"], r["prompt"], int(r["rank"]))] = (float(r["area_ratio"]), float(r["mean_vis"]))

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                  dtype=torch.float32, device="cuda")
nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure()
vlm.load_model()
snag.feat = center_torch(snag.feat, renorm=False)
root_level = len(snag.labels) - 1
leaf_level = root_level - 1


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat]


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


def gap_len(c):
    d = [0] + [c[i][3] - c[i + 1][3] for i in range(len(c) - 1)]
    return d.index(max(d)) if d else 0


def render_mask(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
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
rows = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
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
            sims = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            n = len(p5)
            pv = [preds.get((name, prompt, i + 1)) for i in range(n)]
            vv = [ev.get((name, prompt, i + 1)) for i in range(n)]
            if any(x is None for x in pv) or any(x is None for x in vv):
                continue
            area = np.array([v[0] for v in vv], float)
            mvis = np.array([v[1] for v in vv], float)
            A = zs(area) + zs(mvis)
            g = gap_len(p5)
            S0 = list(range(g)) if g > 0 else []
            mp = max(pv)
            champ = int(np.argmax(A))
            include_gate = (area[0] < 0.001) and (area.max() > 2 * area[0]) and champ != 0
            row = dict(scene=args.scene, image=name, prompt=prompt, n=n, gap_len=g, max_pred=mp)
            row["s0_len"] = len(S0)
            row["s0_iou"] = iou(render_mask([p5[i] for i in S0], cam, h, w), gt) if S0 else 0.0
            for a, b, tag in CONFIGS:
                prune = ([i for i in S0 if i == 0 or pv[i] >= a * mp] or ([0] if S0 else [])) \
                    if mp >= 0.5 else list(S0)
                S = set(prune)
                if include_gate:
                    S.add(champ)
                if b is not None:
                    for i in range(n):
                        if i not in S and pv[i] >= b * mp:
                            S.add(i)
                sel = sorted(S)
                row[tag + "_len"] = len(sel)
                row[tag + "_iou"] = iou(render_mask([p5[i] for i in sel], cam, h, w), gt) if sel else 0.0
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_modv2.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))