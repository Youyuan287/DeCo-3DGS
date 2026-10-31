"""Per-query gating between module C and mask rerank (PREREG_c_mask_gate.md).

main : released prefix
c    : module C (prune+include) on NAG order
mask : top-g by z(NAG)+z(id_top3_mask)
gate : c if C changes the released selection, else mask
gateb: symmetric (mask if mask changes selection, else c)
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
for k in ["scene", "source", "model", "nag", "gt_root", "out", "preds", "evidence", "id_csv"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--center", default="image", choices=["none", "image"])
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

preds = {}
with open(args.preds, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["scene"] == args.scene:
            preds[(r["image"], r["prompt"], int(r["rank"]))] = float(r["pred"])
ev = {}
with open(os.path.join(args.evidence, "%s_evidence.csv" % args.scene), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ev[(r["image"], r["prompt"], int(r["rank"]))] = (float(r["area_ratio"]), float(r["mean_vis"]))
msig = {}
with open(args.id_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        v = r.get("id_top3", "")
        if v != "":
            msig[(r["image"], r["prompt"], int(r["rank"]))] = float(v)

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
if args.center == "image":
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
    d = [0] + [c[i] - c[i + 1] for i in range(len(c) - 1)]
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
            mm = [msig.get((name, prompt, i + 1)) for i in range(n)]
            if any(x is None for x in pv) or any(x is None for x in vv):
                continue
            mm_arr = np.array([np.nan if x is None else x for x in mm], float)
            mm_arr = np.where(np.isfinite(mm_arr), mm_arr,
                              np.nanmean(mm_arr) if np.isfinite(mm_arr).any() else 0.0)
            sc = np.array([c[3] for c in p5], float)
            g = gap_len(sc)
            S0 = list(range(g)) if g > 0 else []
            area = np.array([v[0] for v in vv], float)
            mvis = np.array([v[1] for v in vv], float)
            A = zs(area) + zs(mvis)
            mp = max(pv)
            prune = [i for i in S0 if i == 0 or pv[i] >= 0.5 * mp] or ([0] if S0 else [])
            champ = int(np.argmax(A))
            include = (area[0] < 0.001) and (area.max() > 2 * area[0]) and champ != 0
            S_C = (prune if mp >= 0.5 else S0)
            if include:
                S_C = sorted(set(S_C) | {champ})
            S = zs(sc) + args.lam * zs(mm_arr)
            order_m = list(np.argsort(-S, kind="stable"))
            S_M = order_m[:g] if g > 0 else []
            S_gate = S_C if set(S_C) != set(S0) else S_M
            S_gateb = S_M if set(S_M) != set(S0) else S_C
            row = dict(scene=args.scene, image=name, prompt=prompt, n=n, gap_len=g, center=args.center)
            for tag, sel in (("main", S0), ("c", S_C), ("mask", S_M), ("gate", S_gate), ("gateb", S_gateb)):
                row[tag + "_len"] = len(sel)
                row[tag + "_iou"] = iou(render_mask([p5[i] for i in sel], cam, h, w), gt) if sel else 0.0
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_cmaskgate_%s.csv" % (args.scene, args.center)), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, args.center, len(rows))