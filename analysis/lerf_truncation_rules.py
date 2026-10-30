"""Evidence-gated multi-label truncation rules for LERF-OVS (training-free).

Builds on the released gap selection. Three pre-registered rule families, all
driven only by GT-free evidence (candidate screen area + mean observation count):

  R1 include  : if the top-1 candidate is tiny (area < tau) and a much larger
                challenger exists (area > k * top1 area), union the gap prefix
                with the challenger with the highest evidence score.
  R2 re-gap   : same gate, but move the challenger to the front and re-run the
                released gap rule on the reordered candidate list.
  R3 prune    : within the gap prefix, drop candidates whose area < tau2
                (keep the top-1; fall back to top-1 if everything is pruned).

Renders every variant per query against GT (evaluation only). Thresholds are NOT
tuned here; the CPU script does leave-one-scene-out selection over them.
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
for k in ["scene", "source", "model", "nag", "gt_root", "ev", "out"]:
    ap.add_argument("--" + k, required=True)
args = ap.parse_args()

TAUS = [0.001, 0.002, 0.005]
KS = [2.0, 3.0]
TAU2S = [0.001, 0.002, 0.003]

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

evm = {}
with open(os.path.join(args.ev, "%s_evidence.csv" % args.scene), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        evm[(r["image"], r["prompt"], int(r["rank"]))] = (
            float(r["area_ratio"]), float(r["mean_vis"]))


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat]


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


def gap_prefix_idx(c):
    d = [0] + [c[i][3] - c[i + 1][3] for i in range(len(c) - 1)]
    return set(range(d.index(max(d)))) if d else set()


def render_mask(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def zs(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
rows = []
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        for prompt in prompts:
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            sims = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            feats = [evm.get((name, prompt, i + 1)) for i in range(len(p5))]
            if any(f is None for f in feats):
                continue
            area = np.array([f[0] for f in feats])
            mvis = np.array([f[1] for f in feats])
            E = zs(area) + zs(mvis)
            gap = gap_prefix_idx(p5)
            row = dict(scene=args.scene, image=name, prompt=prompt, n=len(p5),
                       gap_len=len(gap),
                       main=iou(render_mask([p5[i] for i in sorted(gap)], cam, h, w), gt))
            champ = int(np.argmax(E))
            for tau in TAUS:
                for k in KS:
                    gate = (area[0] < tau) and (area.max() > k * area[0]) and champ != 0
                    # R1: union with challenger
                    if gate:
                        sel = sorted(set(gap) | {champ})
                    else:
                        sel = sorted(gap)
                    row["R1_t%s_k%g" % (tau, k)] = iou(
                        render_mask([p5[i] for i in sel], cam, h, w), gt)
                    # R2: challenger to front, re-run gap
                    if gate:
                        order = [champ] + [i for i in range(len(p5)) if i != champ]
                        newp = [p5[i] for i in order]
                        sel2 = [order[j] for j in sorted(gap_prefix_idx(newp))]
                    else:
                        sel2 = sorted(gap)
                    row["R2_t%s_k%g" % (tau, k)] = iou(
                        render_mask([p5[i] for i in sel2], cam, h, w), gt)
            for t2 in TAU2S:
                sel3 = [i for i in sorted(gap) if (area[i] >= t2 or i == 0)]
                if not sel3:
                    sel3 = [0]
                row["R3_t%s" % t2] = iou(render_mask([p5[i] for i in sel3], cam, h, w), gt)
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_rules.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))