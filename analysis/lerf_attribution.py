"""LERF-OVS Step 0 attribution for the DeCo main path (center=image, text=canon,
renorm=0, sel=gap). For every query it records the gap-prefix length g, the IoU of
every prefix and every single candidate, and the best prefix p*. The gap error is
classified as:
  ok_gap    : main IoU == best prefix IoU (truncation already optimal)
  cut_early : p* > g (gap kept too few candidates; parts missing)
  cut_late  : p* < g (gap kept too many candidates; false positives)
Also records whether the best single candidate is the rank-1 candidate (ranking
quality) so ranking and truncation errors can be told apart.
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
for k in ["scene", "source", "model", "nag", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
for k in ["bank", "obs", "members"]:
    ap.add_argument("--" + k, default=None)
args = ap.parse_args()

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


def gap_prefix(c):
    d = [0] + [c[i][3] - c[i + 1][3] for i in range(len(c) - 1)]
    return c[:d.index(max(d))] if d else []


def render_mask(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


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
            gp = gap_prefix(p5)
            g = len(gp)
            ious = [iou(render_mask(p5[:k], cam, h, w), gt) for k in range(1, len(p5) + 1)]
            singles = [iou(render_mask([c], cam, h, w), gt) for c in p5]
            main = ious[g - 1] if g > 0 else iou(render_mask([], cam, h, w), gt)
            if ious:
                bestk = 1 + int(np.nanargmax(ious))
                prefix_best = float(np.nanmax(ious))
            else:
                bestk, prefix_best = 0, 0.0
            if not singles:
                top1, best_single, best_rank = 0.0, 0.0, 0
            else:
                top1 = singles[0]
                best_rank = 1 + int(np.nanargmax(singles))
                best_single = float(np.nanmax(singles))
            if abs(main - prefix_best) <= 1e-9:
                cat = "ok_gap"
            elif bestk > g:
                cat = "cut_early"
            else:
                cat = "cut_late"
            rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                             n_pool=len(p5), gap_len=g, bestk=bestk,
                             main=main, prefix_best=prefix_best,
                             top1_iou=top1, best_single_iou=best_single,
                             best_single_rank=best_rank, cat=cat))
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_attribution.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))