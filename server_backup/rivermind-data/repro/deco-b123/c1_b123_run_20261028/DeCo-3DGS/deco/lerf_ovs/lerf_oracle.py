"""LERF-OVS candidate headroom for the DeCo main path (center=image, text=canon, renorm=0, sel=gap).
Per query (GT oracles, diagnostic only):
  main        : released gap selection over top-5 roots (expanded to leaves)  -- parity vs per_query main_iou
  prefix_best : best prefix of the same sorted pool (ceiling of any threshold/gap rule)
  single_best : best single candidate in the pool
  greedy5     : greedy subset of the top-5 pool (ceiling of selection)
  greedy20    : greedy subset of a top-20-root pool (ceiling of ranking + selection)
  v2_c<k>     : V2 adaptive+reliable correction with coefficient k, then the released gap selection
Greedy search uses unions of per-candidate rendered masks; the chosen set is re-rendered exactly."""
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
from evidence_readout_v2 import auxiliary_correction, descriptors_v2, load_bank_v2, load_mode_support
from eval_lerf_ovs_v2 import polygon_to_mask

COEFS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "bank", "obs", "members", "out"]:
    ap.add_argument("--" + k, required=True)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
raw_mean = snag.feat[-1].mean(dim=0).clone()
vlm = ClipSimMeasure()
vlm.load_model()
bank, _ = load_bank_v2(args.bank, args.nag, int(snag.feat[-1].shape[0]), int(snag.feat[-1].shape[1]),
                       observations_path=args.obs, members_path=args.members)
desc = descriptors_v2(bank, snag.feat[-1].device, center=raw_mean, renorm=False)
_, consistency, _, eligible = load_mode_support(args.obs, args.members, bank)
snag.feat = center_torch(snag.feat, renorm=False)

root_level = len(snag.labels) - 1
leaf_level = root_level - 1


def sims_for(prompt, feats=None):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in (feats if feats is not None else snag.feat)]


def pool(sims, n_roots):
    """Released pool construction, with the root cut at n_roots (released: 5 of the top 10)."""
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


def greedy(cands, cam, h, w, gt):
    masks = [render_mask([c], cam, h, w) for c in cands]
    chosen, cur, best = [], np.zeros((h, w), bool), 0.0
    while True:
        gains = [(iou(cur | m, gt), i) for i, m in enumerate(masks) if i not in chosen]
        if not gains:
            break
        v, i = max(gains)
        if v <= best + 1e-9:
            break
        chosen.append(i); cur |= masks[i]; best = v
    exact = iou(render_mask([cands[i] for i in chosen], cam, h, w), gt) if chosen else 0.0
    single = max((iou(m, gt) for m in masks), default=0.0)
    return exact, single


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
            r = dict(scene=args.scene, image=name, prompt=prompt, pool5=len(p5))
            r["main"] = iou(render_mask(gap_prefix(p5), cam, h, w), gt)
            r["prefix_best"] = max([iou(render_mask(p5[:k], cam, h, w), gt) for k in range(1, len(p5) + 1)] or [0.0])
            r["greedy5"], r["single_best"] = greedy(p5, cam, h, w, gt)
            r["greedy20"], _ = greedy(pool(sims, 20), cam, h, w, gt)
            slot = sims_for(prompt, [desc])[0].reshape(bank["priors"].shape)
            for k in COEFS:
                corr, _ = auxiliary_correction(sims[-1], slot, bank, eligible, consistency,
                                               temperature=0.08, coefficient=k) if k > 0 else (sims[-1], None)
                cs = list(sims); cs[-1] = corr
                r["v2_c%g" % k] = iou(render_mask(gap_prefix(pool(cs, 5)), cam, h, w), gt)
            rows.append(r)
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_oracle.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))
