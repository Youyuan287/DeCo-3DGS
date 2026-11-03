"""LERF-OVS per-candidate dump for the step-2 reranker.

For every query: build the released top-5 candidate pool, render each candidate
individually (gives the IoU label), record candidate-level features (score, rank,
level, sizes, leaf info, mask area) plus query context (top-1 score, gap, n_cands).
Also renders the released gap selection for a per-query parity check.
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
pass  # center=none variant

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
            n = len(p5)
            if n == 0:
                continue
            main = iou(render_mask(gap_prefix(p5), cam, h, w), gt)
            top1s = p5[0][3]
            gap12 = top1s - p5[1][3] if n > 1 else 0.0
            for i, c in enumerate(p5):
                level, root_id, cand_id, score = c[0], c[1], c[2], c[3]
                leaf_idx, leaf_sim, n_points = c[4], c[5], c[6]
                mask = render_mask([c], cam, h, w)
                root_n = int((snag.labels[root_level] == root_id).sum().item())
                root_sc = float(sims[-1][root_id].item())
                if leaf_idx is None:
                    n_leaf = 0
                    leaf_gain = 0.0
                else:
                    n_leaf = int(len(leaf_idx))
                    leaf_gain = float(leaf_sim.max().item() - root_sc) if n_leaf else 0.0
                rows.append(dict(
                    scene=args.scene, image=name, prompt=prompt, rank=i + 1, n_cands=n,
                    level=int(level), is_leaf=int(level == leaf_level),
                    root_id=int(root_id), cand_id=int(cand_id),
                    score=float(score), deficit=float(score - top1s),
                    root_score=root_sc, root_n_points=root_n,
                    n_points=float(n_points), n_points_ratio=float(n_points) / max(root_n, 1),
                    n_leaf=n_leaf, leaf_gain=leaf_gain,
                    area_ratio=float(mask.sum()) / (h * w),
                    gap12=float(gap12), main_iou=main,
                    iou=iou(mask, gt)))
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_candidates.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))