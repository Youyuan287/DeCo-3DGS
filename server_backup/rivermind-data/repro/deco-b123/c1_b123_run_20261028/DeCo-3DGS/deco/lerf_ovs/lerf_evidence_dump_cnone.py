"""Dump candidate-level orthogonal evidence for LERF-OVS hard-query audit.

For every query and every released top-5 candidate, computes evidence that does
not come from the retrieval score:
  vis_frac   : fraction of the candidate's Gaussians visible in >=1 observation
  mean_vis   : mean log(1+#observations seeing each Gaussian)
  mem_frac   : fraction covered by >=1 observation mask
  mean_mem   : mean log(1+#masks covering each Gaussian)
  bbox_diag  : 3D bounding-box diagonal
  flatness   : smallest/largest covariance eigenvalue (0 = flat plate)
  height     : mean z
Also renders each candidate for the IoU label (GT used for evaluation only).
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
for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
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
xyz_t = gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz
xyz = xyz_t.detach().cpu().numpy().astype(np.float64)
N = xyz.shape[0]

nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure()
vlm.load_model()
pass  # center=none variant

root_level = len(snag.labels) - 1
leaf_level = root_level - 1
lab_root = snag.labels[root_level].cpu().numpy()
lab_leaf = snag.labels[leaf_level].cpu().numpy()

obs = np.load(os.path.join(args.sas, "observations_level3.npz"))
vis_count = np.bincount(obs["visible_gaussian_ids"],
                        minlength=N).astype(np.float64)
mem_count = np.bincount(obs["member_gaussian_ids"],
                        minlength=N).astype(np.float64)
obs_region = obs["region_ids"]
obs_view = obs["view_ids"]
F = torch.from_numpy(obs["features"]).cuda().float()
F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat]


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


def render_mask(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def geom(idx):
    P = xyz[idx]
    if len(P) > 20000:
        P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
    bb = float(np.linalg.norm(P.max(0) - P.min(0)))
    Q = P - P.mean(0)
    ev = np.linalg.eigvalsh(Q.T @ Q / max(len(P) - 1, 1))
    ev = np.clip(ev, 1e-12, None)
    flat = float(ev[0] / ev[-1])
    return bb, flat, float(P[:, 2].mean())


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
            sims_obs = (F @ vlm.text_feature.T).squeeze(-1).cpu().numpy()
            p5 = pool(sims, 5)
            if not p5:
                continue
            top1 = p5[0][3]
            for i, c in enumerate(p5):
                level, root_id, cand_id, score = c[0], c[1], c[2], c[3]
                lab = lab_root if level == root_level else lab_leaf
                idx = np.where(lab == cand_id)[0]
                if len(idx) == 0:
                    continue
                bb, flat, ht = geom(idx)
                mask = render_mask([c], cam, h, w)
                rm = obs_region == int(root_id)
                if rm.any():
                    os_ = sims_obs[rm]
                    omean, ostd, omax = (float(os_.mean()), float(os_.std()),
                                         float(os_.max()))
                    oviews = int(np.unique(obs_view[rm]).size)
                else:
                    omean = ostd = omax = float("nan")
                    oviews = 0
                rows.append(dict(
                    scene=args.scene, image=name, prompt=prompt, rank=i + 1,
                    is_leaf=int(level == leaf_level), score=float(score),
                    deficit=float(score - top1),
                    n_points=float(len(idx)),
                    vis_frac=float(np.mean(vis_count[idx] > 0)),
                    mean_vis=float(np.mean(np.log1p(vis_count[idx]))),
                    mem_frac=float(np.mean(mem_count[idx] > 0)),
                    mean_mem=float(np.mean(np.log1p(mem_count[idx]))),
                    bbox_diag=bb, flatness=flat, height=ht,
                    obs_sim_mean=omean, obs_sim_std=ostd, obs_sim_max=omax,
                    obs_views=oviews,
                    area_ratio=float(mask.sum()) / (h * w),
                    iou=iou(mask, gt)))
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_evidence.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))