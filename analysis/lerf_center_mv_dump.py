"""Centered multi-view candidate scoring dump (PREREG_center_mv.md).

For each candidate in the released top-5 pool compute:
  S1_raw/S2_raw     : aggregated raw observation features / top-3 view sims
  S1_center/S2_center: same after per-view mean subtraction of observation features
Covering observations: root = region observations; leaf = support coverage >= 0.4,
dedupe by view, fallback to region observations if <3 (flagged).
No rendering; GT not used.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "sas", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--tau", type=float, default=0.4)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure()
vlm.load_model()
snag.feat = center_torch(snag.feat, renorm=False)
root_level = len(snag.labels) - 1
leaf_level = root_level - 1

obs = np.load(os.path.join(args.sas, "observations_level3.npz"))
F = obs["features"].astype(np.float32)
F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
w_vis = obs["weights"].astype(np.float64)
region = obs["region_ids"].astype(np.int64)
view = obs["view_ids"].astype(np.int64)
side = np.load(os.path.join(args.sas, "observation_semantic_sidecar.npz"))
so = side["support_offsets"].astype(np.int64)
sids = side["support_gaussian_ids"].astype(np.int64)
sw = side["support_weights"].astype(np.float64)
No = len(F)
obs_of_entry = np.repeat(np.arange(No), np.diff(so))
sup_total = np.bincount(obs_of_entry, weights=sw, minlength=No)
order = np.argsort(sids, kind="stable")
sid_sorted = sids[order]; sw_sorted = sw[order]; obs_sorted = obs_of_entry[order]
region_obs = {int(r): np.flatnonzero(region == r) for r in np.unique(region)}

# per-view mean of observation features (unit per observation, mean then unit)
view_mean = {}
for v in np.unique(view):
    m = view == v
    mu = F[m].mean(0)
    view_mean[int(v)] = mu / max(np.linalg.norm(mu), 1e-8)
Fc = np.stack([F[i] - view_mean[int(view[i])] for i in range(No)]).astype(np.float32)
Fc /= np.maximum(np.linalg.norm(Fc, axis=1, keepdims=True), 1e-8)


def entries_for(G):
    lo = np.searchsorted(sid_sorted, G, "left"); hi = np.searchsorted(sid_sorted, G, "right")
    cnt = hi - lo; tot = int(cnt.sum())
    if tot == 0:
        return None, None
    cum = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    idx = np.repeat(lo - cum, cnt) + np.arange(tot)
    return idx, np.repeat(np.arange(len(G)), cnt)


def dedupe_views(obs_ids, cvals):
    if len(obs_ids) == 0:
        return obs_ids, cvals
    o = np.lexsort((cvals, view[obs_ids]))
    oo, cc = obs_ids[o], cvals[o]
    vv = view[oo]
    last = np.r_[vv[1:] != vv[:-1], True]
    return oo[last], cc[last]


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat], vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


lab_root = snag.labels[root_level].cpu().numpy().astype(np.int64)
lab_leaf = snag.labels[leaf_level].cpu().numpy().astype(np.int64)

gt_dir = os.path.join(args.gt_root, args.scene)
rows = []
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        for prompt in prompts:
            sims, t = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            for i, c in enumerate(p5):
                level, rid, cid = c[0], int(c[1]), int(c[2])
                G = np.flatnonzero(lab_root == cid) if level == root_level else np.flatnonzero(lab_leaf == cid)
                if len(G) == 0:
                    continue
                ridx = region_obs.get(rid, np.array([], np.int64))
                if level == root_level:
                    sel = ridx; cvals = np.ones(len(sel)); fb = 0
                    if len(sel) < 3:
                        fb = 1
                else:
                    e_idx, e_g = entries_for(G)
                    if e_idx is None:
                        cvals = np.ones(len(ridx))
                    else:
                        e_obs = obs_sorted[e_idx]; e_w = sw_sorted[e_idx]
                        mass = np.bincount(e_obs, weights=e_w, minlength=No)
                        co = np.clip(mass / np.where(sup_total > 0, sup_total, np.inf), 0, 1)
                        cvals = co[ridx] if len(ridx) else np.array([])
                    sel = ridx[cvals >= args.tau] if len(ridx) else ridx
                    if len(sel):
                        cv_sel = cvals[np.isin(ridx, sel)]
                        sel, cv_sel = dedupe_views(sel, cv_sel)
                    else:
                        cv_sel = np.array([])
                    fb = 0
                    if len(sel) < 3:
                        sel = ridx; cv_sel = np.ones(len(ridx)); fb = 1
                if len(sel) == 0:
                    rows.append(dict(scene=args.scene, image=name, prompt=prompt, rank=i + 1,
                                     S1_raw=float("nan"), S2_raw=float("nan"),
                                     S1_c=float("nan"), S2_c=float("nan"), n_cov=0, fb=1))
                    continue
                a = w_vis[sel] * (cv_sel if level != root_level else 1.0)
                a = a / a.sum()
                Fr = F[sel]; Fce = Fc[sel]
                mr = (a[:, None] * Fr).sum(0); mr /= max(np.linalg.norm(mr), 1e-8)
                mc = (a[:, None] * Fce).sum(0); mc /= max(np.linalg.norm(mc), 1e-8)
                sr = Fr @ t; sc = Fce @ t
                rows.append(dict(scene=args.scene, image=name, prompt=prompt, rank=i + 1,
                                 S1_raw=float(mr @ t), S2_raw=float(np.sort(sr)[-3:].mean()),
                                 S1_c=float(mc @ t), S2_c=float(np.sort(sc)[-3:].mean()),
                                 n_cov=int(len(sel)), fb=int(fb)))
        print("IMG", name, len(prompts), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_center_mv.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))