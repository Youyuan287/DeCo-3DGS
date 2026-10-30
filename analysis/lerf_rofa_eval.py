"""ROFA-style aggregation variants + end-to-end evaluation (PREREG_rofa.md).

Per candidate: AGG_raw, AGG_rofa (raw consistency), AGG_rofa_center (centered consistency,
raw aggregation). For each variant reorder the released top-5 pool and run the released gap
rule; render the selection. Also render main@5 for parity.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from gaussian_renderer import render
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--tau", type=float, default=0.4)
ap.add_argument("--limit-queries", type=int, default=0)
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
view_mean = {}
for v in np.unique(view):
    m = view == v
    mu = F[m].mean(0)
    view_mean[int(v)] = mu / max(np.linalg.norm(mu), 1e-8)
Fc = F - np.stack([view_mean[int(view[i])] for i in range(No)])
Fc /= np.maximum(np.linalg.norm(Fc, axis=1, keepdims=True), 1e-8)

lab_root = snag.labels[root_level].cpu().numpy().astype(np.int64)
lab_leaf = snag.labels[leaf_level].cpu().numpy().astype(np.int64)
cams = {c.image_name: c for c in scene.getTrainCameras()}
view_index = {name: i for i, name in enumerate(scene.getTrainCameras())}


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


def rofa_keep(X):
    n = len(X)
    if n <= 1:
        return np.ones(n, bool)
    S = X @ X.T
    q = (S.sum(1) - 1) / (n - 1)
    cut = q.mean() - 2.0 * q.std(ddof=0)
    keep = q > cut
    if not keep.any():
        keep[np.argmax(q)] = True
    return keep


def agg_score(idx, weights, feats, t):
    w = weights / max(weights.sum(), 1e-12)
    m = (w[:, None] * feats).sum(0)
    m /= max(np.linalg.norm(m), 1e-8)
    return float(m @ t)


def gap_prefix_scores(scores):
    d = [0] + [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    return d.index(max(d)) if d else 0


def render_sel(sel, cam, h, w):
    rel = snag._get_related_gaussians(sel) if sel else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat], vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


gt_dir = os.path.join(args.gt_root, args.scene)
rows = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        qview = view_index.get(name, -1)
        for prompt in prompts:
            nq += 1
            if args.limit_queries and nq > args.limit_queries:
                break
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            sims, t = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            scores = {tag: [] for tag in ("raw", "rofa", "rofac")}
            removed = {tag: 0 for tag in ("rofa", "rofac")}
            total_views = 0
            qv_removed = {tag: 0 for tag in ("rofa", "rofac")}
            for c in p5:
                level, rid, cid = c[0], int(c[1]), int(c[2])
                G = np.flatnonzero(lab_root == cid) if level == root_level else np.flatnonzero(lab_leaf == cid)
                ridx = region_obs.get(rid, np.array([], np.int64))
                if level == root_level:
                    sel = ridx; cvals = np.ones(len(sel))
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
                    if len(sel) < 3:
                        sel = ridx; cv_sel = np.ones(len(ridx))
                if len(sel) == 0:
                    for tag in scores:
                        scores[tag].append(-1e9)
                    continue
                a = w_vis[sel] * (cv_sel if level != root_level else 1.0)
                total_views += len(sel)
                scores["raw"].append(agg_score(sel, a, F[sel], t))
                keep_r = rofa_keep(F[sel])
                scores["rofa"].append(agg_score(sel[keep_r], a[keep_r], F[sel][keep_r], t))
                removed["rofa"] += int((~keep_r).sum())
                if qview >= 0 and qview in view[sel]:
                    qv_removed["rofa"] += int(not keep_r[np.where(view[sel] == qview)[0][0]])
                keep_c = rofa_keep(Fc[sel])
                scores["rofac"].append(agg_score(sel[keep_c], a[keep_c], F[sel][keep_c], t))
                removed["rofac"] += int((~keep_c).sum())
                if qview >= 0 and qview in view[sel]:
                    qv_removed["rofac"] += int(not keep_c[np.where(view[sel] == qview)[0][0]])
            main_sel = p5[:gap_prefix_scores([c[3] for c in p5])] if p5 else []
            row = dict(scene=args.scene, image=name, prompt=prompt, n=len(p5),
                       total_views=total_views,
                       rem_rofa=removed["rofa"], rem_rofac=removed["rofac"],
                       qv_rem_rofa=qv_removed["rofa"], qv_rem_rofac=qv_removed["rofac"],
                       main_iou=iou(render_sel(main_sel, cam, h, w), gt) if main_sel else 0.0)
            for tag in ("raw", "rofa", "rofac"):
                s = np.array(scores[tag])
                order = np.argsort(-s)
                k = gap_prefix_scores(s[order])
                sel = [p5[i] for i in order[:k]]
                row[tag + "_iou"] = iou(render_sel(sel, cam, h, w), gt) if sel else 0.0
            rows.append(row)
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_rofa.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))