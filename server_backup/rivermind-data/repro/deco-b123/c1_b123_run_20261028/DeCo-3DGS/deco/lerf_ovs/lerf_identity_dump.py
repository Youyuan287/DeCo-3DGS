"""Identity-evidence dump for LERF-OVS Phase A (see PREREG_identity.md).

Per candidate: multi-view observation features -> identity evidence
(id_mean / id_top3 / agreement / mask-confirmation) plus rendered IoU label.
Training-free; GT used only to render the IoU label. No pandas here (GPU env).
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
raw_mean = snag.feat[-1].mean(dim=0).detach().cpu().numpy().astype(np.float64)
snag.feat = center_torch(snag.feat, renorm=False)

root_level = len(snag.labels) - 1
leaf_level = root_level - 1
lab_root = snag.labels[root_level].cpu().numpy().astype(np.int64)
lab_leaf = snag.labels[leaf_level].cpu().numpy().astype(np.int64)
N = int(lab_root.shape[0])

obs_npz = np.load(os.path.join(args.sas, "observations_level3.npz"))
F = obs_npz["features"].astype(np.float32)
F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
w_vis = obs_npz["weights"].astype(np.float64)
region = obs_npz["region_ids"].astype(np.int64)
view = obs_npz["view_ids"].astype(np.int64)
side = np.load(os.path.join(args.sas, "observation_semantic_sidecar.npz"))
so = side["support_offsets"].astype(np.int64)
sids = side["support_gaussian_ids"].astype(np.int64)
sw = side["support_weights"].astype(np.float64)
No = len(F)
obs_of_entry = np.repeat(np.arange(No), np.diff(so))
sup_total = np.bincount(obs_of_entry, weights=sw, minlength=No)
order = np.argsort(sids, kind="stable")
sid_sorted = sids[order]
sw_sorted = sw[order]
obs_sorted = obs_of_entry[order]
gmax = np.zeros(N, np.float64)
np.maximum.at(gmax, sids, sw)
gcnt = np.bincount(sids, minlength=N)
region_obs = {int(r): np.flatnonzero(region == r) for r in np.unique(region)}
print("obs", No, "support", len(sids), "gaussians", N, flush=True)


def sims_for(prompt):
    vlm.encode_text(prompt)
    sims = [vlm.compute_similarity(f) for f in snag.feat]
    t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)
    return sims, t


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


def entries_for(G):
    lo = np.searchsorted(sid_sorted, G, "left")
    hi = np.searchsorted(sid_sorted, G, "right")
    cnt = hi - lo
    tot = int(cnt.sum())
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


def selected_entries(sel_obs, e_obs, e_g, e_w):
    if e_obs is None or len(sel_obs) == 0:
        return None
    o = np.sort(sel_obs)
    pos = np.searchsorted(o, e_obs)
    keep = pos < len(o)
    keep[keep] &= (o[pos[keep]] == e_obs[keep])
    if not keep.any():
        return None
    return e_g[keep], e_w[keep]


def cover_stats(sel_obs, G, e_obs, e_g, e_w):
    se = selected_entries(sel_obs, e_obs, e_g, e_w)
    if se is None:
        return float((gcnt[G] > 0).mean()), float("nan")
    eg, ew = se
    covered = np.zeros(len(G), bool)
    covered[eg] = True
    arr = np.zeros(len(G))
    np.maximum.at(arr, eg, ew)
    return float(covered.mean()), float(arr.mean())


def pairwise_mean_cos(X):
    n = len(X)
    if n < 2:
        return float("nan")
    if n > 600:
        X = X[np.linspace(0, n - 1, 600).astype(int)]
        n = len(X)
    G = X @ X.T
    return float((G.sum() - n) / (n * (n - 1)))


def identity_features(sel, cvals_sel, G, t):
    X = F[sel].astype(np.float64)
    a = w_vis[sel] * cvals_sel
    a = a / a.sum()
    fmean = (a[:, None] * X).sum(0)
    sims = X @ t
    top3 = np.argsort(-sims)[:3]
    Fc = X - raw_mean[None, :]
    fmean_c = (a[:, None] * Fc).sum(0)
    return dict(
        id_mean=float(np.dot(fmean / max(np.linalg.norm(fmean), 1e-12), t)),
        id_top3=float(np.mean(np.sort(sims)[-3:])),
        agree=pairwise_mean_cos(X),
        agree_top=pairwise_mean_cos(X[top3]),
        id_mean_c=float(np.dot(fmean_c, t)),
        agree_c=pairwise_mean_cos(Fc))


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
            sims, t = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            top1 = p5[0][3]
            for i, c in enumerate(p5):
                level, root_id, cand_id, score = c[0], c[1], c[2], c[3]
                lab = lab_root if level == root_level else lab_leaf
                G = np.flatnonzero(lab == cand_id)
                if len(G) == 0:
                    continue
                ridx = region_obs.get(int(root_id), np.array([], np.int64))
                if level == root_level:
                    base_obs = ridx
                    e_obs = e_g = e_w = None
                    per_tau = {tt: (int(len(base_obs)), float((gcnt[G] > 0).mean()))
                               for tt in (0.3, 0.4, 0.5)}
                    sel, cvals_sel = base_obs, np.ones(len(base_obs))
                    mwp_fixed = float(gmax[G].mean())
                else:
                    e_idx, e_g = entries_for(G)
                    if e_idx is not None:
                        e_obs = obs_sorted[e_idx]
                        e_w = sw_sorted[e_idx]
                        mass = np.bincount(e_obs, weights=e_w, minlength=No)
                        co = np.clip(mass / np.where(sup_total > 0, sup_total, np.inf), 0, 1)
                        cvals = co[ridx] if len(ridx) else np.array([])
                    else:
                        e_obs = e_w = None
                        cvals = np.ones(len(ridx))
                    base_obs = ridx
                    per_tau = {}
                    for tt in (0.3, 0.4, 0.5):
                        sel_ = base_obs[cvals >= tt] if len(base_obs) else base_obs
                        if len(sel_):
                            sel_, _ = dedupe_views(sel_, cvals[np.isin(base_obs, sel_)])
                        per_tau[tt] = (int(len(sel_)), cover_stats(sel_, G, e_obs, e_g, e_w)[0])
                    thr = args.tau
                    sel_ = base_obs[cvals >= thr] if len(base_obs) else base_obs
                    if len(sel_):
                        sel_, cvals_sel = dedupe_views(sel_, cvals[np.isin(base_obs, sel_)])
                    else:
                        cvals_sel = np.array([])
                    sel = sel_
                    mwp_fixed = None
                fallback = 0
                if len(sel) < 3:
                    sel = base_obs
                    cvals_sel = np.ones(len(sel))
                    fallback = 1
                feats = identity_features(sel, cvals_sel, G, t) if len(sel) else \
                    {k: float("nan") for k in ("id_mean", "id_top3", "agree", "agree_top",
                                               "id_mean_c", "agree_c")}
                cf, mwp = cover_stats(sel, G, e_obs, e_g, e_w)
                if level == root_level:
                    mwp = mwp_fixed
                    cf = per_tau[0.4][1]
                mask = render_mask([c], cam, h, w)
                rows.append(dict(
                    scene=args.scene, image=name, prompt=prompt, rank=i + 1,
                    n_cands=len(p5), level=int(level), is_leaf=int(level == leaf_level),
                    root_id=int(root_id), cand_id=int(cand_id),
                    score=float(score), deficit=float(score - top1),
                    n_points=int(len(G)),
                    area_ratio=float(mask.sum()) / (h * w),
                    n_cov30=per_tau[0.3][0], n_cov40=per_tau[0.4][0],
                    n_cov50=per_tau[0.5][0],
                    cover_frac30=per_tau[0.3][1], cover_frac40=per_tau[0.4][1],
                    cover_frac50=per_tau[0.5][1],
                    cover_views=int(len(np.unique(view[sel]))) if len(sel) else 0,
                    fallback_root=int(fallback),
                    id_mean=feats["id_mean"], id_top3=feats["id_top3"],
                    agree=feats["agree"], agree_top=feats["agree_top"],
                    id_mean_c=feats["id_mean_c"], agree_c=feats["agree_c"],
                    mwp_conf=mwp, cover_frac=cf,
                    iou=iou(mask, gt)))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_identity.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))