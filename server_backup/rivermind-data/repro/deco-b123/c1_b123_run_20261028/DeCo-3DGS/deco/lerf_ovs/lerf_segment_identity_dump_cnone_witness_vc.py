"""Phase B segment-level identity dump (PREREG_identity_phaseB.md).

Leaf candidates: per-view 2D-mask assignment of the candidate's own Gaussians
(per-Gaussian mask labels from merge_proj's trace mechanism) -> per-view segment
features -> identity evidence. Root candidates reuse the frozen observation
features (they are exactly the segment-level evidence for a root region).
"""
import os, sys, json, csv
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merge_proj import SAI3D
from gaussian_renderer import trace, render
from nag_data import SemanticNAG
from deco_ops import center_torch
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--tau", type=float, default=0.3)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

ns = Namespace(source_path=args.source, model_path=args.model, iteration=0,
               images="images", resolution=-1, white_background=False,
               data_device="cuda", eval=False, sh_degree=3, vlm_type="clip",
               thres_connect=[.9, .7, .7], thres_merge=20,
               max_neighbor_distance=2, view_freq=5, dis_decay=.5,
               thres_dis=.15, thres_trunc=0., seg_enhance=False, tau=3., seed=0,
               mix="soft", agg="mean", observation_out="",
               render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"])
agent = SAI3D(ns)
agent.init_data(ns)
views = agent.scene.getTrainCameras()
gaussians = agent.gaussians
for v in views:
    if not hasattr(v, "semantic"):
        raise SystemExit("scene has no semantic maps; check dataset layout")

nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure()
vlm.load_model()
raw_mean = snag.feat[-1].mean(dim=0).detach().cpu().numpy().astype(np.float64)
pass  # center=none variant

root_level = len(snag.labels) - 1
leaf_level = root_level - 1
lab_root = snag.labels[root_level].cpu().numpy().astype(np.int64)
lab_leaf = snag.labels[leaf_level].cpu().numpy().astype(np.int64)
N = int(lab_root.shape[0])

obs_npz = np.load(os.path.join(args.sas, "observations_level3.npz"))
oF = obs_npz["features"].astype(np.float64)
oF /= np.maximum(np.linalg.norm(oF, axis=1, keepdims=True), 1e-8)
oW = obs_npz["weights"].astype(np.float64)
oRegion = obs_npz["region_ids"].astype(np.int64)
oView = obs_npz["view_ids"].astype(np.int64)
region_views = {}
for rid in np.unique(oRegion):
    m = oRegion == rid
    region_views[int(rid)] = list(zip(oView[m].tolist(), oW[m].tolist()))

mu_root = {}
mu_leaf = {}
for _v in np.unique(oView):
    _m = oView == _v
    mu_root[int(_v)] = oF[_m].mean(0)
print("rendering per-view mask labels...", flush=True)
per_view = []
with torch.no_grad():
    for vid, view in enumerate(views):
        seg_map = view.semantic["seg_map"][3].clone()
        img_mask = view.semantic["fg_mask"][3]
        if int(img_mask.sum()) == 0:
            per_view.append(None)
            continue
        smin = int(seg_map[img_mask].min().item())
        smax = int(seg_map[img_mask].max().item())
        seg_num = smax - smin + 2
        sg = seg_map - (smin - 1)
        sg[~img_mask] = 0
        enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
        enc = F.normalize(enc, p=2, dim=-1)
        rp = trace(view, gaussians, enc[sg], None, agent.pipeline, agent.background)
        sem = F.normalize(rp["gaussian_semantics"], p=2, dim=-1)
        sim = sem @ enc.T
        lab = sim.argmax(dim=-1)
        lab[sim.max(dim=-1).values <= 0.85] = 0
        vf = view.semantic["sem"][smin:smax + 1].float()
        vf = F.normalize(vf, p=2, dim=-1).cpu().numpy().astype(np.float32)
        per_view.append(dict(lab=lab.cpu().numpy().astype(np.uint8), vf=vf))
        mu_leaf[vid] = vf.mean(0)
        if (vid + 1) % 50 == 0:
            print("view", vid + 1, "/", len(views), flush=True)


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
    sem = render(cam, gaussians, agent.pipeline, agent.background)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def pairwise_mean_cos(X):
    n = len(X)
    if n < 2:
        return float("nan")
    if n > 600:
        X = X[np.linspace(0, n - 1, 600).astype(int)]
        n = len(X)
    G = X @ X.T
    return float((G.sum() - n) / (n * (n - 1)))


def feats_from_views(FVs, ws, mfs, scs, t):
    X = np.stack(FVs)
    a = np.asarray(ws, float)
    a = a / a.sum()
    fmean = (a[:, None] * X).sum(0)
    sims = X @ t
    Fc = X - raw_mean[None, :]
    fmean_c = (a[:, None] * Fc).sum(0)
    return dict(id_mean=float(np.dot(fmean / max(np.linalg.norm(fmean), 1e-12), t)),
                id_top3=float(np.mean(np.sort(sims)[-3:])),
                agree=pairwise_mean_cos(X),
                agree_top=pairwise_mean_cos(X[np.argsort(-sims)[:3]]),
                id_mean_c=float(np.dot(fmean_c, t)),
                agree_c=pairwise_mean_cos(Fc),
                cover_frac=float(np.dot(a, np.asarray(mfs, float))),
                mwp_conf=float(np.dot(a, np.asarray(scs, float))),
                cover_views=int(len(X)))


gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in views}
rows = []
wrows = []
wrows_vc = []
nq = 0
tau = args.tau
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
                rv = region_views.get(int(root_id), [])
                if level == root_level:
                    mask = oRegion == int(root_id)
                    FVs = oF[mask]
                    ws = oW[mask].tolist()
                    vw = oView[mask].astype(np.int64).tolist()
                    mfs = [1.0] * len(ws)
                    scs = [1.0] * len(ws)
                    fb = 0 if len(ws) >= 3 else 1
                else:
                    FVs, ws, mfs, scs = [], [], [], []
                    vw = []
                    for v, wv in rv:
                        pv = per_view[v]
                        if pv is None:
                            continue
                        sub = pv["lab"][G].astype(np.int64)
                        if sub.size == 0:
                            continue
                        mf = float((sub > 0).mean())
                        cnt = np.bincount(sub, minlength=len(pv["vf"]) + 1)[1:]
                        ratio = cnt / float(len(G))
                        keep = np.flatnonzero(ratio >= tau)
                        if len(keep) == 0:
                            continue
                        fv = (ratio[keep, None] * pv["vf"][keep]).sum(0)
                        nrm = np.linalg.norm(fv)
                        if nrm < 1e-8:
                            continue
                        FVs.append(fv / nrm)
                        ws.append(wv)
                        vw.append(int(v))
                        mfs.append(mf)
                        scs.append(float(ratio[keep].max()))
                    fb = 0 if len(FVs) >= 3 else 1
                n_fv = len(FVs) if FVs is not None else 0
                if n_fv > 0:
                    fe = feats_from_views(FVs, ws, mfs, scs, t) if len(FVs) >= 2 else \
                        {k: float("nan") for k in ("id_mean", "id_top3", "agree", "agree_top",
                                                   "id_mean_c", "agree_c")} | \
                        dict(cover_frac=mfs[0] if mfs else float("nan"),
                             mwp_conf=scs[0] if scs else float("nan"),
                             cover_views=len(FVs))
                else:
                    fe = {k: float("nan") for k in ("id_mean", "id_top3", "agree", "agree_top",
                                                    "id_mean_c", "agree_c", "cover_frac",
                                                    "mwp_conf")}
                    fe["cover_views"] = 0
                if fb:
                    for k in ("id_mean", "id_top3", "agree", "agree_top",
                              "id_mean_c", "agree_c"):
                        fe[k] = float("nan")
                if FVs is not None and len(FVs) > 0:
                    _is_root = (level == root_level)
                    for _vid, _fv in zip(vw, FVs):
                        wrows.append((name, prompt, i + 1, int(_vid), float(np.dot(_fv, t))))
                        _mu = mu_root.get(int(_vid)) if _is_root else mu_leaf.get(int(_vid))
                        if _mu is not None:
                            _r = _fv - _mu
                            _n = np.linalg.norm(_r)
                            if _n > 1e-8:
                                wrows_vc.append((name, prompt, i + 1, int(_vid), float(np.dot(_r / _n, t))))
                mask_r = render_mask([c], cam, h, w)
                rows.append(dict(
                    scene=args.scene, image=name, prompt=prompt, rank=i + 1,
                    n_cands=len(p5), level=int(level), is_leaf=int(level == leaf_level),
                    root_id=int(root_id), cand_id=int(cand_id),
                    score=float(score), deficit=float(score - top1),
                    n_points=int(len(G)),
                    area_ratio=float(mask_r.sum()) / (h * w),
                    cover_frac=fe["cover_frac"], cover_views=fe["cover_views"],
                    mwp_conf=fe["mwp_conf"],
                    id_mean=fe["id_mean"], id_top3=fe["id_top3"],
                    agree=fe["agree"], agree_top=fe["agree_top"],
                    id_mean_c=fe["id_mean_c"], agree_c=fe["agree_c"],
                    fallback_seg=int(fb),
                    iou=iou(mask_r, gt)))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_segidentity.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
with open(os.path.join(args.out, "%s_witness.csv" % args.scene), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", "prompt", "rank", "view_id", "sim"])
    wr.writerows(wrows)
with open(os.path.join(args.out, "%s_witness_vc.csv" % args.scene), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", "prompt", "rank", "view_id", "sim"])
    wr.writerows(wrows_vc)
print("SCENE_DONE", args.scene, len(rows), "witness_rows", len(wrows))