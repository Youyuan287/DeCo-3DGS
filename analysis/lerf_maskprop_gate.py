"""GateB + text-selected 2D mask proposal (Stage 2, PREREG_mask_proposal.md).

Computes per query:
  gateB : mask rerank if it changes the released selection, else module C
  prop  : lift the top-1 text-selected SAM mask of the query view
  union : gateB mask OR prop mask
  lowconf_union / lowconf_prop : use union/prop only when max_pred < 0.5, else gateB
All selections are training-free; GT only for IoU.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, render_point
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
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                  dtype=torch.float32, device="cuda")
xyz = (gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz).detach()
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


def render_sel(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def render_bool(bm, cam, h, w):
    rel = bm.float().view(-1, 1).cuda()
    gaussians._semantics = rel.expand(-1, 20)
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
        seg = cam.semantic["seg_map"][3].detach().cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].detach().cpu().numpy().astype(bool)
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        sem = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-8)
        labels = [int(L) for L in np.unique(seg[fg]) if 0 <= int(L) < sem.shape[0]]
        wt = render_point(cam, gaussians, pipe, bg)["weight"].detach().flatten()
        vis = (wt > 1e-4)
        R = torch.from_numpy(np.asarray(cam.R)).float().cuda()
        T = torch.from_numpy(np.asarray(cam.T)).float().cuda()
        pc = xyz @ R + T
        z = torch.clamp(pc[:, 2], min=1e-6)
        fx = w / (2 * np.tan(cam.FoVx * 0.5))
        fy = h / (2 * np.tan(cam.FoVy * 0.5))
        ixn = ((pc[:, 0] / z) * fx + w / 2).round().long().cpu().numpy()
        iyn = ((pc[:, 1] / z) * fy + h / 2).round().long().cpu().numpy()
        frontn = (vis & (pc[:, 2] > 1e-4)).cpu().numpy()
        inb = frontn & (ixn >= 0) & (ixn < w) & (iyn >= 0) & (iyn < h)
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
            S_M = list(np.argsort(-S, kind="stable"))[:g] if g > 0 else []
            gate_sel = S_M if set(S_M) != set(S0) else S_C
            gate_mask = render_sel([p5[i] for i in gate_sel], cam, h, w) if gate_sel else np.zeros((h, w), bool)
            # proposal: top-1 text-selected SAM mask in the query view
            vlm.encode_text(prompt)
            t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
            t /= max(np.linalg.norm(t), 1e-8)
            cand_labels = labels if labels else []
            if not cand_labels:
                continue
            sims_m = {L: float(sem[L] @ t) for L in cand_labels}
            order = sorted(cand_labels, key=lambda L: sims_m[L], reverse=True)
            prop_sim3 = float(np.mean([sims_m[L] for L in order[:3]])) if order else -1e9
            gate_sel_idx = gate_sel if gate_sel else []
            max_mm_gate = max([mm_arr[i] for i in gate_sel_idx], default=-1e9)
            use_prop = prop_sim3 > max_mm_gate
            bm = np.zeros(len(xyz), bool)
            m = (seg == order[0]) & fg
            bm[inb] = m[iyn[inb], ixn[inb]]
            prop_mask = render_bool(torch.from_numpy(bm), cam, h, w) if bm.any() else np.zeros((h, w), bool)
            union = np.logical_or(gate_mask, prop_mask)
            lowconf = mp < 0.5
            sel_union = union if lowconf else gate_mask
            sel_prop = prop_mask if lowconf else gate_mask
            sel_ev = prop_mask if use_prop else gate_mask
            sel_ev_union = union if use_prop else gate_mask
            rows.append(dict(scene=args.scene, image=name, prompt=prompt, n=n, gap_len=g,
                             max_pred=mp, lowconf=int(lowconf),
                             prop_sim3=prop_sim3, max_mm_gate=float(max_mm_gate),
                             use_prop=int(use_prop),
                             gate_iou=iou(gate_mask, gt), prop_iou=iou(prop_mask, gt),
                             union_iou=iou(union, gt),
                             lowconf_union_iou=iou(sel_union, gt),
                             lowconf_prop_iou=iou(sel_prop, gt),
                             sel_ev_iou=iou(sel_ev, gt),
                             sel_ev_union_iou=iou(sel_ev_union, gt),
                             sim_margin=float(sims_m[order[0]] - sims_m[order[1]]) if len(order) > 1 else 0.0))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_maskprop_gate.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))