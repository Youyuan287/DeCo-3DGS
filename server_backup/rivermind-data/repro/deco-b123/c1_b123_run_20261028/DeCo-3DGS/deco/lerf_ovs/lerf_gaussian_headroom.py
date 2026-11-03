"""GT headroom for Gaussian-level boundary refinement inside ReLaGS selections.

For every query:
  - reproduce the released top-5 pool and gap prefix (selected superpoints);
  - build the selected Gaussian set with snag._get_related_gaussians (whole superpoint);
  - build a query-view GT Gaussian set by projecting visible Gaussians into the GT mask;
  - measure selected superpoints/Gaussians, precision/recall/IoU, and the oracles:
      * prune false positives inside selected superpoints (best possible without adding);
      * prune only boundary false positives (GT boundary band in the query view);
      * split selected root superpoints into their leaves and keep leaves with GT purity>=0.5;
      * GT Gaussians outside selected roots (candidate-generation loss).
No rendering; all numbers are Gaussian-set oracles on the query view.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_point
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask
from scipy.ndimage import binary_erosion, binary_dilation

ap = ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
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
snag.feat = center_torch([f.cuda().float() for f in snag.feat], renorm=False)
vlm = ClipSimMeasure()
vlm.load_model()
root_level = len(snag.labels) - 1
leaf_level = root_level - 1
root_np = snag.labels[root_level].detach().cpu().numpy()
leaf_np = snag.labels[leaf_level].detach().cpu().numpy()
n_roots = int(root_np.max()) + 1
xyz = (gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz).detach()


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


def band(mask, h, w, ratio=0.02):
    if not mask.any():
        return np.zeros_like(mask, bool)
    d = max(1, int(round(ratio * np.hypot(h, w))))
    st = np.ones((3, 3), bool)
    edge = mask ^ binary_erosion(mask, structure=st, iterations=1, border_value=0)
    return binary_dilation(edge, structure=st, iterations=d)


def set_iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


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
            sc = np.array([c[3] for c in p5], float)
            g = gap_len(sc)
            cands = [p5[i] for i in range(g)] if g > 0 else []
            if not cands:
                continue
            cur = snag._get_related_gaussians(cands).detach().cpu().numpy().reshape(-1).astype(bool)
            cur_vis = cur & inb

            gt_gauss = np.zeros(len(xyz), bool)
            gt_gauss[inb] = gt[iyn[inb], ixn[inb]]
            n_gt = int(gt_gauss.sum())
            n_sel = int(cur_vis.sum())
            n_sel_gt = int((cur_vis & gt_gauss).sum())

            # Prefix / pool oracles: does adding superpoints help when precision is already low?
            cur_all = snag._get_related_gaussians(p5).detach().cpu().numpy().reshape(-1).astype(bool)
            cur_all_vis = cur_all & inb
            n_all = int(cur_all_vis.sum())
            n_all_gt = int((cur_all_vis & gt_gauss).sum())
            best_pref_iou, best_pref_k = -1.0, 0
            best_single_iou, best_single_k = -1.0, 0
            for kk in range(1, len(p5) + 1):
                ck = snag._get_related_gaussians(p5[:kk]).detach().cpu().numpy().reshape(-1).astype(bool)
                ck = ck & inb
                ik = set_iou(ck, gt_gauss)
                if ik > best_pref_iou:
                    best_pref_iou, best_pref_k = ik, kk
            for ii, c in enumerate(p5):
                ck = snag._get_related_gaussians([c]).detach().cpu().numpy().reshape(-1).astype(bool)
                ck = ck & inb
                ik = set_iou(ck, gt_gauss)
                if ik > best_single_iou:
                    best_single_iou, best_single_k = ik, ii + 1
            prec = n_sel_gt / n_sel if n_sel else float("nan")
            rec = n_sel_gt / n_gt if n_gt else float("nan")

            roots_sel = np.unique(root_np[cur_vis]) if n_sel else np.array([], int)
            root_n = np.bincount(root_np, minlength=n_roots)
            root_gt = np.bincount(root_np[gt_gauss], minlength=n_roots)
            n_root_with_gt = int((root_gt > 0).sum())
            n_root_with_gt_sel = int(((root_gt > 0) & np.isin(np.arange(n_roots), roots_sel)).sum())
            n_gt_in_sel_roots = int(root_gt[roots_sel].sum()) if len(roots_sel) else 0

            gb = np.zeros(len(xyz), bool)
            gband = band(gt, h, w, 0.02)
            gb[inb] = gband[iyn[inb], ixn[inb]]
            n_boundary_vis = int(gb.sum())
            n_boundary_gt = int((gb & gt_gauss).sum())
            n_boundary_sel = int((gb & cur_vis).sum())
            n_boundary_sel_fp = int((gb & cur_vis & ~gt_gauss).sum())
            n_boundary_sel_tp = int((gb & cur_vis & gt_gauss).sum())
            cur_bp = cur_vis & ~(gb & ~gt_gauss)
            iou_bp = set_iou(cur_bp, gt_gauss)
            iou_cur = set_iou(cur_vis, gt_gauss)

            n_leaf_sel_root = n_leaf_pure = n_leaf_empty = n_leaf_mixed = 0
            cur_split = cur_vis.copy()
            n_root_cand = 0
            for (lev, root_id, index, score, _, _, _) in cands:
                if lev != root_level:
                    continue
                n_root_cand += 1
                rm = root_np == index
                cur_split[rm] = False
                leaves = np.unique(leaf_np[rm])
                for lf in leaves:
                    lm = (leaf_np == lf) & inb
                    nn = int(lm.sum())
                    ng = int((lm & gt_gauss).sum())
                    n_leaf_sel_root += 1
                    if ng == 0:
                        n_leaf_empty += 1
                    elif ng == nn:
                        n_leaf_pure += 1
                    else:
                        n_leaf_mixed += 1
                    if nn and ng / nn >= 0.5:
                        cur_split |= lm
            iou_split = set_iou(cur_split, gt_gauss) if n_root_cand else float("nan")

            rows.append(dict(
                scene=args.scene, image=name, prompt=prompt,
                n_pool=len(p5), gap_len=g, n_cands=len(cands),
                n_sel_super=len(cands),
                n_sel_root=sum(1 for c in cands if c[0] == root_level),
                n_sel_leaf=sum(1 for c in cands if c[0] == leaf_level),
                n_scene_gauss=len(xyz), n_visible=inb.sum(),
                n_sel_gauss=n_sel, n_sel_raw=int(cur.sum()),
                n_gt_gauss=n_gt, n_sel_gt=n_sel_gt,
                precision=prec, recall=rec, iou3d=iou_cur,
                n_fp=n_sel - n_sel_gt, n_fn=n_gt - n_sel_gt,
                n_all_gauss=n_all, n_all_raw=int(cur_all.sum()), n_all_gt=n_all_gt,
                all_precision=n_all_gt / n_all if n_all else float("nan"),
                all_recall=n_all_gt / n_gt if n_gt else float("nan"),
                all_iou=set_iou(cur_all_vis, gt_gauss),
                best_pref_iou=best_pref_iou, best_pref_k=best_pref_k,
                best_single_iou=best_single_iou, best_single_k=best_single_k,
                n_root_total=n_roots, n_root_with_gt=n_root_with_gt,
                n_root_sel=len(roots_sel), n_root_with_gt_sel=n_root_with_gt_sel,
                n_gt_in_sel_roots=n_gt_in_sel_roots,
                n_gt_out_sel_roots=n_gt - n_gt_in_sel_roots,
                n_gt_per_sel_root=(n_gt_in_sel_roots / len(roots_sel)) if len(roots_sel) else float("nan"),
                n_boundary_vis=n_boundary_vis, n_boundary_gt=n_boundary_gt,
                n_boundary_sel=n_boundary_sel,
                n_boundary_sel_fp=n_boundary_sel_fp, n_boundary_sel_tp=n_boundary_sel_tp,
                iou_after_boundary_prune=iou_bp,
                n_leaf_sel_root=n_leaf_sel_root, n_leaf_pure=n_leaf_pure,
                n_leaf_empty=n_leaf_empty, n_leaf_mixed=n_leaf_mixed,
                iou_leaf_split=iou_split,
                gt_area=int(gt.sum()),
            ))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_gaussheadroom.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))