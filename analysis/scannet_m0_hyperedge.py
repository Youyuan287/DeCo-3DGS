"""M0: measure the 2D-mask / superpoint relation that the merge stage actually sees.

Three quantities decide whether a Lifted-Multicut or Hypergraph reformulation can
do anything at all:

  hyperedge size   how many superpoints one mask holds (>=30% share, merge_proj's
                   threshold).  If this is ~2 the hypergraph IS the pairwise graph.
  mask purity      GT class concentration inside one mask.  If masks are ~pure the
                   "all superpoints in a mask are one region" constraint is sound;
                   if they mix classes it is wrong by construction.
  masks per sp     the dual quantity; reproduces merge_proj's own mixture statistic.

Superpoints are the ones merge_proj actually consumes (nag-l1.pt), not the readout
root.  GT is used to score purity only: nothing here proposes, selects or merges.

No CLIP, no text, no rendering of features, no evaluation harness.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ext"))

from arguments import (ArgumentParser, ModelParams, PipelineParams,  # noqa: E402
                       get_combined_args)
from gaussian_renderer import trace  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402

NYU40 = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator",
    25: "television", 26: "paper", 27: "towel", 28: "shower curtain", 29: "box",
    30: "whiteboard", 31: "person", 32: "nightstand", 33: "toilet", 34: "sink",
    35: "lamp", 36: "bathtub", 37: "bag", 38: "otherstructure",
    39: "otherfurniture", 40: "otherprop",
}


def read_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64)
    return xyz, lab


def q(a, ps=(10, 25, 50, 75, 90, 99)):
    a = np.asarray(a, np.float64)
    if a.size == 0:
        return {}
    return {int(p): float(np.percentile(a, p)) for p in ps}


def extract_gaussian_features(view, gaussians, pipeline, background, level):
    """Verbatim copy of SAI3D.extract_gaussian_features for one view.

    Imported from merge_proj in the pipeline, but that module pulls in open3d,
    which this env does not have.  The function itself only needs the renderer.
    """
    seg_map = view.semantic["seg_map"][level].clone()
    img_mask = view.semantic["fg_mask"][level]
    seg_min, seg_max = seg_map[img_mask].min().item(), seg_map[img_mask].max().item()
    seg_num = seg_max - seg_min + 2
    seg_map -= seg_min - 1
    seg_map[~img_mask] = 0
    enc = torch.normal(mean=0, std=1, size=(seg_num, 20), device="cuda")
    enc = torch.nn.functional.normalize(enc, p=2, dim=-1)
    feature_map = enc[seg_map]
    pkg = trace(view, gaussians, feature_map, None, pipeline, background)
    gau_sem = torch.nn.functional.normalize(pkg["gaussian_semantics"], p=2, dim=-1)
    sim = torch.matmul(gau_sem, enc.T)
    lab = sim.argmax(dim=-1)
    lab[sim.max(dim=-1)[0] <= 0.85] = 0
    return lab


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--share", type=float, default=0.3)
    parser.add_argument("--iteration", type=int, default=0)
    args = get_combined_args(parser)
    pargs = model.extract(args)

    pipeline = PipelineParams(ArgumentParser())
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    gaussians = GaussianModel(3, 0)
    scene = Scene(pargs, gaussians, load_iteration=args.iteration,
                  shuffle=False, load_sem=True, load_test=False, vlm_type="clip")
    views = scene.getTrainCameras()
    N = gaussians.get_xyz.shape[0]

    sp = torch.load(os.path.join(pargs.model_path, "nag-l1.pt"),
                    map_location="cpu")
    sp = np.asarray(sp).astype(np.int64).ravel()
    if sp.min() >= 1:
        sp = sp - 1
    assert sp.shape[0] == N, (sp.shape, N)
    n_sp = int(sp.max()) + 1
    print("[load] gaussians=%d superpoints=%d views=%d level=%d"
          % (N, n_sp, len(views), args.level), flush=True)

    g_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    gt_xyz, gt_lab = read_ply(args.gt)
    g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]

    pick = np.linspace(0, len(views) - 1, args.views).astype(int)
    rows, hsz_all, pur_all, mps_all, n_ok = [], [], [], [], 0

    with torch.no_grad():
        for k, vi in enumerate(pick):
            view = views[vi]
            lab = extract_gaussian_features(view, gaussians, pipeline,
                                            background, args.level)
            lab = lab.cpu().numpy().astype(np.int64)
            ok = lab > 0
            if not ok.any():
                continue
            m = lab[ok]
            c = sp[ok]
            cl = g_cls[ok]
            nm = int(m.max()) + 1

            key = c.astype(np.int64) * 4096 + m
            uniq, cnt = np.unique(key, return_counts=True)
            su, mu = uniq // 4096, uniq % 4096

            tot = np.bincount(c, minlength=n_sp)
            share = cnt / np.maximum(tot[su].astype(np.float64), 1.0)
            sel = share > args.share
            su_s, mu_s = su[sel], mu[sel]

            hsz = np.bincount(mu_s, minlength=nm)
            hsz = hsz[hsz > 0]
            mps = np.bincount(su_s, minlength=n_sp)
            mps = mps[mps > 0]

            hist = np.bincount((m * 41 + cl).astype(np.int64),
                               minlength=nm * 41).reshape(nm, 41)
            tot_m = hist.sum(1)
            keep = tot_m > 0
            pur = hist[keep].max(1) / tot_m[keep]
            dom = hist[keep].argmax(1)

            hsz_all.append(hsz)
            pur_all.append(pur)
            mps_all.append(mps)
            n_ok += 1
            rows.append(dict(scene=os.path.basename(os.path.dirname(
                pargs.model_path)), view=vi, frame=view.image_name,
                n_masks=int(len(hsz)), n_gauss=int(ok.sum()),
                hyperedge_p50=float(np.median(hsz)),
                hyperedge_max=int(hsz.max()),
                masks_per_sp=float(mps.mean()),
                purity_p50=float(np.median(pur)),
                purity_mean=float(pur.mean())))
            if k % 5 == 0:
                print("  view %2d/%d  masks=%3d  hyperedge p50=%5.1f max=%4d  "
                      "masks/sp=%.2f  purity p50=%.2f"
                      % (k + 1, len(pick), len(hsz), np.median(hsz), hsz.max(),
                         mps.mean(), np.median(pur)), flush=True)

    hsz_all = np.concatenate(hsz_all) if hsz_all else np.array([])
    pur_all = np.concatenate(pur_all) if pur_all else np.array([])
    mps_all = np.concatenate(mps_all) if mps_all else np.array([])

    # what a mask looks like: how many GT classes, and is it one-class-dominant
    multi = None
    summary = dict(
        scene=os.path.basename(os.path.dirname(pargs.model_path)),
        level=args.level, views_used=n_ok, n_superpoints=n_sp,
        share_thr=args.share,
        hyperedge_size=q(hsz_all), hyperedge_mean=float(hsz_all.mean()),
        hyperedge_eq2_frac=float((hsz_all <= 2).mean()),
        hyperedge_eq1_frac=float((hsz_all == 1).mean()),
        masks_per_sp_mean=float(mps_all.mean()),
        masks_per_sp_ge2_frac=float((mps_all >= 2).mean()),
        purity=q(pur_all), purity_mean=float(pur_all.mean()),
        purity_ge090=float((pur_all >= 0.90).mean()),
    )
    os.makedirs(args.out, exist_ok=True)
    tag = summary["scene"]
    with open(os.path.join(args.out, tag + "_m0.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if rows:
        with open(os.path.join(args.out, tag + "_m0_views.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("[M0] " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()