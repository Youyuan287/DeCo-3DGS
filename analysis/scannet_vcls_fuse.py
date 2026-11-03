"""View-conditioned local scaling (VCLS) on the ReLaGS ScanNet field.

Per (root, view) similarity s_v(r,c) = <f_obs, t_c>; calibrate
    s'_v = s_v - alpha * mean_{r' in view v} s_v(r', c)
then score a root by the mean of its top-k calibrated view similarities and
take the class argmax. alpha=0 reproduces the uncalibrated readout.
"""
import os, sys, glob, json
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import load_gt, load_gaussians, metrics, NYU40, SUBSETS, text_features
import score_scannet as published


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    ap = ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)      # export dir with observations
    ap.add_argument("--gt_root", required=True)    # dir with *_vh_clean_2.labels.ply
    ap.add_argument("--alphas", default="0,0.5,1.0")
    ap.add_argument("--betas", default="0,0.5,1.0,2.0")
    ap.add_argument("--released_npz", default="")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    gt_path = glob.glob(os.path.join(args.gt_root, "*_vh_clean_2.labels.ply"))
    gtxyz, gtlabel = load_gt(gt_path[0])
    xyz, opacity, keep, ply = load_gaussians(args.model, 0)
    dist, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    labels_root = nag["nag"][-1].numpy().astype(np.int64)
    n_root = int(labels_root.max()) + 1
    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    one_word = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                             prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    one_word = l2n(one_word)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    S = F @ one_word.T                                # (n_obs, 19)

    # aggregate observations to (root, view) means
    key = reg.astype(np.int64) * 100000 + view
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), S.shape[1]), np.float64)
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(np.float64)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)
    vv = (uk % 100000).astype(np.int64)

    # per-view mean over roots
    view_mean = {}
    for v in np.unique(vv):
        m = vv == v
        view_mean[int(v)] = sv[m].mean(0)

    alphas = [float(x) for x in args.alphas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]
    # released score: root feature . text (official readout)
    if args.released_npz:
        released = np.load(args.released_npz)["scores_off"].astype(np.float64)
    else:
        root_feat = l2n(nag["nag_feat"][-1].numpy().astype(np.float32))
        released = root_feat @ one_word.T
    out = {}
    for a in alphas:
        cal = sv - a * np.stack([view_mean[int(v)] for v in vv])
        # root score = mean of top-k calibrated view sims
        root_scores = np.full((n_root, len(cls)), -1e9, np.float64)
        for c in range(len(cls)):
            vals = cal[:, c]
            order = np.lexsort((-vals, vr))          # group by root, value desc
            vr_s = vr[order]; vals_s = vals[order]
            starts = np.r_[0, np.flatnonzero(np.diff(vr_s)) + 1]
            counts = np.diff(np.r_[starts, len(vr_s)])
            rank = np.arange(len(vr_s)) - np.repeat(starts, counts)
            keepk = rank < args.topk
            sums_r = np.bincount(vr_s[keepk], weights=vals_s[keepk], minlength=n_root)
            cnt_r = np.bincount(vr_s[keepk], minlength=n_root)
            root_scores[:, c] = sums_r / np.maximum(cnt_r, 1)
        zr = (released - released.mean(0)) / (released.std(0) + 1e-8)
        zv = (root_scores - root_scores.mean(0)) / (root_scores.std(0) + 1e-8)
        for b in betas:
            fused = zr + b * zv
            pred = np.array(cls)[fused.argmax(1)][labels_root]
            mm = metrics(gt[keep], pred[keep], cls)
            key = "a%.1f_b%.1f" % (a, b)
            out[key] = (100 * mm["mIoU"], 100 * mm["mAcc"])
            print("%s  mIoU %.2f  mAcc %.2f" % (key, *out[key]), flush=True)

    print("RESULT", args.scene, json.dumps({k: [round(x, 2) for x in v] for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()