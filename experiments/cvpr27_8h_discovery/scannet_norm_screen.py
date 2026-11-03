"""Screen normalization operators on the ScanNet score matrix.

Everything so far normalises ONE axis at a time (zR = column affine; a_g = row).
This screens operators that act on the matrix as a whole, plus multiplicative
variants, against the same strong baseline zR + beta*z(E0) = 48.77.

All operators are training-free and use no GT. Same candidates, same evidence.
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


def z(x):
    # per-column (per-class) standardisation; x[bool] would flatten, so do NOT mask
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def sinkhorn(M, T, iters=30):
    """Doubly-stochastic normalisation of exp(M/T). Rows are no-ops for a row
    argmax only when applied alone; here they interact through the columns."""
    P = np.exp((M - M.max()) / T)
    for _ in range(iters):
        P /= np.maximum(P.sum(0, keepdims=True), 1e-12)
        P /= np.maximum(P.sum(1, keepdims=True), 1e-12)
    return P


def main():
    ap = ArgumentParser()
    for k in ["scene", "model", "input", "gt_root", "released_npz"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    gt_path = glob.glob(os.path.join(args.gt_root, "*_vh_clean_2.labels.ply"))
    gtxyz, gtlabel = load_gt(gt_path[0])
    xyz, op, keep, ply = load_gaussians(args.model, 0)
    _, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    labels_root = nag["nag"][-1].numpy().astype(np.int64)
    n_root = int(labels_root.max()) + 1
    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    T = l2n(text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                          prompts=published.PROMPTS).cpu().numpy().astype(np.float32))
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    S = F @ T.T
    key = reg * 100000 + obs["view_ids"].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), len(cls)))
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(np.float64)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)
    E = np.zeros((n_root, len(cls)))
    for c in range(len(cls)):
        vals = sv[:, c]
        order = np.lexsort((-vals, vr))
        vr_s, vals_s = vr[order], vals[order]
        starts = np.r_[0, np.flatnonzero(np.diff(vr_s)) + 1]
        counts = np.diff(np.r_[starts, len(vr_s)])
        rank = np.arange(len(vr_s)) - np.repeat(starts, counts)
        kk = rank < 3
        s_r = np.bincount(vr_s[kk], weights=vals_s[kk], minlength=n_root)
        c_r = np.bincount(vr_s[kk], minlength=n_root)
        E[:, c] = s_r / np.maximum(c_r, 1)

    zR = (R - R.mean(0)) / (R.std(0) + 1e-8)
    base = zR + args.beta * z(E)
    sd = float(base.std())

    ops = {"base_zR+bE0": base, "E0_alone": E, "zR_alone": zR, "raw": R}
    # additive column centering of the fused matrix (removes class offset once more)
    ops["colsub"] = base - base.mean(0)
    # multiplicative column normalisation (positive shift first so all > 0)
    P0 = base - base.min() + 1e-6
    ops["colmul"] = P0 / np.maximum(P0.sum(0, keepdims=True), 1e-12)
    # per-column rank (distribution-free)
    rk = np.zeros_like(base)
    for c in range(base.shape[1]):
        o = base[:, c].argsort().argsort().astype(float)
        rk[:, c] = o / max(len(o) - 1, 1)
    ops["colrank"] = rk
    # Sinkhorn at several temperatures
    for tm in (0.25, 0.5, 1.0):
        ops["sinkhorn_T%.2f" % (tm * sd)] = sinkhorn(base, tm * sd)
    # Sinkhorn on the raw released scores (no evidence fusion), several temperatures
    rstd = float(R.std())
    for tm in (0.02, 0.05, 0.1, 0.25, 0.5):
        ops["sinkraw_%d" % int(tm * 1000)] = sinkhorn(R, tm * rstd)
    for tm in (0.05, 0.25):
        ops["sinkfuse_%d" % int(tm * 1000)] = sinkhorn(base, tm * sd)

    lr, gk = labels_root[keep], gt[keep]
    out = {}
    for k, M in ops.items():
        pred = np.array(cls)[M.argmax(1)][labels_root]
        m = metrics(gk, pred[keep], cls)
        out[k] = [round(100 * m["mIoU"], 2), round(100 * m["mAcc"], 2)]
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()