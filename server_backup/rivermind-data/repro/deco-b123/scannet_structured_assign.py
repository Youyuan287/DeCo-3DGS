"""Structured assignment on ScanNet: do per-root labels become better if adjacent
roots are encouraged to agree?

Every method so far assigns each root independently by argmax over classes. The
field's regions are spatially adjacent, so a cheap ICM over a root adjacency
graph adds a neighbour term. This is a different family from score calibration:
it never changes the score matrix, only how labels are chosen from it.
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
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def main():
    ap = ArgumentParser()
    for k in ["scene", "model", "input", "gt_root", "released_npz"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    gt_path = glob.glob(os.path.join(args.gt_root, "*_vh_clean_2.labels.ply"))
    gtxyz, gtlabel = load_gt(gt_path[0])
    xyz, op, keep, ply = load_gaussians(args.model, 0)
    _, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    labels_root = nag["nag"][-1].numpy().astype(np.int64)
    n = int(labels_root.max()) + 1
    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    T = l2n(text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                          prompts=published.PROMPTS).cpu().numpy().astype(np.float32))
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    vw = obs["view_ids"].astype(np.int64)
    S = F @ T.T
    key = reg * 100000 + vw
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), len(cls)))
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(float)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)
    E = np.zeros((n, len(cls)))
    for c in range(len(cls)):
        vals = sv[:, c]
        o = np.lexsort((-vals, vr))
        vr_s, vs = vr[o], vals[o]
        st = np.r_[0, np.flatnonzero(np.diff(vr_s)) + 1]
        ct = np.diff(np.r_[st, len(vr_s)])
        rk = np.arange(len(vr_s)) - np.repeat(st, ct)
        kk = rk < 3
        a = np.bincount(vr_s[kk], weights=vs[kk], minlength=n)
        b = np.bincount(vr_s[kk], minlength=n)
        E[:, c] = a / np.maximum(b, 1)

    base = z(R) + args.beta * z(E)

    cen = np.zeros((n, 3))
    cntr = np.bincount(labels_root, minlength=n).astype(float)
    for k in range(3):
        cen[:, k] = np.bincount(labels_root, weights=xyz[:, k], minlength=n) / np.maximum(cntr, 1)
    kk2 = min(args.knn + 1, n)
    _, nn = cKDTree(cen).query(cen, k=kk2, workers=8)
    nn = np.atleast_2d(nn)
    nbr = [nn[r][nn[r] != r][: args.knn] for r in range(n)]

    lr, gk = labels_root[keep], gt[keep]

    def score(lab):
        pred = np.array(cls)[lab][labels_root]
        m = metrics(gk, pred[keep], cls)
        return 100 * m["mIoU"], 100 * m["mAcc"]

    out = {}
    out["base"] = [round(v, 2) for v in score(base.argmax(1))]
    for lam in (0.5, 1.0, 2.0):
        lab = base.argmax(1).copy()
        for _ in range(args.iters):
            new = lab.copy()
            for r in range(n):
                votes = np.bincount(lab[nbr[r]], minlength=len(cls)).astype(float)
                new[r] = int((base[r] + lam * votes).argmax())
            if np.array_equal(new, lab):
                break
            lab = new
        out["icm_%.1f" % lam] = [round(v, 2) for v in score(lab)]
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()