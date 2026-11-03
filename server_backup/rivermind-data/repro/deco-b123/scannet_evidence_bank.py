"""Recompute per-view evidence with the SAME text bank that produced the
released scores.

Discovery: `text_features(ViT-B-16, laion2b_s34b_b88k)` has cosine 0.495 with
`text_bank.npz` (10 templates averaged) for the same class names -- i.e. the
released scores use a different text encoder. Every previous evidence experiment
(E0, vote, local contrast) therefore used a mismatched text space.

This script rebuilds the evidence with the bank and re-tests the same operators.
"""
import os, sys, glob, json
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import load_gt, load_gaussians, metrics, SUBSETS


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def z(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def main():
    ap = ArgumentParser()
    for k in ["scene", "model", "input", "gt_root", "released_npz", "bank"]:
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
    n = int(labels_root.max()) + 1
    cls = SUBSETS[19]

    bank = np.load(args.bank)["textbank"].astype(np.float64)   # [T, C, D]
    T = bank.mean(0)
    T = l2n(T)
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    vw = obs["view_ids"].astype(np.int64)
    S = F @ T.T
    key = reg * 100000 + vw
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), len(T)))
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(float)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)

    E = np.zeros((n, len(T)))
    for c in range(len(T)):
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

    votes = np.zeros((n, len(T)))
    for i in range(len(reg)):
        votes[reg[i], S[i].argmax()] += 1
    vote_s = votes / np.maximum(votes.sum(1, keepdims=True), 1)

    zR = z(R)

    variants = {
        "released_raw": R,
        "zR": zR,
        "E0_bank": E,
        "E0_bank_fused": zR + args.beta * z(E),
        "vote_bank_fused": zR + args.beta * z(vote_s),
        "E0_bank_only_z": z(E),
    }
    lr, gk = labels_root[keep], gt[keep]
    out = {}
    for k, M in variants.items():
        pred = np.array(cls)[M.argmax(1)][labels_root]
        m = metrics(gk, pred[keep], cls)
        out[k] = [round(100 * m["mIoU"], 2), round(100 * m["mAcc"], 2)]
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()