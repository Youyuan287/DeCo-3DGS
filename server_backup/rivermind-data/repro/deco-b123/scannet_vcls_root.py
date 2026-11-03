"""Root-level paired outcomes for released / per-class z-score / z-score+VCLS.

For each root with observations: predicted class = argmax score; ground truth =
majority GT class of the root's Gaussians. Saves per-root correctness arrays so
a paired bootstrap over roots (scene-stratified) can be run offline.
"""
import os, sys, glob
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import load_gt, load_gaussians, SUBSETS, NYU40, text_features
import score_scannet as published


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    ap = ArgumentParser()
    for k in ["scene", "model", "input", "gt_root", "released_npz", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
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
    released = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    S = F @ one_word.T
    key = reg * 100000 + view
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), S.shape[1])); np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(np.float64)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64); vv = (uk % 100000).astype(np.int64)
    view_mean = {int(v): sv[vv == v].mean(0) for v in np.unique(vv)}
    cal = sv - args.alpha * np.stack([view_mean[int(v)] for v in vv])
    vcls = np.full((n_root, len(cls)), -1e9)
    for c in range(len(cls)):
        vals = cal[:, c]
        order = np.lexsort((-vals, vr))
        vr_s = vr[order]; vals_s = vals[order]
        starts = np.r_[0, np.flatnonzero(np.diff(vr_s)) + 1]
        counts = np.diff(np.r_[starts, len(vr_s)])
        rank = np.arange(len(vr_s)) - np.repeat(starts, counts)
        kk = rank < args.topk
        sums_r = np.bincount(vr_s[kk], weights=vals_s[kk], minlength=n_root)
        cnt_r = np.bincount(vr_s[kk], minlength=n_root)
        vcls[:, c] = sums_r / np.maximum(cnt_r, 1)
    observed = cnt_r > 0

    zr = (released - released.mean(0)) / (released.std(0) + 1e-8)
    zv = (vcls - vcls.mean(0)) / (vcls.std(0) + 1e-8)
    fused = zr + args.beta * zv

    # majority GT per root
    gt_major = np.full(n_root, -1, np.int64)
    for r in range(n_root):
        m = labels_root[keep] == r
        if m.sum() == 0:
            continue
        b = np.bincount(gt[keep][m])
        gt_major[r] = int(b.argmax())

    ok = observed & (gt_major >= 0)
    p_rel = np.array(cls)[released.argmax(1)]
    p_zs = np.array(cls)[zr.argmax(1)]
    p_fu = np.array(cls)[fused.argmax(1)]
    np.savez(args.out, root=np.flatnonzero(ok), gt=gt_major[ok],
             rel=p_rel[ok], zs=p_zs[ok], fu=p_fu[ok])
    print("ROOT", args.scene, "roots", int(ok.sum()), flush=True)


if __name__ == "__main__":
    main()