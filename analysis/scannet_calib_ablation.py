"""Attribution + calibration-scope ablation on the ReLaGS ScanNet field.

Scores:
  R      released (root feature . text)
  E0     per-view evidence, no calibration
  Ev     per-view evidence after VIEW-mean centering  (alpha)
  Es     per-view evidence after SCENE-mean centering (alpha)
Evidence = mean of top-3 per-view similarities of the root.

Variants (argmax over classes):
  raw            R
  zR             per-class z-score of R
  E0 / Ev / Es   evidence only
  R+bzEv         released + beta*z(Ev)          (fusion only, no zR)
  zR+bzE0/Ev/Es  per-class z(R) + beta*z(E_*)   (full)
This separates "class-score recalibration" (zR) from "view/scene centering".
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
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

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
    scene_mean = sv.mean(0)
    view_mean = {int(v): sv[vv == v].mean(0) for v in np.unique(vv)}

    def evidence(cal):
        out = np.full((n_root, len(cls)), -1e9)
        for c in range(len(cls)):
            vals = cal[:, c]
            order = np.lexsort((-vals, vr))
            vr_s = vr[order]; vals_s = vals[order]
            starts = np.r_[0, np.flatnonzero(np.diff(vr_s)) + 1]
            counts = np.diff(np.r_[starts, len(vr_s)])
            rank = np.arange(len(vr_s)) - np.repeat(starts, counts)
            kk = rank < args.topk
            s_r = np.bincount(vr_s[kk], weights=vals_s[kk], minlength=n_root)
            c_r = np.bincount(vr_s[kk], minlength=n_root)
            out[:, c] = s_r / np.maximum(c_r, 1)
        return out

    E0 = evidence(sv)
    Ev = evidence(sv - args.alpha * np.stack([view_mean[int(v)] for v in vv]))
    Es = evidence(sv - args.alpha * scene_mean[None, :])

    zR = (R - R.mean(0)) / (R.std(0) + 1e-8)

    def z(x):
        return (x - x.mean(0)) / (x.std(0) + 1e-8)

    V = {
        "raw": R,
        "zR": zR,
        "E0": E0, "Ev": Ev, "Es": Es,
        "R_bzEv": R + args.beta * z(Ev),
        "zR_bzE0": zR + args.beta * z(E0),
        "zR_bzEv": zR + args.beta * z(Ev),
        "zR_bzEs": zR + args.beta * z(Es),
    }
    out = {}
    for k, sc in V.items():
        pred = np.array(cls)[sc.argmax(1)][labels_root]
        m = metrics(gt[keep], pred[keep], cls)
        out[k] = (100 * m["mIoU"], 100 * m["mAcc"])
    print("RESULT", args.scene, json.dumps({k: [round(a, 2), round(b, 2)] for k, (a, b) in out.items()}), flush=True)


if __name__ == "__main__":
    main()