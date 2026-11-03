"""Can a DECISION-level operator work on ScanNet at all?

The certificate fails on ScanNet because roots are not objects. But is the whole
decision-level family dead there, or only that particular witness design?

Test the simplest decision-level operator with existing data: per-view class
argmax, then majority vote per root, against argmax of the mean feature.
If voting beats the mean, decision-level operators can work on ScanNet and the
certificate's failure is specific (witness granularity), not structural.
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
    T = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                      prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    T = l2n(T)
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    S = F @ T.T                       # per (region,view) class scores

    # per-view argmax -> majority vote per root
    votes = np.full((n_root, len(cls)), 0, np.int64)
    per_view_arg = S.argmax(1)
    for i in range(len(reg)):
        votes[reg[i], per_view_arg[i]] += 1
    vote_lab = np.array(cls)[votes.argmax(1)]

    # per-view class scores averaged -> argmax (this is E0 without top-3)
    mean_s = np.zeros((n_root, len(cls)))
    cnt = np.zeros(n_root)
    np.add.at(mean_s, reg, S)
    np.add.at(cnt, reg, 1)
    mean_s /= np.maximum(cnt, 1)[:, None]
    mean_lab = np.array(cls)[mean_s.argmax(1)]

    zR = (R - R.mean(0)) / (R.std(0) + 1e-8)
    rel_lab = np.array(cls)[zR.argmax(1)]

    # zR + beta*z(mean_s)
    def z(x):
        ok = np.isfinite(x)
        out = np.zeros_like(x)
        if ok.any():
            sub = x[ok]
            out[ok] = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)
        return out
    fuse_lab = np.array(cls)[(zR + args.beta * z(mean_s)).argmax(1)]
    vote_s = votes / np.maximum(votes.sum(1, keepdims=True), 1)
    zvote_lab = np.array(cls)[(zR + args.beta * z(vote_s)).argmax(1)]
    zmean_lab = np.array(cls)[(zR + args.beta * z(mean_s)).argmax(1)]

    lr, gk = labels_root[keep], gt[keep]

    def ev(pred_lab):
        pred = pred_lab[labels_root]
        m = metrics(gk, pred[keep], cls)
        return 100 * m["mIoU"], 100 * m["mAcc"]

    out = {}
    for tag, lab in [("vote", vote_lab), ("meanE0", mean_lab),
                     ("releasedZR", rel_lab), ("zR+mean", zmean_lab),
                     ("zR+vote", zvote_lab)]:
        out[tag] = [round(x, 2) for x in ev(lab)]
    # how often vote and mean agree
    both = (cnt >= 1)
    out["agree_vote_mean"] = round(100 * float((vote_lab[both] == mean_lab[both]).mean()), 1)
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()