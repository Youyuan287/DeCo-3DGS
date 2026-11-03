"""Port the LERF certificate to ScanNet's class axis.

LERF: incumbent set vs challenger set, decided by a half-sample majority over
views using PAIRED per-view evidence. ScanNet has no candidate set, but every
root has a second-best class -- so the same operator applies:

    incumbent  = argmax of the zR-calibrated released score
    challenger = runner-up class (by zR) or best class by per-view evidence
    delta_v    = S_v[root, challenger] - S_v[root, incumbent]
    switch iff a half-sample bootstrap over the root's views has mean>0 in
    at least pi of resamples.

This is the exact analogue of the LERF audit: no re-scoring, only a certified
override. Uses the true text bank.
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
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--challenger", default="runnerup", choices=["runnerup", "evid"])
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

    bank = np.load(args.bank)["textbank"].astype(np.float64)
    T = l2n(bank.mean(0))
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    vw = obs["view_ids"].astype(np.int64)
    S = F @ T.T                       # [n_obs, C] per-view class scores

    # per (root, view) mean over duplicate observations
    key = reg * 100000 + vw
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), len(T)))
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(float)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)

    by_root = {}
    for i in range(len(sv)):
        by_root.setdefault(int(vr[i]), []).append(i)

    zR = z(R)
    inc = zR.argmax(1)

    # per-view evidence per root/class = mean over that root's views
    E = np.zeros((n, len(T)))
    for g, ix in by_root.items():
        E[g] = sv[np.array(ix)].mean(0)

    rng = np.random.default_rng(20261103)
    lab = inc.copy()
    fired = 0
    for g, ix in by_root.items():
        ix = np.array(ix)
        if len(ix) < 5:
            continue
        i0 = int(inc[g])
        if args.challenger == "runnerup":
            tmp = zR[g].copy(); tmp[i0] = -1e18; i1 = int(tmp.argmax())
        else:
            i1 = int(E[g].argmax())
        if i1 == i0:
            continue
        d = sv[ix, i1] - sv[ix, i0]
        k = max(2, len(ix) // 2)
        means = np.array([d[rng.choice(len(ix), k, replace=False)].mean()
                          for _ in range(args.nboot)])
        if (means > 0).mean() >= args.pi:
            lab[g] = i1
            fired += 1

    lr, gk = labels_root[keep], gt[keep]

    def score(labv):
        pred = np.array(cls)[labv][labels_root]
        m = metrics(gk, pred[keep], cls)
        return round(100 * m["mIoU"], 2), round(100 * m["mAcc"], 2)

    out = {
        "released_raw": list(score(R.argmax(1))),
        "zR": list(score(inc)),
        "zR_fused_E": list(score((zR + args.beta * z(E)).argmax(1))),
        "cert": list(score(lab)),
        "fired": fired,
        "roots_with_views": len(by_root),
        "roots": n,
    }
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()