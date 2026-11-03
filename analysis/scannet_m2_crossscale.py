"""M2: is cross-scale agreement a real, size-independent signal of region quality?

Same 3D region, three mask scales (level 1/2/3 are the ones merge_proj uses).  If a
region's semantics are real it should be scale invariant; if it is carried by
context (a window inside a wall, a table under appliances) the small mask drops the
context and the large mask swallows it, so the feature direction drifts with scale.

Kill test (pre-registered before looking):
  H1  agreement(purity)   > 0, Spearman, and survives controlling for
                            log(region size) and log(views covering the region)
  H2  AUC(agreement -> impure region, purity < 0.90) meaningfully above 0.5
  H3  the L1-L3 gap carries the signal (not just L1-L2 or L2-L3)
If H1 is null and H2 ~ 0.5, cross-scale consistency is not a signal and the idea dies.

Features are the pipeline's own: proj_gaussian_features(root_partition, mask level).
No new encoder, no training, no GT in any feature.  GT is used only to score regions.
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
from scipy.stats import spearmanr, rankdata

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ext"))

from arguments import ArgumentParser, ModelParams, get_combined_args  # noqa: E402
from merge_proj import SAI3D  # noqa: E402


def read_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64)
    return xyz, lab


def residuals(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(residuals(rx, Z), residuals(ry, Z))[0, 1])


def auc(score, label):
    """P(score of a positive > score of a negative), ties counted half."""
    s = rankdata(score)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--labels", default="",
                        help="unused; labels are extracted in-process to keep the "
                             "view order aligned")
    parser.add_argument("--root", required=True,
                        help="npy with the baseline root partition (per gaussian)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--dis_decay", type=float, default=0.5)
    parser.add_argument("--view_freq", type=int, default=5)
    parser.add_argument("--max_neighbor_distance", type=int, default=2)
    parser.add_argument("--thres_trunc", type=float, default=0.0)
    parser.add_argument("--seg_enhance", action="store_true")
    args = get_combined_args(parser)
    args.vlm_type = getattr(args, "vlm_type", "clip")
    tag = os.path.basename(os.path.dirname(args.model_path))
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, tag + "_m2.log"), "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    agent = SAI3D(args)
    agent.init_data(args)
    N = agent.N
    g_xyz = agent.gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    gt_xyz, gt_lab = read_ply(args.gt)
    g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]

    root = np.load(args.root).astype(np.int64)
    if root.min() >= 1:
        root = root - 1
    assert root.shape[0] == N
    n_sp = int(root.max()) + 1
    size = np.bincount(root, minlength=n_sp).astype(np.float64)
    log("[load] %s  gaussians=%d  root regions=%d" % (tag, N, n_sp))

    views = agent.scene.getTrainCameras()
    feats, nviews = {}, np.zeros(n_sp)
    rl = torch.from_numpy(np.array(root, dtype=np.int64, copy=True)).cuda()
    seen = np.zeros((n_sp, len(views)), bool)
    for lvl in (1, 2, 3):
        # NB: extract in-process. Scene does not fix the view order, so labels
        # cached by another process are not aligned with this one's views.
        lab = agent.extract_gaussian_features(agent.gaussians, views, lvl)
        arr = lab.cpu().numpy().astype(np.int16)          # (N, V)
        lab_t = torch.from_numpy(np.array(arr.T, dtype=np.int16,
                                          copy=True)).cuda()   # (V, N)
        vis = arr.T > 0
        rv = np.zeros((n_sp, len(views)), bool)
        for vi in range(len(views)):
            col = vis[vi]
            rv[:, vi] = np.bincount(root[col], minlength=n_sp) > 0
        nviews = rv.sum(1).astype(np.float64)
        log("  level %d: visible-region counts p50=%.0f"
            % (lvl, np.median(nviews)))
        f = agent.proj_gaussian_features(agent.gaussians, views, lvl, rl, lab_t)
        feats[lvl] = f.detach().cpu().numpy()
        del lab_t
        torch.cuda.empty_cache()
        log("  level %d features %s" % (lvl, feats[lvl].shape))

    def cos(a, b):
        na = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        nb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        return (na * nb).sum(1)

    def agree3(f1, f2, f3):
        return (cos(f1, f2) + cos(f2, f3) + cos(f1, f3)) / 3.0

    # centred: remove each level's scene-mean direction first, otherwise every
    # region agrees with every other and the metric measures the common mode
    cen = {}
    for lvl in (1, 2, 3):
        mu = feats[lvl].mean(0, keepdims=True)
        cen[lvl] = feats[lvl] - mu

    c12 = cos(feats[1], feats[2])
    c23 = cos(feats[2], feats[3])
    c13 = cos(feats[1], feats[3])
    agree = agree3(feats[1], feats[2], feats[3])
    agree_c = agree3(cen[1], cen[2], cen[3])

    hist = np.bincount((root * 41 + g_cls).astype(np.int64),
                       minlength=n_sp * 41).reshape(n_sp, 41)
    tot = hist.sum(1)
    ok = tot > 0
    purity = np.where(ok, hist.max(1) / np.maximum(tot, 1), 0.0)

    m = ok & (size >= 50) & (nviews >= 3)
    log("[filter] regions kept %d / %d" % (m.sum(), n_sp))

    x, y = agree[m], purity[m]
    Z = [np.log(size[m]), np.log(nviews[m])]
    impure = (y < 0.90)
    res = dict(scene=tag, n_regions=int(m.sum()), frac_impure=round(float(impure.mean()), 4))
    for name, xx in (("raw", agree[m]), ("centred", agree_c[m])):
        rho, pval = spearmanr(xx, y)
        res["spearman_%s" % name] = round(float(rho), 4)
        res["p_%s" % name] = float(pval)
        res["partial_rho_%s" % name] = round(float(partial_spearman(xx, y, Z)), 4)
        res["auc_%s" % name] = round(float(auc(xx, impure)), 4)
        res["auc_resid_%s" % name] = round(float(auc(residuals(rankdata(xx), Z), impure)), 4)
        res["q_%s" % name] = {str(q): round(float(np.percentile(xx, q)), 3)
                               for q in (10, 50, 90)}
    for a_, b_, nm in ((1, 2, "L1L2"), (2, 3, "L2L3"), (1, 3, "L1L3")):
        cc = cos(cen[a_], cen[b_])[m]
        res["rho_%s_centred" % nm] = round(float(spearmanr(cc, y)[0]), 4)
    res["purity_q"] = {str(q): round(float(np.percentile(y, q)), 3)
                       for q in (10, 50, 90)}
    log("[M2] " + json.dumps(res))

    np.savez(os.path.join(args.out, tag + "_m2_feats.npz"),
             L1=feats[1], L2=feats[2], L3=feats[3],
             root=root, g_cls=g_cls, size=size, nviews=nviews)
    with open(os.path.join(args.out, tag + "_m2_regions.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "size", "nviews", "purity", "agree", "agree_c",
                    "c12", "c23", "c13"])
        for i in np.flatnonzero(m):
            w.writerow([int(i), int(size[i]), int(nviews[i]),
                        round(float(purity[i]), 4), round(float(agree[i]), 4),
                        round(float(agree_c[i]), 4),
                        round(float(c12[i]), 4), round(float(c23[i]), 4),
                        round(float(c13[i]), 4)])
    with open(os.path.join(args.out, tag + "_m2.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()