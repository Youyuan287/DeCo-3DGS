"""Closed-set audit on ScanNet-10.

Incumbent: released label per root, argmax_c (x_root . t_c), x_root = l2(nag_feat[-1]).
Witnesses: per-view observation tokens F_o of the root; verdict s_o(c) = F_o . t_c.
Certificate for challenging class c:
    delta_o = s_o(c) - s_o(L0) over the root's views (one view one vote)
    z_c = mean(delta) / (std(delta) / sqrt(n/2))     (half-sample majority statistic)
Revise L0 -> argmax_c z_c only if max_c z_c > tau. tau=0 is the majority rule;
tau=1.64 is the conservative (one-sided 95%) rule.
Ablations: no certificate (argmax of mean delta), and the known top-3 late interaction.

GT is only used for evaluation.
"""
import os, sys, json, argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scene = args.scene
    gtxyz, gtlabel = load_gt(os.path.join(args.input, f"{scene}_vh_clean_2.labels.ply"))
    xyz, opacity, keep, ply = load_gaussians(args.model, 0)
    dist, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    labels = nag["nag"][-1].numpy().astype(np.int64)
    feats = nag["nag_feat"][-1].numpy().astype(np.float32)
    n_root = int(labels.max()) + 1
    assert len(feats) == n_root

    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    one_word = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                             prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    one_word = l2n(one_word)
    x = l2n(feats)
    scores_off = x @ one_word.T
    base_idx = scores_off.argmax(1)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    w = obs["weights"].astype(np.float64)
    reg = obs["region_ids"].astype(np.int64)
    order = np.argsort(reg, kind="stable")
    reg_s = reg[order]
    F_s = F[order]
    w_s = w[order]

    def root_slice(rid):
        lo = np.searchsorted(reg_s, rid, "left")
        hi = np.searchsorted(reg_s, rid, "right")
        return order[lo:hi]

    # class-prior correction: global mean class score over the scene's observations
    class_mean = (F_s @ one_word.T).mean(0)

    tags = ("audit_majority", "audit95", "no_cert", "vote",
            "audit_cal", "audit_cal95", "rank_audit")
    pred = {t: base_idx.copy() for t in tags}
    n_fire = {t: 0 for t in tags}
    n_obs_roots = 0
    for rid in range(n_root):
        oi = root_slice(rid)
        if len(oi) < 3:
            continue
        n_obs_roots += 1
        X = F_s[oi]
        S = X @ one_word.T                       # (n, C)
        l0 = int(base_idx[rid])

        def fire(tag, c):
            if c != l0:
                pred[tag][rid] = c
                n_fire[tag] += 1

        D = S - S[:, [l0]]                       # paired per-view advantage
        mu = D.mean(0)
        sd = D.std(0)
        k = max(2, len(X) // 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = mu / (sd / np.sqrt(k))
        z = np.where(np.isfinite(z), z, 0.0)
        z[l0] = -1e9
        fire("no_cert", int(np.argmax(mu)))
        c2 = int(np.argmax(z))
        if z[c2] > 0:
            fire("audit_majority", c2)
        if z[c2] > 1.64:
            fire("audit95", c2)

        # class-prior corrected verdict
        Sc = S - class_mean[None, :]
        Dc = Sc - Sc[:, [l0]]
        muc = Dc.mean(0)
        sdc = Dc.std(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = muc / (sdc / np.sqrt(k))
        zc = np.where(np.isfinite(zc), zc, 0.0)
        zc[l0] = -1e9
        c3 = int(np.argmax(zc))
        if zc[c3] > 0:
            fire("audit_cal", c3)
        if zc[c3] > 1.64:
            fire("audit_cal95", c3)

        # ordinal verdict: per-view rank of each class
        R = (-S).argsort(1).argsort(1).astype(np.float64)   # 0 = best
        Dr = R[:, [l0]] - R                          # positive = challenger ranks better
        mur = Dr.mean(0)
        sdr = Dr.std(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zr = mur / (sdr / np.sqrt(k))
        zr = np.where(np.isfinite(zr), zr, 0.0)
        zr[l0] = -1e9
        c4 = int(np.argmax(zr))
        if zr[c4] > 0:
            fire("rank_audit", c4)

        # diagnostic vote: argmax per view, majority
        votes = S.argmax(1)
        vals, cnt = np.unique(votes, return_counts=True)
        fire("vote", int(vals[np.argmax(cnt)]))

    out_metrics = {}
    base_pred = np.array(cls)[base_idx][labels]
    out_metrics["baseline"] = metrics(gt[keep], base_pred[keep], cls)["mIoU"]
    for t, pidx in pred.items():
        p = np.array(cls)[pidx][labels]
        out_metrics[t] = metrics(gt[keep], p[keep], cls)["mIoU"]

    out = dict(scene=scene, n_roots=n_root, n_roots_with_obs=n_obs_roots,
               metrics=out_metrics, fired=n_fire)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{scene}.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("SCENE", scene, "baseline %.4f" % out_metrics["baseline"],
          "audit %.4f audit95 %.4f cal %.4f cal95 %.4f rank %.4f no_cert %.4f vote %.4f" % tuple(
              out_metrics[k] for k in ("audit_majority", "audit95", "audit_cal",
                                       "audit_cal95", "rank_audit", "no_cert", "vote")),
          "fires", {k: n_fire[k] for k in ("audit_majority", "audit_cal", "rank_audit")}, flush=True)


if __name__ == "__main__":
    main()