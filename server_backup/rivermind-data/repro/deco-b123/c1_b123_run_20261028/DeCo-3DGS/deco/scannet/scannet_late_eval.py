"""ScanNet 19-class per-root classification with multi-view late interaction.

Baseline: released root feature x @ one_word.T (certified DeCo off protocol).
Variant: for each root, per-view observation tokens F_o; score_class = mean(top-k <F_o, t_class>).
Also computes mean-feature aggregation (control) and top-k with background contrast.
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
    ap.add_argument("--input", required=True)   # .../<scene>/export
    ap.add_argument("--model", required=True)   # .../<scene>/model
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmax", type=int, default=5)
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

    classes = SUBSETS[19]
    names = [NYU40[c] for c in classes]
    one_word = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                             prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    one_word = l2n(one_word)

    x = l2n(feats)
    scores_off = x @ one_word.T

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = obs["features"].astype(np.float32)
    F = l2n(F)
    w = obs["weights"].astype(np.float64)
    reg = obs["region_ids"].astype(np.int64)
    order = np.argsort(reg, kind="stable")
    reg_s = reg[order]
    F_s = F[order]
    w_s = w[order]

    def root_slice(rid):
        lo = np.searchsorted(reg_s, rid, "left"); hi = np.searchsorted(reg_s, rid, "right")
        return order[lo:hi]

    scores = {tag: scores_off.copy() for tag in
              ["top1", "top3", "top5", "meanfeat", "top3_bg"]}
    n_obs_used = 0
    for rid in range(n_root):
        oi = root_slice(rid)
        if len(oi) == 0:
            continue
        n_obs_used += 1
        X = F[oi]                       # (n, D), unit
        S = X @ one_word.T              # (n, C)
        for k in (1, 3, 5):
            kk = min(k, len(X))
            val = np.sort(S, axis=0)[-kk:].mean(0)
            scores["top%d" % k][rid] = val
        a = w[oi]; a = a / max(a.sum(), 1e-12)
        m = (a[:, None] * X).sum(0)
        m = m / max(np.linalg.norm(m), 1e-8)
        scores["meanfeat"][rid] = m @ one_word.T
        # background contrast: tokens of other roots (sample)
        Bidx = np.flatnonzero(reg != rid)
        if len(Bidx) > 1200:
            Bidx = Bidx[np.linspace(0, len(Bidx) - 1, 1200).astype(int)]
        b = (X @ F[Bidx].T).max(1) if len(Bidx) else np.zeros(len(X), np.float32)
        # per-class z of S over tokens, and z of b
        zs = (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-8)
        zb = (b - b.mean()) / (b.std() + 1e-8)
        base = zs - zb[:, None]
        kk = min(3, len(X))
        scores["top3_bg"][rid] = np.sort(base, axis=0)[-kk:].mean(0)

    pred_roots = {tag: np.array(classes)[s.argmax(1)] for tag, s in scores.items()}
    preds = {tag: p[labels] for tag, p in pred_roots.items()}
    base_pred = np.array(classes)[scores_off.argmax(1)][labels]
    results = {"baseline": metrics(gt[keep], base_pred[keep], classes)}
    for tag in scores:
        results[tag] = metrics(gt[keep], preds[tag][keep], classes)
    out = dict(scene=scene, n_gaussians=len(xyz), n_roots=n_root,
               n_roots_with_obs=n_obs_used,
               metrics={k: v["mIoU"] for k, v in results.items()},
               detail=results)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{scene}.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("SCENE", scene, "baseline %.4f" % out["metrics"]["baseline"],
          "top1 %.4f top3 %.4f top5 %.4f mean %.4f bg %.4f" % tuple(
              out["metrics"][k] for k in ["top1", "top3", "top5", "meanfeat", "top3_bg"]), flush=True)


if __name__ == "__main__":
    main()