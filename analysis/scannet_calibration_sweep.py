#!/usr/bin/env python3
"""Calibration sweep around C1 on ScanNet-10.

C1 (deco_ops.center_np) is a per-scene, per-class additive offset:
    f <- f - mean_roots(f)   =>   s_c <- s_c - mean_roots(s_c)
It is transductive and GT-free.  This script asks whether the same axis --
class-score calibration inside one scene -- can be pushed further than the
single mean subtraction, using only the released score matrix.

Variants (all per scene, no GT used):
  raw          : released argmax
  center       : column mean subtraction (C1 at the score level)
  zscore       : column standardisation
  median       : column median subtraction
  pc{k}        : remove the top-k principal components of the score matrix
  sinkhorn_lam : entropic prior-matching to a uniform class marginal
  rankq        : per-class quantile normalisation

Evaluation: scene-mean macro mIoU (the audit protocol) and dataset-pooled mIoU
(the RESULTS.md protocol, 18-class divisor).
"""
import argparse
import json
import os
import sys

import numpy as np

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import SUBSETS, metrics  # noqa: E402


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sinkhorn_bias(S, target, tau=0.05, iters=50, lam=1.0):
    """Per-class additive offsets so the softmax marginal approaches target."""
    logits = S / tau
    b = np.zeros(S.shape[1])
    logt = np.log(target + 1e-12)
    for _ in range(iters):
        P = softmax(logits + b[None, :])
        col = P.mean(0)
        b += lam * (logt - np.log(col + 1e-12))
    return S + tau * b[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(os.path.join(args.abl, args.scene, "predictions_and_scores.npz"))
    gt = d["gt"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    S = d["scores_off"].astype(np.float64)
    n_root, K = S.shape
    cls = SUBSETS[19]
    cls_arr = np.array(cls)

    mu = S.mean(0, keepdims=True)
    sd = S.std(0, keepdims=True)
    med = np.median(S, 0, keepdims=True)

    variants = {}
    variants["raw"] = S
    variants["center"] = S - mu
    variants["zscore"] = (S - mu) / np.maximum(sd, 1e-8)
    variants["median"] = S - med

    Sc = S - mu
    U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    for k in (1, 2, 3):
        Pk = U[:, :k] @ U[:, :k].T
        variants["pc%d" % k] = Sc - Pk @ Sc

    target = np.full(K, 1.0 / K)
    for lam in (0.25, 0.5, 1.0):
        variants["sinkhorn_l%g" % lam] = sinkhorn_bias(S, target, lam=lam)
    for it in (1, 2, 5, 10, 20):
        variants["sinkhorn_it%d" % it] = sinkhorn_bias(S, target, iters=it, lam=1.0)

    # per-class quantile normalisation: score -> within-scene quantile in [0,1]
    ranks = np.argsort(np.argsort(S, axis=0), axis=0).astype(np.float64)
    variants["rankq"] = ranks / max(1, n_root - 1)

    out = {"scene": args.scene, "n_root": int(n_root), "variants": {}}
    for name, Sv in variants.items():
        idx = Sv.argmax(1)
        pred = cls_arr[idx][labels]
        m = metrics(gt, pred, cls)
        out["variants"][name] = {
            "macro": float(m["mIoU"]),
            "pooled_inter": {k: int(v["inter"]) for k, v in m["per_class"].items()},
            "pooled_union": {k: int(v["union"]) for k, v in m["per_class"].items()},
        }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cal_%s.json" % args.scene), "w") as f:
        json.dump(out, f)
    print("SCENE", args.scene, " ".join(
        "%s %.4f" % (k, v["macro"]) for k, v in out["variants"].items()), flush=True)


if __name__ == "__main__":
    main()