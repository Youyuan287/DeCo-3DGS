"""Axis-agnostic visual calibration: choose alpha per scene by a criterion that
does not touch any label, then measure class-relative separation.

QTC fixes alpha = 0.3 for every scene.  Here alpha is chosen per scene as the one
that minimises the leading-eigenvalue share of the centred, renormalised features
("how much of the field still lies on one axis") -- an unsupervised quantity.

Reported per alpha:
  lead_share   top eigenvalue share of the uncentred second moment   (QTC's metric)
  part_rank    participation rank = (sum lambda)^2 / sum(lambda^2)   (QTC's metric)
  sep_ratio    between-class / within-class scatter of region features
               NB: GT only measures; it never chooses alpha.

Label-free alpha* = argmax part_rank.  If alpha* also maximises sep_ratio, the
criterion is doing what the label-side metric would have wanted.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree


def read_ply_xyz(path):
    v = PlyData.read(path)["vertex"]
    return np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                     np.asarray(v["z"])], 1).astype(np.float64)


def read_ply_lab(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                    np.asarray(v["z"])], 1).astype(np.float64)
    return xyz, np.asarray(v["label"]).astype(np.int64)


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def metrics(F, cls):
    """F (N,d) unit feats, cls (N,) labels >=0 valid.  Returns lead_share, part_rank, sep."""
    F = l2n(F.astype(np.float64))
    M = F.T @ F / F.shape[0]
    ev = np.linalg.eigvalsh(M)[::-1]
    ev = np.clip(ev, 0, None)
    lead = float(ev[0] / ev.sum())
    part = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
    ok = cls >= 0
    Fv, cv = F[ok], cls[ok]
    u = np.unique(cv)
    mu = l2n(Fv.mean(0, keepdims=True))[0]
    cm = np.stack([l2n(Fv[cv == c].mean(0, keepdims=True))[0] for c in u])
    between = float(((cm - mu) ** 2).sum(1).mean())
    within = float(np.mean([((Fv[cv == c] - Fv[cv == c].mean(0)) ** 2).sum(1).mean()
                            for c in u]))
    return lead, part, between / max(within, 1e-12), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0000_00,scene0347_00,scene0070_00")
    ap.add_argument("--model_root", default="/root/verify58_20261029/full10")
    ap.add_argument("--gt", default="/root/rivermind-data/repro/deco-b123/"
                                    "scannet10_full_20261029/gt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for sc in args.scenes.split(","):
        mp = os.path.join(args.model_root, sc, "model")
        d = torch.load(os.path.join(mp, "sai_nag_3.0.pt"), map_location="cpu")
        feat = [f.float() for f in d["nag_feat"]]
        region = d["nag"][-1].numpy().astype(np.int64)          # root partition per gaussian
        g_xyz = read_ply_xyz(os.path.join(mp, "point_cloud", "iteration_0", "point_cloud.ply"))
        gt_xyz, gt_lab = read_ply_lab(os.path.join(args.gt, sc + "_vh_clean_2.labels.ply"))
        g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]
        n_sp = int(region.max()) + 1
        cnt = np.bincount(region * 41 + g_cls, minlength=n_sp * 41).reshape(n_sp, 41)
        tot = cnt.sum(1)
        dom = np.where(tot > 0, cnt.argmax(1), -1)
        F0 = feat[-1].numpy().astype(np.float64)                 # level-last region features
        n = min(len(F0), n_sp)
        F0, dom = F0[:n], dom[:n]
        mu = l2n(F0.mean(0, keepdims=True))
        for a in alphas:
            Fa = l2n(F0 - a * mu)
            lead, part, sep, ncls = metrics(Fa, dom)
            rows.append(dict(scene=sc, alpha=a, lead_share=round(lead, 4),
                             part_rank=round(part, 3), sep_ratio=round(sep, 4),
                             n_classes=ncls, n_regions=int((dom >= 0).sum())))
            print("[%s] a=%.2f lead=%.4f part=%.3f sep=%.4f (classes=%d)"
                  % (sc, a, lead, part, sep, ncls), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "alpha_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for a in alphas:
        r = [x for x in rows if x["alpha"] == a]
        agg[str(a)] = dict(lead=round(float(np.mean([x["lead_share"] for x in r])), 4),
                           part=round(float(np.mean([x["part_rank"] for x in r])), 3),
                           sep=round(float(np.mean([x["sep_ratio"] for x in r])), 4))
    with open(os.path.join(args.out, "alpha_scan.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()