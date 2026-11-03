#!/usr/bin/env python3
"""Are impure clusters inconsistent across views, or consistent but unassignable?

For every root cluster, take the dominant 2D mask in each view that sees it and
compare those per-view CLIP features: mean pairwise cosine (feature dispersion) and
cosine to the cluster centroid. Join with the cluster's GT purity.

  dispersion HIGH on impure clusters  -> the cluster genuinely mixes objects, and the
                                         loss is the partition, not the descriptor.
  dispersion LOW even when impure      -> the observations agree but the semantics do
                                         not separate; the loss is CLIP as a region
                                         descriptor (or the text side).

No GPU forward, no rendering: reads the dump written by scannet_mask_mixture_probe.py.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ext"))

from arguments import ModelParams, get_combined_args  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--obs", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iteration", type=int, default=0)
    args = get_combined_args(parser)
    pargs = model.extract(args)

    gaussians = GaussianModel(3, 0)
    scene = Scene(pargs, gaussians, load_iteration=args.iteration, shuffle=False,
                  load_sem=True, load_test=False, vlm_type="clip")
    views = scene.getTrainCameras()
    z = np.load(args.obs)
    if "names" in z.files:
        want = [str(n) for n in z["names"]]
        by_name = {v.image_name: v for v in views}
        if [v.image_name for v in views] != want and all(n in by_name for n in want):
            views = [by_name[n] for n in want]
            print("view order realigned to the dump: %d views" % len(views))

    sp, view, mask, cnt = z["sp"], z["view"], z["mask"], z["count"]
    n_sp = int(z["n_sp"])

    feats = []
    for v in views:
        f = v.semantic["sem"].detach().float().cpu().numpy()
        feats.append(f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8))

    # dominant mask per (cluster, view)
    key = sp.astype(np.int64) * 100000 + view
    order = np.argsort(key, kind="stable")
    key, sp_s, view_s, mask_s, cnt_s = key[order], sp[order], view[order], mask[order], cnt[order]
    bounds = np.flatnonzero(np.diff(key)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, key.size]

    dom_view = np.zeros(n_sp, np.int64)
    disp = np.full(n_sp, np.nan)
    to_centroid = np.full(n_sp, np.nan)

    # step 1: dominant mask per (cluster, view)
    rec = np.empty((len(starts), 3), np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        j = a + int(np.argmax(cnt_s[a:b]))
        rec[i] = (sp_s[a], view_s[j], mask_s[j])

    # step 2: dispersion over the views of each cluster
    order2 = np.argsort(rec[:, 0], kind="stable")
    rec = rec[order2]
    b2 = np.flatnonzero(np.diff(rec[:, 0])) + 1
    s2 = np.r_[0, b2]
    e2 = np.r_[b2, rec.shape[0]]
    for a, b in zip(s2, e2):
        s = int(rec[a, 0])
        vs, ms = rec[a:b, 1], rec[a:b, 2]
        keep = np.array([m < feats[v].shape[0] for v, m in zip(vs, ms)], bool)
        if keep.sum() < 2:
            continue
        Z = np.stack([feats[int(v)][int(m)] for v, m in zip(vs[keep], ms[keep])])
        n = Z.shape[0]
        G = Z @ Z.T
        disp[s] = float((G.sum() - n) / (n * (n - 1)))
        c = Z.mean(0)
        c /= max(np.linalg.norm(c), 1e-8)
        to_centroid[s] = float((Z @ c).mean())
        dom_view[s] = n

    rows = {int(r["cluster"]): r for r in csv.DictReader(open(args.rows))}
    out = []
    for i in range(n_sp):
        r = rows.get(i)
        if r is None or not np.isfinite(disp[i]):
            continue
        out.append(dict(scene=r["scene"], cluster=i, purity=float(r["purity"]),
                        n_gauss=int(r["n_gauss"]), n_views=int(dom_view[i]),
                        disp=float(disp[i]), to_centroid=float(to_centroid[i])))
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "featdisp_%s.csv" % os.path.basename(pargs.model_path))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    pur = np.array([o["purity"] for o in out])
    dd = np.array([o["disp"] for o in out])
    tc = np.array([o["to_centroid"] for o in out])
    nv = np.array([o["n_views"] for o in out])
    print("%s clusters=%d  corr(purity,disp)=%+.3f  corr(purity,to_centroid)=%+.3f"
          % (os.path.basename(pargs.model_path), len(out),
             float(np.corrcoef(pur, dd)[0, 1]), float(np.corrcoef(pur, tc)[0, 1])))
    for lo, hi, name in ((0, 0.7, "impure<0.7"), (0.7, 0.95, "mid"), (0.95, 1.01, "pure>=0.95")):
        m = (pur >= lo) & (pur < hi)
        if m.sum():
            print("  %-11s n=%3d disp=%.3f to_centroid=%.3f nviews=%.0f"
                  % (name, m.sum(), dd[m].mean(), tc[m].mean(), nv[m].mean()))


if __name__ == "__main__":
    main()