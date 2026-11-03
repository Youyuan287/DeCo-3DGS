#!/usr/bin/env python3
"""Rebuild the per-scene cache that deco/scannet/score_scannet.py expects.

score_scannet reads only `gt` (per-Gaussian NYU40 id) and `keep` from the cache; `lab`/`feats`
are carried for compatibility. The shipped caches were built against a reconstruction with
1,015,937 Gaussians, while the artefacts on disk now have 1,012,438, so they cannot be mixed.
"""
import argparse
import os

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def xyz_of(path):
    v = PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    args = ap.parse_args()

    g_xyz, v = xyz_of(os.path.join(args.model, "point_cloud",
                                   "iteration_%d" % args.iteration, "point_cloud.ply"))
    t_xyz, tv = xyz_of(args.gt)
    ninfo = [pr.name for pr in tv.properties]
    if "label" not in ninfo:
        raise SystemExit("no label in %s" % args.gt)
    t_lab = np.asarray(tv["label"]).astype(np.int64)
    idx = cKDTree(t_xyz).query(g_xyz, workers=8)[1]
    gt = t_lab[idx].astype(np.int16)
    keep = np.ones(g_xyz.shape[0], dtype=bool)

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, args.scene + ".npz")
    np.savez_compressed(p, gt=gt, keep=keep, lab=np.zeros(g_xyz.shape[0], np.int32))
    print("%s N=%d gt>0=%.3f -> %s" % (args.scene, g_xyz.shape[0],
                                       float((gt > 0).mean()), p))


if __name__ == "__main__":
    main()