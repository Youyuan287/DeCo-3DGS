"""ScanNet SINGLE-QUERY evaluation (the missing cell).

ScanNet is normally evaluated as multiclass (argmax over classes per primitive).
This runs the other axis: one class at a time, rank regions by score, and measure
the point-level average precision of the retrieved set against the GT points of
that class.  AP is threshold-free, so no test-set tuning is involved.

Arms (centre the same field, change only the score):
  raw      s = <f_g, t_c>
  minus_a  s - alpha * a_g ,  a_g = <f_g, mu_bank>   (row bias; lemma says this is
           IDENTITY on the multiclass axis but changes the ranking here)
  minus_b  s - mean_g s        (column bias; lemma says this is IDENTITY on the
           single-query axis -> AP must not move; zero control for the lemma)

The reference direction mu_bank comes from a fixed 45-word generic vocabulary,
so nothing touches the evaluation class list.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

SUBSETS19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
NYU40 = {1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa",
         7: "table", 8: "door", 9: "window", 10: "bookshelf", 11: "picture",
         12: "counter", 14: "desk", 16: "curtain", 24: "refrigerator",
         28: "shower curtain", 33: "toilet", 34: "sink", 36: "bathtub"}
BANK = ("object stuff surface material room structure furniture clutter corner space interior "
        "scene region pattern shape volume boundary arrangement composition portion section "
        "expanse form void span zone patch field ridge groove bulk plane seam face side "
        "depth extent scale spacing texture detail contour profile margin interval").split()


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


def average_precision(scores, tp, fp, npos):
    """scores: (R,), tp/fp: point counts contributed by each region, npos: |GT_c|."""
    order = np.argsort(-scores)
    tp_c = np.cumsum(tp[order])
    fp_c = np.cumsum(fp[order])
    rec = tp_c / max(npos, 1)
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float((drec * prec).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="scene0000_00,scene0347_00,scene0070_00")
    ap.add_argument("--model_root", default="/root/verify58_20261029/full10")
    ap.add_argument("--gt", default="/root/rivermind-data/repro/deco-b123/"
                                    "scannet10_full_20261029/gt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        return l2n(f.detach().cpu().numpy())

    names = [NYU40[c] for c in SUBSETS19]
    T = emb_txt(names)
    mu_bank = l2n(emb_txt(BANK).mean(0, keepdims=True))[0]

    rows = []
    for sc in args.scenes.split(","):
        mp = os.path.join(args.model_root, sc, "model")
        d = torch.load(os.path.join(mp, "sai_nag_3.0.pt"), map_location="cpu")
        F = l2n(d["nag_feat"][-1].float()).numpy().astype(np.float64)
        region = d["nag"][-1].numpy().astype(np.int64)
        n_reg = int(region.max()) + 1
        assert n_reg <= F.shape[0], (n_reg, F.shape)
        g_xyz = read_ply_xyz(os.path.join(mp, "point_cloud", "iteration_0", "point_cloud.ply"))
        gt_xyz, gt_lab = read_ply_lab(os.path.join(args.gt, sc + "_vh_clean_2.labels.ply"))
        g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]
        reg_size = np.bincount(region, minlength=n_reg).astype(np.float64)
        a_g = F @ mu_bank

        for ci, c in enumerate(SUBSETS19):
            npos = int((g_cls == c).sum())
            if npos < 200:
                continue
            # per-region contribution: how many of its gaussians are class c
            tp = np.bincount(region, weights=(g_cls == c).astype(np.float64),
                             minlength=n_reg)
            fp = reg_size - tp
            Fc = l2n(F - F.mean(0, keepdims=True))
            s_raw = F @ T[ci]
            s_cen = Fc @ T[ci]
            s_a = s_raw - args.alpha * a_g
            s_b = s_raw - s_raw.mean()
            ap_raw = average_precision(s_raw, tp, fp, npos)
            ap_a = average_precision(s_a, tp, fp, npos)
            ap_b = average_precision(s_b, tp, fp, npos)
            ap_c = average_precision(s_cen, tp, fp, npos)
            rows.append(dict(scene=sc, cls=c, name=NYU40[c],
                             gt_px=npos, ap_raw=round(ap_raw, 4),
                             ap_minus_a=round(ap_a, 4), ap_minus_b=round(ap_b, 4),
                             ap_center=round(ap_c, 4)))
            print("[%s] %-14s npos=%6d  AP raw=%.4f  -a=%.4f  -b=%.4f"
                  % (sc, NYU40[c], npos, ap_raw, ap_a, ap_b), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scannet_singlequery_ap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = dict(n=len(rows),
               ap_raw=round(float(np.mean([r["ap_raw"] for r in rows])), 4),
               ap_minus_a=round(float(np.mean([r["ap_minus_a"] for r in rows])), 4),
               ap_minus_b=round(float(np.mean([r["ap_minus_b"] for r in rows])), 4),
               ap_center=round(float(np.mean([r["ap_center"] for r in rows])), 4),
               per_scene={s: round(float(np.mean([r["ap_raw"] for r in rows if r["scene"] == s])), 4)
                          for s in args.scenes.split(",")})
    with open(os.path.join(args.out, "scannet_singlequery_ap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()