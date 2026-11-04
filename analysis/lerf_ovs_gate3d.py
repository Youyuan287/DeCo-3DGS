"""Which 3D-side statistic predicts where Grounded-SAM would help?

The 2D entropy gate failed because 2D mask difficulty and 3D render difficulty are
decoupled.  This computes statistics from the FIELD itself (cluster-level scores,
no rendering, no labels) and asks which one predicts "gsam3d > clip3d".

Statistics per query (all label-free):
  top1    max cluster score at the root level
  gap     top1 - top2 among root clusters
  sd      std of the root score vector
  snr     top1 / sd
  ent     entropy of softmax(root scores)
  ztop    (top1 - mean) / sd  (how far the winner stands out in this scene)

Joined with the end-to-end CSVs to get, for each query, clip3d and gsam3d.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nag_root", default="/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf")
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--base", required=True, help="dir with <scene>/per_query.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    args = ap.parse_args()

    import open_clip
    clip, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        return (f / f.norm(dim=-1, keepdim=True)).float()

    e3 = {(r["scene"], r["frame"], r["name"]): r for r in csv.DictReader(open(args.e2e))}
    base = {}
    for sc in args.scenes.split(","):
        p = os.path.join(args.base, sc, "per_query.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                base[(r["scene"], r["image"], r["prompt"])] = float(r["iou"])

    rows = []
    for sc in args.scenes.split(","):
        nag = torch.load(os.path.join(args.nag_root, sc, "sai_nag.pt"), map_location="cpu")
        F = nag["nag_feat"][-1].float().cuda()
        F = F / F.norm(dim=-1, keepdim=True)
        keys = [k for k in e3 if k[0] == sc]
        cats = sorted({k[2] for k in keys})
        T = emb(cats)
        for ci, cat in enumerate(cats):
            s = (F @ T[ci]).cpu().numpy()
            o = np.argsort(-s)
            top1, top2 = float(s[o[0]]), float(s[o[1]]) if len(o) > 1 else float(s[o[0]])
            sd = float(s.std())
            p = np.exp((s - s.max()) / 0.02)
            p /= p.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            for (sc2, fr, c2) in keys:
                if c2 != cat:
                    continue
                b = base.get((sc2, fr, c2))
                g = e3[(sc2, fr, c2)]["iou3d"]
                if b is None or g in ("", "nan"):
                    continue
                rows.append(dict(scene=sc2, frame=fr, name=c2,
                                 clip3d=b, gsam3d=float(g),
                                 top1=round(top1, 4), gap=round(top1 - top2, 4),
                                 sd=round(sd, 4), snr=round(top1 / max(sd, 1e-6), 3),
                                 ent=round(ent, 3), ztop=round((top1 - s.mean()) / max(sd, 1e-6), 3)))
        print("[%s] done, rows=%d" % (sc, len(rows)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate3d.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def auc(x, y):
        from scipy.stats import rankdata
        r = rankdata(x)
        pos = np.asarray(y, bool)
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 == 0 or n0 == 0:
            return float("nan")
        return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    c = np.array([r["clip3d"] for r in rows])
    g = np.array([r["gsam3d"] for r in rows])
    sw = (g > c + 0.05)
    out = dict(n=len(rows), n_switch_would_help=int(sw.sum()),
               clip_med=round(float(np.median(c)), 4),
               gsam_med=round(float(np.median(g)), 4),
               oracle_med=round(float(np.median(np.maximum(c, g))), 4))
    for k in ("top1", "gap", "sd", "snr", "ent", "ztop"):
        x = [r[k] for r in rows]
        out["auc_" + k] = round(auc(x, sw), 3)
        out["auc_" + k + "_clipbad"] = round(auc(x, c < 0.5), 3)
    with open(os.path.join(args.out, "gate3d.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()