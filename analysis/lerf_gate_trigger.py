"""Can a label-free trigger predict when switching from CLIP to Grounded-SAM pays?

The gate's target is not "CLIP is bad" but "switching would help", i.e.
gsam_iou > clip_iou.  For every (scene, frame, category) we recompute the pipeline's
score vector over its own mask pool and derive trigger statistics that use no labels:

  top1        max score
  gap         top1 - top2
  sd          std of scores over the pool
  snr         top1 / sd
  entropy     entropy of softmax(scores)
  n_masks     pool size
  area        area fraction of the top-1 mask

Reported: AUC of each statistic for (a) clip_iou < 0.5 and (b) gsam > clip + 0.05,
plus the oracle ceiling (per-query best of the two arms).
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

LABEL = "/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label"
POOL = "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs"


def gt_masks(h, w, anno):
    out = {}
    for o in anno["objects"]:
        c = o["category"]
        im = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(im)
        seg = o["segmentation"]
        polys = seg if (seg and isinstance(seg[0], (list, tuple))
                        and seg[0] and isinstance(seg[0][0], (list, tuple))) else [seg]
        for poly in polys:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(arr) >= 3:
                dr.polygon([tuple(p) for p in arr], fill=1)
        m = np.asarray(im) > 0
        out[c] = np.logical_or(out[c], m) if c in out else m
    return out


def pool_masks(scene, frame):
    p = os.path.join(POOL, scene, "language_features", frame)
    s = np.load(p + "_s.npy")
    ids = sorted({int(v) for v in np.unique(s) if v >= 0})
    ms = []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append(s[lv] == mid)
    return ms


def auc(x, y):
    from scipy.stats import rankdata
    s = rankdata(x)
    pos = np.asarray(y, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="lerfovs_gsam.csv from the previous run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    args = ap.parse_args()

    import open_clip
    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    def emb_mask(img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return np.zeros(512, np.float32)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.asarray(img)[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1]
        sub[~sm] = 127
        t = prep(Image.fromarray(sub)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            e = clip.encode_image(t)
        return ((e / e.norm(dim=-1, keepdim=True))[0]).cpu().numpy().astype(np.float32)

    prev = {(r["scene"], r["frame"], r["name"]): r for r in csv.DictReader(open(args.csv))}
    rows = []
    for sc in args.scenes.split(","):
        gdir = os.path.join(LABEL, sc)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(f for f in os.listdir(gdir) if f.endswith(".json")):
            fr = fn[:-5]
            anno = json.load(open(os.path.join(gdir, fn)))
            img = Image.open(os.path.join(gdir, fr + ".jpg")).convert("RGB")
            W, H = img.size
            gts = gt_masks(H, W, anno)
            ms = pool_masks(sc, fr)
            if not ms:
                continue
            F = np.stack([emb_mask(img, m) for m in ms])
            T = emb_txt(list(gts.keys()))
            print("[%s/%s] pool=%d" % (sc, fr, len(ms)), flush=True)
            for ci, cat in enumerate(gts):
                k = prev.get((sc, fr, cat))
                if k is None:
                    continue
                s = F @ T[ci]
                o = np.argsort(-s)
                top1, top2 = float(s[o[0]]), float(s[o[1]]) if len(o) > 1 else float(s[o[0]])
                sd = float(s.std())
                p = np.exp((s - s.max()) / 0.02)
                p = p / p.sum()
                ent = float(-(p * np.log(p + 1e-12)).sum())
                rows.append(dict(scene=sc, frame=fr, name=cat,
                                 clip_iou=float(k["clip_iou"]), gsam_iou=float(k["gsam_iou"]),
                                 top1=round(top1, 4), gap=round(top1 - top2, 4),
                                 sd=round(sd, 4), snr=round(top1 / max(sd, 1e-6), 3),
                                 entropy=round(ent, 3), n_masks=len(ms),
                                 area=round(float(ms[o[0]].mean()), 4)))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gate_trigger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    c = np.array([r["clip_iou"] for r in rows])
    g = np.array([r["gsam_iou"] for r in rows])
    extra = dict(n=len(rows),
                 clip_med=round(float(np.median(c)), 3),
                 gsam_med=round(float(np.median(g)), 3),
                 oracle_med=round(float(np.median(np.maximum(c, g))), 3),
                 oracle_ge50=round(float((np.maximum(c, g) >= 0.5).mean()), 3),
                 clip_ge50=round(float((c >= 0.5).mean()), 3),
                 n_switch_helps=int((g > c + 0.05).sum()))
    for stat in ("top1", "gap", "sd", "snr", "entropy", "n_masks", "area"):
        x = [r[stat] for r in rows]
        extra["auc_%s_clipbad" % stat] = round(auc(x, c < 0.5), 3)
        extra["auc_%s_switch" % stat] = round(auc(x, (g > c + 0.05).astype(int)), 3)
    with open(os.path.join(args.out, "gate_trigger.json"), "w") as f:
        json.dump(extra, f, indent=2)
    print(json.dumps(extra, indent=2), flush=True)


if __name__ == "__main__":
    main()