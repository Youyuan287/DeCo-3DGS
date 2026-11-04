"""Margin-Maximizing Descriptor: is selecting the crop by margin better than chance?

For every candidate mask we can build K admissible crops (jitter / scale / context).
The proposal is to pick, per (candidate, query), the crop that maximises the
relevance margin against a fixed reference bank.  Max over K trivially raises the
margin, so the decisive controls are:

  K=1          the pipeline's single crop
  margin-sel   pick the crop with the largest margin            (the method)
  random-sel   pick a crop at random                             (noise control)
  min-sel      pick the crop with the smallest margin            (anti-control)

If margin-sel ~ random-sel, the gain is max-over-noise, not information.
Reported: rank1 (best-IoU mask ranked first) and the column gap on failures,
under two scoring rules -- plain cosine s, and the margin itself.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerf_segmap_attribution import mask_nms, iou  # noqa: E402

BANK = ("object stuff surface material room structure furniture clutter corner space interior "
        "scene region pattern shape volume boundary arrangement composition portion section "
        "expanse form void span zone patch field ridge groove bulk plane seam face side "
        "depth extent scale spacing texture detail contour profile margin interval").split()


def crops(img, m):
    """K=8 admissible crops; all label-free."""
    ys, xs = np.nonzero(m)
    if len(xs) < 8:
        return []
    H, W = img.shape[:2]
    out = []
    for pad, ctx in ((0.0, False), (0.0, True), (0.25, False), (0.25, True),
                     (0.5, False), (0.5, True)):
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        dy, dx = int(h * pad), int(w * pad)
        y0 = max(0, ys.min() - dy); y1 = min(H, ys.max() + 1 + dy)
        x0 = max(0, xs.min() - dx); x1 = min(W, xs.max() + 1 + dx)
        sub = img[y0:y1, x0:x1].copy()
        if not ctx:
            sm = m[y0:y1, x0:x1]
            sub[~sm] = 127
        out.append(sub)
    # dilated / eroded mask variants on the tight box
    for r in (3, -3):
        mm = ndimage.binary_dilation(m, iterations=r) if r > 0 else \
             ndimage.binary_erosion(m, iterations=-r)
        ys2, xs2 = np.nonzero(mm)
        if len(xs2) < 8:
            out.append(out[0]); continue
        sub = img[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1].copy()
        sm = mm[ys2.min():ys2.max() + 1, xs2.min():xs2.max() + 1]
        sub[~sm] = 127
        out.append(sub)
    return [c for c in out if c.shape[0] >= 8 and c.shape[1] >= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    args = ap.parse_args()

    import open_clip
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry["vit_h"](
        checkpoint="/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth").to("cuda")
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.7,
        stability_score_thresh=0.85, crop_n_layers=1,
        crop_n_points_downscale_factor=1, min_mask_region_area=100)
    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb(arr):
        t = prep(Image.fromarray(arr)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            f = clip.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy()

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    TB = emb_txt(BANK)
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        T = emb_txt(prompts)
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            MS, F = [], []
            for m in kept:
                cs = crops(img, m)
                if not cs:
                    continue
                if len(cs) < 8:                      # keep K fixed across masks
                    cs = cs + [cs[0]] * (8 - len(cs))
                MS.append(m)
                F.append(np.stack([emb(c) for c in cs]))       # (K, d)
            if len(MS) < 6:
                continue
            F = np.stack(F)                                     # (N, K, d)
            K = F.shape[1]
            print("[%s/%s] masks=%d K=%d" % (sc, v, len(MS), K), flush=True)
            for qi, nm in enumerate(prompts):
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != img.shape[:2]:
                    g2 = np.zeros(img.shape[:2], bool)
                    hh, ww = min(gt.shape[0], img.shape[0]), min(gt.shape[1], img.shape[1])
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    continue
                ious = np.array([iou(m, gt) for m in MS])
                b = int(ious.argmax())
                s_all = np.einsum("nkd,d->nk", F, T[qi])          # (N,K)
                bank = np.einsum("nkd,bd->nkb", F, TB).max(axis=2)  # (N,K)
                marg = s_all - bank
                rng = np.random.default_rng(20261104)
                idx = {"margin": marg.argmax(1),
                       "rand": rng.integers(0, K, len(MS)),
                       "min": marg.argmin(1)}
                row = dict(scene=sc, view=v, name=nm, iou_best=round(float(ious[b]), 3))
                for rule in ("s", "m"):
                    # s: rank by plain cosine ; m: rank by margin
                    base = s_all[:, 0] if rule == "s" else marg[:, 0]
                    row["rank1_K1_" + rule] = int(np.argsort(-base)[0] == b)
                    row["gap_K1_" + rule] = round(float(base[np.argsort(-base)[0]] - base[b]), 4)
                    for k, sel in idx.items():
                        v2 = s_all[np.arange(len(MS)), sel] if rule == "s" \
                            else marg[np.arange(len(MS)), sel]
                        rk = int(np.where(np.argsort(-v2) == b)[0][0]) + 1
                        row["rank1_%s_%s" % (k, rule)] = int(rk == 1)
                        row["gap_%s_%s" % (k, rule)] = round(float(
                            v2[np.argsort(-v2)[0]] - v2[b]), 4)
                row["best_iou"] = round(float(ious[b]), 3)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "mmd.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = dict(n=len(rows))
    for rule in ("s", "m"):
        for k in ("K1", "margin", "rand", "min"):
            out["rank1_%s_%s" % (k, rule)] = round(float(np.mean(
                [r["rank1_%s_%s" % (k, rule)] for r in rows])), 3)
    hard = [r for r in rows if r["rank1_K1_s"] == 0]
    out["n_hard"] = len(hard)
    for k in ("margin", "rand", "min"):
        out["hard_rescue_%s" % k] = round(float(np.mean(
            [r["rank1_%s_s" % k] for r in hard])), 3) if hard else None
        out["hard_medgap_%s" % k] = round(float(np.median(
            [r["gap_%s_s" % k] for r in hard])), 4) if hard else None
    out["hard_medgap_K1"] = round(float(np.median(
        [r["gap_K1_s"] for r in hard])), 4) if hard else None
    with open(os.path.join(args.out, "mmd.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()