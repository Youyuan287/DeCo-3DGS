"""Is the failure explained by the crop itself having low discriminability?

Earlier measurement: even the correct mask exists (IoU 0.85) and CLIP's pick is
a zero-overlap mask that wins by only 0.032 cosine.  The remaining hypothesis is
that the correct crop simply does not look distinctive -- it is equally similar
to every query name, so no scalar scorer could break the tie.

Two independent measures of that, per query, computed on the CORRECT (best-IoU)
mask only:

  disc      cos(f_best, t_query) - max over the scene's OTHER query names
            (CLIP-side: is the crop specific to this name at all?)
  contrast  std of luminance inside the mask, mean gradient magnitude,
            mask area fraction, bbox aspect
            (image-side: is there anything to see?)

then: does either predict whether CLIP ranked that mask first?

GT picks the reference mask; CLIP never sees GT.  No 3D lifting involved.
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


def auc(score, label):
    from scipy.stats import rankdata
    s = rankdata(score)
    pos = np.asarray(label, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    args = ap.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import open_clip

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

    def emb_img(arr):
        t = prep(Image.fromarray(arr)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            f = clip.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True))[0]

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        return f / f.norm(dim=-1, keepdim=True)

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
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
            H, W = img.shape[:2]
            gray = np.asarray(Image.open(ip).convert("L")).astype(np.float32)
            gx = ndimage.sobel(gray, axis=1)
            gy = ndimage.sobel(gray, axis=0)
            gmag = np.hypot(gx, gy)
            kept = mask_nms(list(gen.generate(img)[0]))
            feats = []
            for m in kept:
                ys, xs = np.nonzero(m)
                if len(xs) < 8:
                    feats.append(None)
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = img[y0:y1, x0:x1].copy()
                sm = m[y0:y1, x0:x1]
                sub[~sm] = 127
                feats.append(emb_img(sub))
            print("[%s/%s] kept=%d" % (sc, v, len(kept)), flush=True)

            for qi, nm in enumerate(prompts):
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != (H, W):
                    g2 = np.zeros((H, W), bool)
                    hh, ww = min(gt.shape[0], H), min(gt.shape[1], W)
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    continue
                ious = np.array([iou(m, gt) for m in kept])
                b = int(ious.argmax())
                if feats[b] is None:
                    continue
                cos_all = np.array([float(T[q] @ feats[b]) for q in range(len(prompts))])
                others = np.delete(cos_all, qi)
                disc = float(cos_all[qi] - others.max())
                # does CLIP pick this mask for this query among candidates?
                cand = np.array([float(T[qi] @ f) if f is not None else -9 for f in feats])
                rank_best = int(np.where(np.argsort(-cand) == b)[0][0]) + 1
                mstat = gt if gt.sum() else np.zeros_like(gt)
                lv = gray[gt]
                ys, xs = np.nonzero(gt)
                bw = xs.max() - xs.min() + 1
                bh = ys.max() - ys.min() + 1
                rows.append(dict(scene=sc, view=v, name=nm, gt_px=int(gt.sum()),
                                 best_iou=round(float(ious[b]), 4),
                                 rank_best=rank_best,
                                 hit=int(rank_best == 1),
                                 disc=round(disc, 4),
                                 std=round(float(lv.std()), 2),
                                 grad=round(float(gmag[gt].mean()), 2),
                                 area_frac=round(float(gt.mean()), 4),
                                 aspect=round(float(max(bw, bh) / max(1, min(bw, bh))), 2),
                                 mean_l=round(float(lv.mean()), 1)))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_discriminability.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    hit = [x for x in rows if x["hit"]]
    miss = [x for x in rows if not x["hit"]]
    out = dict(n=len(rows), n_hit=len(hit), n_miss=len(miss))
    for k in ("disc", "std", "grad", "area_frac", "aspect", "best_iou"):
        out["med_%s_hit" % k] = round(float(np.median([x[k] for x in hit])), 4) if hit else None
        out["med_%s_miss" % k] = round(float(np.median([x[k] for x in miss])), 4) if miss else None
        # AUC of the feature for predicting a MISS
        out["auc_%s_to_miss" % k] = round(auc([x[k] for x in rows],
                                              [0 if x["hit"] else 1 for x in rows]), 3)
    with open(os.path.join(args.out, "lerf_discriminability.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()