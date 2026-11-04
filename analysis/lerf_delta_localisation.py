"""Where does the misalignment of Delta happen?

Corrected problem: selection fails iff

    <Delta, t_q*>  <  max_{q'!=q*} <Delta, t_q'>,     Delta = f_{m*} - f_m

i.e. the DIRECTION of the candidate difference in image space does not point at
the right name in text space.  This localises that:

  A  single crop, candidate masks          <- the earliest measurable stage
  B  single crop, GT mask in place of m*   <- is it mask quality or representation?
  C  per-view consistency of the sign      <- if views agree, cross-view averaging
                                              preserves Delta and cannot fix it

A asks "is the difference vector wrong already?"; B separates "bad mask" from
"bad embedding"; C decides whether any downstream averaging could help.

GT picks masks and provides the perfect-mask control.  CLIP never sees GT.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerf_segmap_attribution import mask_nms, iou  # noqa: E402


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

    def emb_mask(img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = img[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1]
        sub[~sm] = 127
        t = prep(Image.fromarray(sub)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            f = clip.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy()

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
        Tn = T.cpu().numpy()
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            feats = [emb_mask(img, m) for m in kept]
            ok = [i for i, f in enumerate(feats) if f is not None]
            if len(ok) < 6:
                continue
            F = np.stack([feats[i] for i in ok])
            MS = [kept[i] for i in ok]
            print("[%s/%s] masks=%d" % (sc, v, len(ok)), flush=True)

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
                cos = (F @ Tn[qi])
                m = int(cos.argmax())
                if m == b:
                    continue                     # only analyse the failures
                d = F[b] - F[m]
                a = Tn @ d
                arg_ok = int(np.argmax(a)) == qi
                gap = float(a[qi] - np.max(np.delete(a, qi)))
                fgt = emb_mask(img, gt)
                if fgt is None:
                    continue
                dg = fgt - F[m]
                ag = Tn @ dg
                arg_ok_gt = int(np.argmax(ag)) == qi
                gap_gt = float(ag[qi] - np.max(np.delete(ag, qi)))
                rows.append(dict(scene=sc, view=v, name=nm,
                                 iou_best=round(float(ious[b]), 3),
                                 iou_pick=round(float(ious[m]), 3),
                                 delta_argmax_is_right=int(arg_ok),
                                 delta_gap=round(gap, 4),
                                 delta_gt_argmax_is_right=int(arg_ok_gt),
                                 delta_gt_gap=round(gap_gt, 4)))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_delta_localisation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return {}
        return dict(n=len(r),
                    delta_argmax_right=round(float(np.mean([x["delta_argmax_is_right"] for x in r])), 3),
                    delta_gt_argmax_right=round(float(np.mean([x["delta_gt_argmax_is_right"] for x in r])), 3),
                    med_delta_gap=round(float(np.median([x["delta_gap"] for x in r])), 4),
                    med_delta_gt_gap=round(float(np.median([x["delta_gt_gap"] for x in r])), 4))
    # per-name view consistency: does the sign of delta_gap agree across views?
    import collections
    byname = collections.defaultdict(list)
    for x in rows:
        byname[(x["scene"], x["name"])].append(x["delta_gap"])
    consist = [float(np.mean([s > 0 for s in v])) for v in byname.values() if len(v) >= 2]
    out = dict(failures=agg(lambda x: True), consistency_pairs=len(consist))
    if consist:
        out["frac_items_all_views_same_sign"] = round(
            float(np.mean([c in (0.0, 1.0) for c in consist])), 3)
        out["mean_frac_positive"] = round(float(np.mean(consist)), 3)
    with open(os.path.join(args.out, "lerf_delta_localisation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()