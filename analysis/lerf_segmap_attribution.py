"""At which step do part semantics collapse? Test the seg_map overwrite.

Chain (all of it is CLIP(mask pixels), so the loss is one of these):

  1 SAM raw proposals
  2 mask_nms                       <- measured: parts survive (kept IoU 0.76-0.99)
  3 CLIP per mask                  <- measured: clean crop ranks parts #1 (0.611)
  4 mask2segmap                    <- THIS SCRIPT: exclusive per-pixel id, later
                                      mask overwrites earlier one
  5 extract_gaussian_features      <- one mask id per gaussian per view
  6 proj_gaussian_features         <- sp_mask_mat @ features, ratio >= 0.3
  7 ROFA across views
  8 progressive merge

For each GT query mask, ask: after step 4, which mask id owns its pixels?
  own_frac = fraction of GT pixels still carrying the id of the best-IoU mask
  if own_frac is low, the part was overwritten by another mask at step 4 and
  every later aggregation inherits the parent's identity.

No CLIP is involved, so a negative here cannot be blamed on CLIP.
GT is used only to score; nothing is proposed or selected with GT.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
from PIL import Image

# verbatim from image_encoding_fast.py
def mask_nms(masks, iou_thr=0.8, score_thr=0.7, inner_thr=0.5):
    sc = np.array([m["stability_score"] * m["predicted_iou"] for m in masks])
    order = np.argsort(-sc)
    seg = [masks[i]["segmentation"] for i in order]
    ar = np.array([s.sum() for s in seg], float)
    n = len(seg)
    bb = []
    for s in seg:
        ys, xs = np.nonzero(s)
        bb.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, -1, -1))
    keep = np.ones(n, bool)
    inn = np.zeros(n)
    for i in range(n):
        x0i, y0i, x1i, y1i = bb[i]
        if x1i < 0:
            keep[i] = False
            continue
        for j in range(i + 1, n):
            x0j, y0j, x1j, y1j = bb[j]
            if x1j < 0 or x1j < x0i or x0j > x1i or y1j < y0i or y0j > y1i:
                continue
            inter = np.count_nonzero(seg[i] & seg[j])
            if inter == 0:
                continue
            un = ar[i] + ar[j] - inter
            if inter / max(un, 1) > iou_thr:
                keep[j] = False
            r1, r2 = inter / max(ar[i], 1), inter / max(ar[j], 1)
            if r1 < 0.5 and r2 >= 0.85:
                inn[j] = max(inn[j], 1 - r2 * r1)
            if r1 >= 0.85 and r2 < 0.5:
                inn[i] = max(inn[i], 1 - r2 * r1)
    keep &= sc > score_thr
    keep &= inn <= 1 - inner_thr
    if keep.sum() == 0:
        keep[:3] = True
    kept = [seg[k] for k in range(n) if keep[k]]
    return kept


def iou(a, b):
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    return inter / (int(a.sum()) + int(b.sum()) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    args = ap.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"](
        checkpoint="/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth").to("cuda")
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=0.7,
        stability_score_thresh=0.85, crop_n_layers=1,
        crop_n_points_downscale_factor=1, min_mask_region_area=100)

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        for test in st[sc]["test"]:
            v = test if isinstance(test, str) else str(test["dir"])
            v = v.replace("test_", "")
            img_name = "test_%s.jpg" % v
            ip = os.path.join(sdir, "images", img_name)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            H, W = img.shape[:2]
            t0 = time.time()
            raw = list(gen.generate(img)[0])
            kept = mask_nms(raw)
            # step 4 verbatim: exclusive per-pixel id, later mask overwrites
            seg_map = -np.ones((H, W), np.int64)
            for i, m in enumerate(kept):
                seg_map[m] = i
            print("[%s/%s] raw=%d kept=%d  (%.1fs)"
                  % (sc, v, len(raw), len(kept), time.time() - t0), flush=True)
            for nm in prompts:
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
                # where do the GT pixels end up after the overwrite?
                ids = seg_map[gt]
                valid = ids >= 0
                if valid.sum() == 0:
                    continue
                own = float((ids[valid] == b).mean())
                hist = np.bincount(ids[valid], minlength=len(kept))
                dom = int(hist.argmax())
                rows.append(dict(scene=sc, view=v, name=nm, gt_px=int(gt.sum()),
                                 n_kept=len(kept),
                                 best_iou=round(float(ious[b]), 4),
                                 best_id=b, dominant_id=dom,
                                 dom_iou=round(float(ious[dom]), 4),
                                 own_frac=round(own, 4),
                                 gt_covered=round(float(valid.mean()), 4),
                                 best_score_rank=b,
                                 dom_key=int(dom == b)))
                print("   %-24s gt=%6d best_iou=%.3f(id%d) dominant=id%d(iou %.3f) "
                      "own=%.2f"
                      % (nm, gt.sum(), ious[b], b, dom, ious[dom], own), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_segmap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return {}
        return dict(n=len(r),
                    med_best_iou=round(float(np.median([x["best_iou"] for x in r])), 3),
                    med_own_frac=round(float(np.median([x["own_frac"] for x in r])), 3),
                    dominant_is_best=round(float(np.mean([x["dom_key"] for x in r])), 3),
                    med_dom_iou=round(float(np.median([x["dom_iou"] for x in r])), 3))
    summary = dict(all=agg(lambda x: True))
    with open(os.path.join(args.out, "lerf_segmap.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()