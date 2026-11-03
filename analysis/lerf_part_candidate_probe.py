"""LERF part candidates: does mask_nms throw away the mask that covers the part?

image_encoding_fast.py generates masks with SamAutomaticMaskGenerator and then
runs mask_nms(iou_thr=0.8, score_thr=0.7, inner_thr=0.5) before anything else.
On ScanNet that keeps only 31-40% of the raw proposals.  If a part query's only
good mask is in the discarded set, then the part is unreachable for a reason that
has nothing to do with CLIP, aggregation or hierarchy -- the observation is
deliberately deleted at encoding time.

For every test view and every scene prompt:
  GT mask  vs  max IoU over RAW masks  and  max IoU over KEPT masks
Report the cases where kept < 0.5 but raw >= 0.5.

SAM settings are the pipeline's own (points_per_side=32, crop_n_layers=1,
pred_iou_thresh=0.7, stability_score_thresh=0.85, min_mask_region_area=100).
GT is used only to score masks; nothing here proposes or selects with GT.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

PART_LIKE = {"porcelain hand", "chopsticks", "egg", "pork belly",
             "glass of water", "wavy noodles in bowl", "onion segments",
             "corn", "door handle", "bear nose", "hooves", "coffee mug"}


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
                continue                       # boxes do not intersect: no overlap
            inter = np.count_nonzero(seg[i] & seg[j])
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
    return [seg[k] for k in range(n) if keep[k]], [seg[k] for k in range(n) if not keep[k]]


def max_iou(seg_list, gt):
    """IoU of each mask with gt, sampling only gt's pixels (gt is sparse)."""
    ng = int(gt.sum())
    if ng == 0 or not seg_list:
        return 0.0
    gy, gx = np.nonzero(gt)
    best = 0.0
    for s in seg_list:
        inter = np.count_nonzero(s[gy, gx])
        if inter == 0:
            continue
        iou = inter / (int(s.sum()) + ng - inter)
        if iou > best:
            best = iou
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    ap.add_argument("--pred-iou", type=float, default=0.7)
    ap.add_argument("--stability", type=float, default=0.85)
    ap.add_argument("--min-area", type=int, default=100)
    args = ap.parse_args()

    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    dev = "cuda"
    sam = sam_model_registry["vit_h"](
        checkpoint="/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth").to(dev)
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=32, pred_iou_thresh=args.pred_iou,
        stability_score_thresh=args.stability, crop_n_layers=1,
        crop_n_points_downscale_factor=1, min_mask_region_area=args.min_area)

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        for test in st[sc]["test"]:
            if isinstance(test, dict):
                v, image = str(test["dir"]), test["image"]
            else:                              # structure.json lists "test_0"
                v, image = str(test).replace("test_", ""), str(test) + ".jpg"
            md = os.path.join(sdir, "test_mask", v)
            if not os.path.isdir(md):
                continue
            ip = os.path.join(sdir, "images", image)
            if not os.path.exists(ip):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            t0 = __import__("time").time()
            raw = list(gen.generate(img)[0])
            kept, dropped = mask_nms(raw)
            raw_segs = [m["segmentation"] for m in raw]
            print("[%s/%s] raw=%d kept=%d dropped=%d  (%.1fs)"
                  % (sc, v, len(raw), len(kept), len(dropped),
                     __import__("time").time() - t0), flush=True)
            for nm in prompts:
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != img.shape[:2]:
                    hh = min(gt.shape[0], img.shape[0])
                    ww = min(gt.shape[1], img.shape[1])
                    g2 = np.zeros(img.shape[:2], bool)
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    continue
                ik = max_iou(kept, gt)
                ir = max_iou(raw_segs, gt)
                idd = max_iou(dropped, gt)
                rows.append(dict(scene=sc, view=v, name=nm,
                                 gt_px=int(gt.sum()), n_raw=len(raw),
                                 n_kept=len(kept),
                                 iou_kept=round(ik, 4), iou_raw=round(ir, 4),
                                 iou_dropped=round(idd, 4),
                                 part_like=int(nm in PART_LIKE)))
                print("   %-24s gt=%6d  kept=%.3f raw=%.3f dropped=%.3f"
                      % (nm, gt.sum(), ik, ir, idd), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_cand_nms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return {}
        rescued = [x for x in r if x["iou_kept"] < 0.5 <= x["iou_raw"]]
        return dict(n=len(r),
                    mean_iou_kept=round(float(np.mean([x["iou_kept"] for x in r])), 4),
                    mean_iou_raw=round(float(np.mean([x["iou_raw"] for x in r])), 4),
                    kept_ge50=round(float(np.mean([x["iou_kept"] >= 0.5 for x in r])), 3),
                    raw_ge50=round(float(np.mean([x["iou_raw"] >= 0.5 for x in r])), 3),
                    in_raw_only=len(rescued),
                    in_raw_only_names=sorted({x["name"] for x in rescued}))

    summary = dict(
        part=agg(lambda x: x["part_like"]),
        other=agg(lambda x: not x["part_like"]),
        all=agg(lambda x: True))
    with open(os.path.join(args.out, "lerf_cand_nms.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()