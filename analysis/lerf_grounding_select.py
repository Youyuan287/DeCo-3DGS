"""Detector-guided mask selection: can a Grounding DINO box pick the right SAM mask?

Grounding DINO gives a >=0.5-IoU box on 72.7% of the queries where CLIP ranking fails
(vs 61.5% where it succeeds), i.e. the detector is complementary.  The open question
is whether a box can *select* a mask.

Rules (all training-free, no GT in the decision):
  box_iou    pick the mask maximising IoU with any box
  box_center pick the smallest mask containing the centre of the best box
  best_any   oracle over the pool (upper bound, reference only)

Reported: IoU(picked, GT) and rank1 (picked == best-IoU mask), split by whether CLIP
ranked that query correctly.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lerf_segmap_attribution import mask_nms, iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--clip_csv", required=True)
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

    B = np.load(args.boxes)
    clip = {}
    for r in csv.DictReader(open(args.clip_csv)):
        clip[(r["scene"], r["view"], r["name"])] = int(r["rank1_ViT-B-16"])

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            H, W = img.shape[:2]
            kept = mask_nms(list(gen.generate(img)[0]))
            print("[%s/%s] masks=%d" % (sc, v, len(kept)), flush=True)
            for nm in prompts:
                p = os.path.join(md, nm + ".png")
                key = "%s|%s|%s" % (sc, v, nm)
                if not os.path.exists(p) or key not in B:
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != (H, W):
                    g2 = np.zeros((H, W), bool)
                    hh, ww = min(gt.shape[0], H), min(gt.shape[1], W)
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    continue
                boxes = B[key]
                if len(boxes) == 0:
                    continue
                ious = np.array([iou(m, gt) for m in kept])
                b = int(ious.argmax())
                bm = []
                for (x0, y0, x1, y1) in boxes:
                    x0, y0 = max(0, int(x0)), max(0, int(y0))
                    x1, y1 = min(W, int(x1)), min(H, int(y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    m = np.zeros((H, W), bool)
                    m[y0:y1, x0:x1] = True
                    bm.append(m)
                if not bm:
                    continue
                # rule 1: mask maximising IoU with any box
                sc_iou = np.array([max(iou(k, bx) for bx in bm) for k in kept])
                pick1 = int(sc_iou.argmax())
                # rule 2: smallest mask containing the centre of the first box
                x0, y0, x1, y1 = boxes[0]
                cy, cx = int((y0 + y1) / 2), int((x0 + x1) / 2)
                cand = [i for i, k in enumerate(kept) if k[min(max(cy, 0), H - 1),
                                                          min(max(cx, 0), W - 1)]]
                pick2 = min(cand, key=lambda i: kept[i].sum()) if cand else pick1
                rows.append(dict(scene=sc, view=v, name=nm,
                                 clip_rank1=clip.get((sc, v, nm)),
                                 best_iou=round(float(ious[b]), 4),
                                 box_iou=round(float(ious[pick1]), 4),
                                 boxc_iou=round(float(ious[pick2]), 4),
                                 box_is_best=int(pick1 == b),
                                 boxc_is_best=int(pick2 == b),
                                 n_boxes=int(len(boxes))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "grounding_select.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel, k):
        r = [x for x in rows if sel(x)]
        if not r:
            return None
        return round(float(np.mean([x[k] for x in r])), 3)
    out = dict(n=len(rows),
               clip_bad_n=sum(1 for x in rows if x["clip_rank1"] == 0),
               clip_ok_n=sum(1 for x in rows if x["clip_rank1"] == 1))
    for tag, sel in (("all", lambda x: True),
                     ("clip_bad", lambda x: x["clip_rank1"] == 0),
                     ("clip_ok", lambda x: x["clip_rank1"] == 1)):
        out["%s_box_is_best" % tag] = agg(sel, "box_is_best")
        out["%s_med_box_iou" % tag] = agg(sel, "box_iou")
        out["%s_med_boxc_iou" % tag] = agg(sel, "boxc_iou")
        out["%s_med_best_iou" % tag] = agg(sel, "best_iou")
    with open(os.path.join(args.out, "grounding_select.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()