"""Grounded-SAM: use the detector box as a SAM prompt to produce the mask itself.

The previous step showed the box does NOT select among existing SAM masks (27-31%).
But Grounding DINO's box is not a selector -- it is a prompt.  The published pipeline
(Grounded-SAM, Ren et al. 2024) feeds the box back into SAM and takes the produced
mask.  That is a different observation, not a re-ranking of the old pool.

Reported: IoU(prompted mask, GT), split by whether CLIP's mask ranking already
succeeded.  If Grounded-SAM recovers the CLIP failures, the two sources are
complementary and a gated fusion is a method.
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
from lerf_segmap_attribution import iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--clip_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    ap.add_argument("--topk-boxes", type=int, default=3)
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](
        checkpoint="/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth").to("cuda")
    pred = SamPredictor(sam)

    B = np.load(args.boxes)
    clip = {(r["scene"], r["view"], r["name"]): int(r["rank1_ViT-B-16"])
            for r in csv.DictReader(open(args.clip_csv))}

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
            pred.set_image(img)
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
                best_iou, best_score, n_used = 0.0, 0.0, 0
                sel_iou, sel_score = 0.0, -1.0
                for (x0, y0, x1, y1) in boxes[:args.topk_boxes]:
                    box = np.array([[x0, y0, x1, y1]], dtype=np.float32)
                    masks, scores, _ = pred.predict(box=box, multimask_output=True)
                    for k in range(masks.shape[0]):
                        v_iou = iou(masks[k].astype(bool), gt)
                        if v_iou > best_iou:
                            best_iou, best_score = float(v_iou), float(scores[k])
                        # deployable rule: trust SAM's own predicted IoU, never GT
                        if float(scores[k]) > sel_score:
                            sel_score, sel_iou = float(scores[k]), float(v_iou)
                    n_used += 1
                rows.append(dict(scene=sc, view=v, name=nm,
                                 clip_rank1=clip.get((sc, v, nm)),
                                 gsam_iou=round(best_iou, 4),
                                 gsam_iou_samsel=round(sel_iou, 4),
                                 sam_score=round(best_score, 4),
                                 n_boxes_used=n_used, gt_px=int(gt.sum())))
                print("[%s/%s] %-24s gsam_iou=%.3f (clip_rank1=%s)"
                      % (sc, v, nm, best_iou, rows[-1]["clip_rank1"]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "grounded_sam.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel, k, fn=np.mean):
        r = [x for x in rows if sel(x)]
        return None if not r else round(float(fn([x[k] for x in r])), 3)
    out = dict(n=len(rows))
    for tag, sel in (("all", lambda x: True),
                     ("clip_bad", lambda x: x["clip_rank1"] == 0),
                     ("clip_ok", lambda x: x["clip_rank1"] == 1)):
        out[tag + "_med_gsam"] = agg(sel, "gsam_iou", np.median)
        out[tag + "_med_gsam_samsel"] = agg(sel, "gsam_iou_samsel", np.median)
        out[tag + "_gsam_ge50"] = round(float(np.mean(
            [x["gsam_iou"] >= 0.5 for x in rows if sel(x)])), 3) if any(sel(x) for x in rows) else None
        out[tag + "_gsam_samsel_ge50"] = round(float(np.mean(
            [x["gsam_iou_samsel"] >= 0.5 for x in rows if sel(x)])), 3) if any(sel(x) for x in rows) else None
        out[tag + "_n"] = sum(1 for x in rows if sel(x))
    with open(os.path.join(args.out, "grounded_sam.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()