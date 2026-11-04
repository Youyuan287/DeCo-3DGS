"""For the 2/3 of queries whose best mask survives mask2segmap: does CLIP pick it?

Step 4 was already measured: 33% of queries lose their best mask to a worse one
inside mask2segmap.  For the rest the pixels carry the right mask id, so whatever
is still lost must happen at, or after, the feature stage.

This splits that remainder in two:

  CLIP ranks the best-IoU mask #1 among the kept masks
      -> the 2D observation and its score are both fine; the loss is in the 3D
         lifting (extract_gaussian_features -> sp_mask_mat -> merge) or the readout
  CLIP does not rank it #1
      -> the loss is the scoring function itself: the query text prefers some
         other mask whose IoU with the target is worse

Crop convention matches the pipeline: bounding box of the mask, pixels outside the
mask set to 127, resized to 224.  No padding.  Plain query text (same as the
clean-crop probe, so the two are comparable).
GT is used only to pick and score the reference mask; CLIP never sees GT.
"""
import argparse
import csv
import json
import os
import time

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
    ap.add_argument("--own-thr", type=float, default=0.5)
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
            raw = list(gen.generate(img)[0])
            kept = mask_nms(raw)
            # exact step-4 semantics: later mask owns the contested pixels
            seg_map = -np.ones((H, W), np.int64)
            for i, m in enumerate(kept):
                seg_map[m] = i
            print("[%s/%s] kept=%d" % (sc, v, len(kept)), flush=True)

            # crop every kept mask once; reuse across queries in this frame
            feats = []
            for m in kept:
                ys, xs = np.nonzero(m)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = img[y0:y1, x0:x1].copy()
                sm = m[y0:y1, x0:x1]
                sub[~sm] = 127
                if sub.shape[0] < 8 or sub.shape[1] < 8:
                    feats.append(None)
                    continue
                feats.append(emb_img(sub))

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
                ids = seg_map[gt]
                valid = ids >= 0
                if valid.sum() == 0:
                    continue
                own = float((ids[valid] == b).mean())
                ok = [i for i in range(len(kept)) if feats[i] is not None]
                cos = np.array([float(T[qi] @ feats[i]) if feats[i] is not None else -9
                                for i in range(len(kept))])
                order = np.argsort(-cos)
                rank_best = int(np.where(order == b)[0][0]) + 1 if b in ok else -1
                top = int(order[0])
                rows.append(dict(scene=sc, view=v, name=nm, gt_px=int(gt.sum()),
                                 best_iou=round(float(ious[b]), 4), own_frac=round(own, 4),
                                 clip_rank_best=rank_best,
                                 clip_top_iou=round(float(ious[top]), 4),
                                 clip_top_cos=round(float(cos[top]), 4),
                                 best_cos=round(float(cos[b]), 4),
                                 margin=round(float(cos[top] - cos[b]), 4)))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_mask_rank.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return {}
        return dict(n=len(r),
                    med_best_iou=round(float(np.median([x["best_iou"] for x in r])), 3),
                    rank1=round(float(np.mean([x["clip_rank_best"] == 1 for x in r])), 3),
                    med_rank=round(float(np.median([x["clip_rank_best"] for x in r])), 1),
                    clip_top_ge50=round(float(np.mean([x["clip_top_iou"] >= 0.5 for x in r])), 3),
                    med_clip_top_iou=round(float(np.median([x["clip_top_iou"] for x in r])), 3))
    summary = dict(
        kept_mask_group=agg(lambda x: x["own_frac"] >= args.own_thr),
        overwritten_group=agg(lambda x: x["own_frac"] < args.own_thr),
        all=agg(lambda x: True))
    with open(os.path.join(args.out, "lerf_mask_rank.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()