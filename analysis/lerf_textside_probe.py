"""Is Delta's misdirection the image side or the text side?

Holding Delta = f_{correct} - f_{picked} FIXED (same crops, same embeddings), vary
only the text side:

  A  uniform template applied to every name  -> does Delta's argmax move toward the
     correct name?  If a template flips it, the defect is partly textual.
  B  best-of-K paraphrases for the CORRECT name only -> an unfair upper bound on
     what any text-side change could buy.  If even that does not flip the sign,
     the defect is on the image side and no rephrasing can recover it.

Both use the same 22 failing samples as the localisation run.
GT picks masks; CLIP never sees GT.
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

TEMPLATES = ["{}", "a photo of {}", "a {}", "this is a {}", "a close-up photo of {}"]
PARAPHRASE_SUFFIX = ["{}", "a photo of {}", "a {} close up", "the {} in the scene",
                     "an object: {}"]


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
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        banks = {t: emb_txt([t.format(p) for p in prompts]) for t in TEMPLATES}
        para = [emb_txt([t.format(p) for p in prompts]) for t in PARAPHRASE_SUFFIX]
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            F = [emb_mask(img, m) for m in kept]
            idx = [i for i, f in enumerate(F) if f is not None]
            if len(idx) < 6:
                continue
            F = np.stack([F[i] for i in idx])
            MS = [kept[i] for i in idx]
            print("[%s/%s] masks=%d" % (sc, v, len(idx)), flush=True)

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
                T0 = banks[TEMPLATES[0]]
                m = int((F @ T0[qi]).argmax())
                if m == b:
                    continue
                d = F[b] - F[m]
                row = dict(scene=sc, view=v, name=nm)
                for t in TEMPLATES:
                    a = banks[t] @ d
                    row["arg_" + t] = int(int(np.argmax(a)) == qi)
                    row["gap_" + t] = round(float(a[qi] - np.max(np.delete(a, qi))), 4)
                # unfair upper bound: best paraphrase for the correct name only
                best = -9.0
                for P in para:
                    best = max(best, float(P[qi] @ d))
                base_other = float(np.max(np.delete(banks[TEMPLATES[0]] @ d, qi)))
                row["upper_bound_gap"] = round(best - base_other, 4)
                row["upper_bound_wins"] = int(best > base_other)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_textside.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = dict(n=len(rows))
    for t in TEMPLATES:
        out["argright_" + t] = round(float(np.mean([r["arg_" + t] for r in rows])), 3)
        out["medgap_" + t] = round(float(np.median([r["gap_" + t] for r in rows])), 4)
    out["upper_bound_wins"] = round(float(np.mean([r["upper_bound_wins"] for r in rows])), 3)
    out["med_upper_bound_gap"] = round(float(np.median([r["upper_bound_gap"] for r in rows])), 4)
    with open(os.path.join(args.out, "lerf_textside.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()