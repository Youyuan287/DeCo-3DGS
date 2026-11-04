"""Replace global pooling with masked patch-token pooling.

Every candidate observation is CLIP's GLOBAL pooled embedding of the crop: all
spatial structure inside the mask is collapsed before anything else sees it.  The
user's hypothesis is that this is where the decision information is lost, and that
preserving more of it (a different pooling operator, not a reweighting) could
recover hard samples.

Same encoder, same crops, same text.  Only the pooling inside the mask changes:

  global       the pipeline's operator (CLS token)
  masked_mean  mean over patch tokens weighted by mask coverage of each patch
  masked_max   elementwise max over patches with coverage >= 0.5

A per-crop self-check asserts that the captured CLS token reproduces
model.encode_image, so "patch pooling" cannot silently mean something else.

Reports rank1 overall, and on the well-masked failures (the Type-A set: correct
mask has IoU >= 0.7 but the baseline ranks it below 1) whether Delta's direction
recovers.  GT picks masks only.
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

PATCH = 16


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

    tokbuf = {}
    h = clip.visual.ln_post.register_forward_hook(
        lambda mod, i, o: tokbuf.__setitem__("t", o))

    def feats(img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = img[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1].astype(np.uint8) * 255
        sub[sm == 0] = 127
        pil = Image.fromarray(sub)
        x = prep(pil).unsqueeze(0).to("cuda")
        with torch.no_grad():
            g = clip.encode_image(x)[0]
        t = tokbuf["t"][0]                        # (1+N, D) after ln_post
        proj = clip.visual.proj
        patch = t[1:] @ proj                      # (N, d)
        # self-check: CLS path must reproduce encode_image
        cls = (t[0] @ proj)
        cls = cls / cls.norm()
        if float((cls * (g / g.norm())).sum()) < 0.99:
            raise RuntimeError("patch/CLS extraction mismatch")
        d = patch.shape[1]
        grid = int(round(patch.shape[0] ** 0.5))
        # mask -> patch grid coverage
        ms = np.asarray(Image.fromarray(sm).resize((grid * PATCH, grid * PATCH),
                                                   Image.NEAREST)) > 127
        cov = ms.reshape(grid, PATCH, grid, PATCH).mean((1, 3)).ravel()
        w = torch.tensor(cov, dtype=patch.dtype, device=patch.device)
        out = {}
        out["global"] = g
        wsum = w.sum()
        out["masked_mean"] = (patch * w[:, None]).sum(0) / wsum.clamp(min=1e-6)
        sel = w >= 0.5
        out["masked_max"] = patch[sel].max(0)[0] if int(sel.sum()) > 0 else patch.max(0)[0]
        return {k: (v.detach() / v.detach().norm()) for k, v in out.items()}

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    VAR = ["global", "masked_mean", "masked_max"]
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
            MS, F = [], {k: [] for k in VAR}
            for m in kept:
                fs = feats(img, m)
                if fs is None:
                    continue
                MS.append(m)
                for k in VAR:
                    F[k].append(fs[k].cpu().numpy())
            if len(MS) < 6:
                continue
            for k in VAR:
                F[k] = np.stack(F[k])
            print("[%s/%s] masks=%d" % (sc, v, len(MS)), flush=True)

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
                row = dict(scene=sc, view=v, name=nm, iou_best=round(float(ious[b]), 3))
                picks = {}
                for k in VAR:
                    cos = F[k] @ T[qi]
                    rk = int(np.where(np.argsort(-cos) == b)[0][0]) + 1
                    row["rank1_" + k] = int(rk == 1)
                    row["rank_" + k] = rk
                    picks[k] = int(cos.argmax())
                # Type-A failures: good mask, baseline mis-ranks it
                if row["rank1_global"] == 0 and ious[b] >= 0.7:
                    for k in VAR:
                        d = F[k][b] - F[k][picks["global"]]
                        a = T @ d
                        row["darg_" + k] = int(int(np.argmax(a)) == qi)
                        row["dgap_" + k] = round(float(a[qi] - np.max(np.delete(a, qi))), 4)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pooling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    typeA = [r for r in rows if "darg_global" in r]
    out = dict(n=len(rows), n_typeA=len(typeA))
    for k in VAR:
        out["rank1_" + k] = round(float(np.mean([r["rank1_" + k] for r in rows])), 3)
        out["meanrank_" + k] = round(float(np.mean([r["rank_" + k] for r in rows])), 2)
        if typeA:
            out["typeA_dargright_" + k] = round(float(np.mean([r["darg_" + k] for r in typeA])), 3)
            out["typeA_medgap_" + k] = round(float(np.median([r["dgap_" + k] for r in typeA])), 4)
    with open(os.path.join(args.out, "pooling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()