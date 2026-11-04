"""Does keeping the context fix Delta?

The pipeline's crop operator zeroes everything outside the mask (pixels -> 127).
That deletes the context, and for objects whose identity lives in the context
(a glass of water vs a bowl) it may be exactly what makes Delta point at the
wrong name.

Same masks, same CLIP, three crop operators:
  masked   bbox of mask, outside -> 127        (the pipeline's own operator)
  bbox     bbox of mask, context kept          (no masking at all)
  pad50    bbox expanded 50% each side, context kept

Reported per operator:
  rank1        is the best-IoU mask ranked first among candidates
  (on the failures defined under `masked`) does Delta's argmax point at the
  correct name, and what is the gap
  |Delta| and cos(Delta_masked, Delta_op) -- if the context collapses Delta to
  zero the comparison is meaningless and that is itself the answer.

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

OPS = ["masked", "bbox", "pad50"]


def crop(img, m, op):
    ys, xs = np.nonzero(m)
    if len(xs) < 8:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if op == "pad50":
        h, w = y1 - y0, x1 - x0
        dy, dx = int(h * 0.5), int(w * 0.5)
        H, W = img.shape[:2]
        y0, y1 = max(0, y0 - dy), min(H, y1 + dy)
        x0, x1 = max(0, x0 - dx), min(W, x1 + dx)
    sub = img[y0:y1, x0:x1].copy()
    if op == "masked":
        sm = m[y0:y1, x0:x1]
        sub[~sm] = 127
    if sub.shape[0] < 8 or sub.shape[1] < 8:
        return None
    return sub


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
    rows = []
    acc = {}
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
            MS, F = [], {op: [] for op in OPS}
            for m in kept:
                done = True
                for op in OPS:
                    sub = crop(img, m, op)
                    if sub is None:
                        done = False
                        break
                    F[op].append(emb(sub))
                if done:
                    MS.append(m)
            if len(MS) < 6:
                continue
            for op in OPS:
                F[op] = np.stack(F[op])
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
                for op in OPS:
                    cos = F[op] @ T[qi]
                    order = np.argsort(-cos)
                    rk = int(np.where(order == b)[0][0]) + 1
                    row["rank1_" + op] = int(rk == 1)
                    row["rank_" + op] = rk
                    acc.setdefault(op, []).append(rk == 1)
                # failures under the pipeline operator only
                m = int((F["masked"] @ T[qi]).argmax())
                if m != b:
                    for op in OPS:
                        d = F[op][b] - F[op][m]
                        a = T @ d
                        row["darg_" + op] = int(int(np.argmax(a)) == qi)
                        row["dgap_" + op] = round(float(a[qi] - np.max(np.delete(a, qi))), 4)
                        row["dabs_" + op] = round(float(np.linalg.norm(d)), 4)
                    dm = F["masked"][b] - F["masked"][m]
                    for op in ("bbox", "pad50"):
                        d2 = F[op][b] - F[op][m]
                        row["dcos_" + op] = round(float(
                            dm @ d2 / (np.linalg.norm(dm) * np.linalg.norm(d2) + 1e-9)), 4)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_context.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fails = [r for r in rows if "darg_masked" in r and r["darg_masked"] == 0]
    out = dict(n=len(rows), n_failures=len(fails))
    for op in OPS:
        out["rank1_" + op] = round(float(np.mean([r["rank1_" + op] for r in rows])), 3)
    if fails:
        for op in OPS:
            out["fail_dargright_" + op] = round(float(np.mean([r["darg_" + op] for r in fails])), 3)
            out["fail_medgap_" + op] = round(float(np.median([r["dgap_" + op] for r in fails])), 4)
            out["fail_medabs_" + op] = round(float(np.median([r["dabs_" + op] for r in fails])), 4)
        for op in ("bbox", "pad50"):
            v = [r["dcos_" + op] for r in fails if "dcos_" + op in r]
            out["fail_med_deltacos_vs_masked_" + op] = round(float(np.median(v)), 3) if v else None
    with open(os.path.join(args.out, "lerf_context.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()