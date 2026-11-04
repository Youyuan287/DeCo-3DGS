"""Screen candidate scoring functionals against the criterion in the problem statement.

For a fixed frame the pipeline picks argmax_m s(m,q).  Which functionals can even
change that argmax?  Some are provable no-ops; the rest can be measured.

Given f_m and t_q (unit), s = <f_m, t_q>:

  row-centered   <f_m - mean_m f, t_q>      subtract a term constant in m  -> NO-OP
  col-centered   <f_m, t_q - mean_q t>      removes <f_m, mu_T>, i.e. the a term
  both           <f_m-mu_F, t_q-mu_T>       differs from col-centered only by an
                                            m-independent constant -> same ranking
  residual cos   unit(f_m-mu_F) . unit(t_q-mu_T)
  whitened       (f-mu_F)' (Fc'Fc/n + lam I)^-1 (t-mu_T)   -- the only family that
                 genuinely re-weights directions by how much masks differ in them

The problem statement claims the discrimination lives in the mask-residual subspace
and that plain dot product over-weights the common mode.  Whitening is the canonical
counter-operator; this measures whether it actually moves the ranking, and by how much.

Rank of the correct (best-IoU) mask is the metric.  GT is evaluation only.
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


def ranks(score, correct):
    """score: (n_mask, n_name); correct: index per name. 1-based rank of correct."""
    out = []
    for q in range(score.shape[1]):
        order = np.argsort(-score[:, q])
        out.append(int(np.where(order == correct[q])[0][0]) + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    ap.add_argument("--lam", type=float, default=0.1)
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
    acc = {}
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        T = emb_txt(prompts).cpu().numpy()
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            F, MS = [], []
            for m in kept:
                ys, xs = np.nonzero(m)
                if len(xs) < 8:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = img[y0:y1, x0:x1].copy()
                sm = m[y0:y1, x0:x1]
                sub[~sm] = 127
                F.append(emb_img(sub).cpu().numpy())
                MS.append(m)
            if len(F) < 6:
                continue
            F = np.stack(F)
            # per-name correct mask id
            correct = []
            for qi, nm in enumerate(prompts):
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    correct.append(-1)
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != img.shape[:2]:
                    g2 = np.zeros(img.shape[:2], bool)
                    hh, ww = min(gt.shape[0], img.shape[0]), min(gt.shape[1], img.shape[1])
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    correct.append(-1)
                    continue
                ious = np.array([iou(m, gt) for m in MS])
                correct.append(int(ious.argmax()))
            if any(c < 0 for c in correct):
                prompts_i = [i for i, c in enumerate(correct) if c >= 0]
            else:
                prompts_i = list(range(len(prompts)))
            Ti = T[prompts_i]
            ci = [correct[i] for i in prompts_i]
            if not prompts_i:
                continue

            muF, muT = F.mean(0, keepdims=True), Ti.mean(0, keepdims=True)
            Fc, Tc = F - muF, Ti - muT

            S = {}
            S["plain"] = F @ Ti.T
            S["row_centered"] = Fc @ Ti.T
            S["col_centered"] = F @ Tc.T
            S["both_centered"] = Fc @ Tc.T
            nF = Fc / np.maximum(np.linalg.norm(Fc, axis=1, keepdims=True), 1e-8)
            nT = Tc / np.maximum(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-8)
            S["residual_cos"] = nF @ nT.T
            U, sg, Vt = np.linalg.svd(Fc, full_matrices=False)
            r = min(len(sg), 20)
            V = Vt[:r].T
            lam = args.lam * (sg[:r] ** 2 / Fc.shape[0]).mean()
            W = 1.0 / (sg[:r] ** 2 / Fc.shape[0] + lam)
            S["whitened"] = ((Fc @ V) * W) @ (Tc @ V).T

            row = dict(scene=sc, view=v, n_mask=len(F), n_name=len(prompts_i))
            for k, M in S.items():
                rk = ranks(M, ci)
                row[k + "_rank1"] = round(float(np.mean([r == 1 for r in rk])), 3)
                row[k + "_meanrank"] = round(float(np.mean(rk)), 2)
                acc.setdefault(k, []).extend([r == 1 for r in rk])
            # does each functional pick the same mask as plain?
            for k in S:
                if k == "plain":
                    continue
                same = np.mean([int(np.argmax(S[k][:, q])) == int(np.argmax(S["plain"][:, q]))
                                for q in range(len(ci))])
                row[k + "_same_as_plain"] = round(float(same), 3)
            rows.append(row)
            print("[%s/%s] masks=%d  " % (sc, v, len(F))
                  + "  ".join("%s r1=%.2f" % (k, row[k + "_rank1"]) for k in S), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "functional_screen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
    out["n_eval"] = len(acc["plain"])
    with open(os.path.join(args.out, "functional_screen.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()