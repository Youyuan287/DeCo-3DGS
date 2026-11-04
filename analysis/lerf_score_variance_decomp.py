"""Where does the discrimination actually live? Variance decomposition of S[mask, name].

The measured symptom: unrelated masks score almost the same for a query (the
correct mask wins by 0.032 cosine; on the correct crop the own name leads the
best other name by ~0).  Question: is that because the *association* term is small,
or because the framework's choice is mostly query-independent in the first place?

For each frame build the full score matrix

    S[i, q] = cos( f_mask_i , t_name_q )

and decompose it the same way the axis-bias finding did for the 3D readout:

    S[i, q] = mu + a_i + b_q + r[i, q]
      a_i  row effect  -> "which masks look object-like at all" (query-free)
      b_q  col effect  -> "which names are generically high"    (mask-free)
      r    association -> the only term that depends on BOTH

Report
  * sum-of-squares shares of a, b, r
  * agreement between argmax_i S[:,q] (what the framework picks) and
    argmax_i a_i (the query-free pick).  If these mostly agree, the selection
    is driven by generic objectness, not by the query -- which would mean the
    "similar semantics" is a symptom of the score being dominated by a_i.
  * std(a) vs std(r), i.e. how much room r has.

No GT in any of this.  Purely a property of the framework's own scores.
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
from lerf_segmap_attribution import mask_nms  # noqa: E402


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
    rows, S_all = [], []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        T = emb_txt(prompts)
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            if not os.path.exists(ip):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            embs = []
            for m in kept:
                ys, xs = np.nonzero(m)
                if len(xs) < 8:
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = img[y0:y1, x0:x1].copy()
                sm = m[y0:y1, x0:x1]
                sub[~sm] = 127
                embs.append(emb_img(sub))
            if len(embs) < 4:
                continue
            F = torch.stack(embs)                    # (n_mask, D)
            S = (F @ T.T).cpu().numpy()              # (n_mask, n_name)
            S_all.append(S)

            mu = S.mean()
            a = S.mean(1) - mu                       # row effect
            b = S.mean(0) - mu                       # col effect
            R = S - mu - a[:, None] - b[None, :]
            sst = float(((S - mu) ** 2).sum())
            ss_a = float((a[:, None] ** 2).sum() * S.shape[1])
            ss_b = float((b[None, :] ** 2).sum() * S.shape[0])
            ss_r = float((R ** 2).sum())
            qfree = int(np.argmax(a))
            agree = float(np.mean([int(np.argmax(S[:, q])) == qfree
                                   for q in range(S.shape[1])]))
            rows.append(dict(scene=sc, view=v, n_mask=int(S.shape[0]),
                             n_name=int(S.shape[1]),
                             share_a=round(ss_a / sst, 3),
                             share_b=round(ss_b / sst, 3),
                             share_r=round(ss_r / sst, 3),
                             std_a=round(float(a.std()), 4),
                             std_b=round(float(b.std()), 4),
                             std_r=round(float(R.std()), 4),
                             agree_with_queryfree=round(agree, 3)))
            print("[%s/%s] masks=%d  share a=%.3f b=%.3f r=%.3f  "
                  "std a=%.4f r=%.4f  argmax agrees with query-free: %.2f"
                  % (sc, v, S.shape[0], ss_a / sst, ss_b / sst, ss_r / sst,
                     a.std(), R.std(), agree), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerf_variance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    S = np.concatenate([x.reshape(-1) for x in S_all])
    out = dict(n_frames=len(rows),
               mean_share_a=round(float(np.mean([r["share_a"] for r in rows])), 3),
               mean_share_b=round(float(np.mean([r["share_b"] for r in rows])), 3),
               mean_share_r=round(float(np.mean([r["share_r"] for r in rows])), 3),
               mean_agree_queryfree=round(float(np.mean([r["agree_with_queryfree"] for r in rows])), 3),
               mean_std_a=round(float(np.mean([r["std_a"] for r in rows])), 4),
               mean_std_r=round(float(np.mean([r["std_r"] for r in rows])), 4))
    with open(os.path.join(args.out, "lerf_variance.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()