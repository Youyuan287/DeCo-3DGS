"""Encoder swap: does a stronger CLIP fix Delta's direction?

The exclusion chain left exactly one candidate: the encoder.  OpenSeg cannot be
run here (no TensorFlow on the box), but ViT-L-14-336 (768-d, the text space the
ScanNet OpenSeg arm used) is available and open_clip can load it, so the same
measurement can be repeated with a 2x larger CLIP and a 1.5x larger input.

For both encoders, on the SAME crops and the SAME candidates:
  rank1        is the best-IoU mask ranked first among candidates
  on failures (defined under ViT-B-16, fixed sample):
    dargright  does Delta = f_correct - f_picked point at the correct name
    dgap       margin of the correct name on Delta
Also reports rank1 when failures are re-defined under the new encoder, so a
changed failure set cannot be mistaken for a fix.

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

ENCODERS = [("ViT-B-16", "laion2b_s34b_b88k"), ("ViT-L-14-336", "openai")]


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

    L336 = "/root/rivermind-data/models/clip/ViT-L-14-336-openai.bin"
    models = {}
    for name, tag in ENCODERS:
        if name == "ViT-L-14-336":
            # the HF/timm route needs network; the OpenAI state_dict is local
            m, _, prep = open_clip.create_model_and_transforms(name, pretrained=None)
            sd = torch.load(L336, map_location="cpu", weights_only=False)
            miss, unexp = m.load_state_dict(sd, strict=False)
            assert not miss and not unexp, (miss[:3], unexp[:3])
        else:
            m, _, prep = open_clip.create_model_and_transforms(name, pretrained=tag)
        models[name] = (m.to("cuda").eval(), prep, open_clip.get_tokenizer(name))
        print("[load] %s ok (%d-d)" % (name, models[name][0].encode_image.__self__.visual.output_dim
                                       if hasattr(models[name][0].encode_image, "__self__") else -1), flush=True)

    def emb_mask(name, img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = img[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1]
        sub[~sm] = 127
        model, prep, _ = models[name]
        t = prep(Image.fromarray(sub)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            f = model.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy()

    def emb_txt(name, names):
        model, _, tok = models[name]
        with torch.no_grad():
            f = model.encode_text(tok(names).to("cuda"))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        T = {n: emb_txt(n, prompts) for n, _ in ENCODERS}
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            keep, F = [], {n: [] for n, _ in ENCODERS}
            for m in kept:
                fs = {n: emb_mask(n, img, m) for n, _ in ENCODERS}
                if any(f is None for f in fs.values()):
                    continue
                keep.append(m)
                for n in fs:
                    F[n].append(fs[n])
            if len(keep) < 6:
                continue
            for n in F:
                F[n] = np.stack(F[n])
            print("[%s/%s] masks=%d" % (sc, v, len(keep)), flush=True)

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
                ious = np.array([iou(m, gt) for m in keep])
                b = int(ious.argmax())
                row = dict(scene=sc, view=v, name=nm, iou_best=round(float(ious[b]), 3))
                picks = {}
                for n, _ in ENCODERS:
                    cos = F[n] @ T[n][qi]
                    rk = int(np.where(np.argsort(-cos) == b)[0][0]) + 1
                    row["rank1_" + n] = int(rk == 1)
                    row["rank_" + n] = rk
                    picks[n] = int(cos.argmax())
                # failure set fixed under the baseline encoder
                if picks["ViT-B-16"] != b:
                    for n, _ in ENCODERS:
                        d = F[n][b] - F[n][picks["ViT-B-16"]]
                        a = T[n] @ d
                        row["darg_" + n] = int(int(np.argmax(a)) == qi)
                        row["dgap_" + n] = round(float(a[qi] - np.max(np.delete(a, qi))), 4)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "encoder_swap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fails = [r for r in rows if "darg_ViT-B-16" in r]
    out = dict(n=len(rows), n_failures_baseline=len(fails))
    for n, _ in ENCODERS:
        out["rank1_" + n] = round(float(np.mean([r["rank1_" + n] for r in rows])), 3)
        out["meanrank_" + n] = round(float(np.mean([r["rank_" + n] for r in rows])), 2)
        out["fail_dargright_" + n] = round(float(np.mean([r["darg_" + n] for r in fails])), 3) if fails else None
        out["fail_medgap_" + n] = round(float(np.median([r["dgap_" + n] for r in fails])), 4) if fails else None
    with open(os.path.join(args.out, "encoder_swap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()