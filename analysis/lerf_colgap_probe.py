"""Recompute on the LERF decision axis.

LERF decides argmax over CANDIDATES for a fixed query: rank_m S[m, q*].
The informative quantity on a failure is the COLUMN gap

    colgap = S[pick, q*] - S[best, q*]   >= 0 by construction on a failure

i.e. how far the correct mask falls behind the pick, in the only direction LERF
actually reads.  (An earlier run reported the margin of Delta across NAME axis,
which is the ScanNet axis, not this one.)

Configs: B16+global (pipeline), L336+global, B16+masked_max pooling.
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
L336 = "/root/rivermind-data/models/clip/ViT-L-14-336-openai.bin"


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

    m16, _, prep16 = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    m16 = m16.to("cuda").eval()
    tok16 = open_clip.get_tokenizer("ViT-B-16")

    m336, _, prep336 = open_clip.create_model_and_transforms(
        "ViT-L-14-336", pretrained=None)
    sd = torch.load(L336, map_location="cpu", weights_only=False)
    miss, unexp = m336.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    m336 = m336.to("cuda").eval()
    tok336 = open_clip.get_tokenizer("ViT-L-14-336")

    buf = {}
    h = m16.visual.ln_post.register_forward_hook(
        lambda mod, i, o: buf.__setitem__("t", o))

    def prep_crop(img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return None, None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = img[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1]
        out = sub.copy()
        out[~sm] = 127
        return Image.fromarray(out), sm.astype(np.uint8) * 255

    def feat16(img, m):
        pil, sm = prep_crop(img, m)
        if pil is None:
            return None
        x = prep16(pil).unsqueeze(0).to("cuda")
        with torch.no_grad():
            g = m16.encode_image(x)[0]
            t = buf["t"][0]
            patch = t[1:] @ m16.visual.proj
        g = g.detach()
        grid = int(round(patch.shape[0] ** 0.5))
        ms = np.asarray(Image.fromarray(sm).resize((grid * PATCH, grid * PATCH),
                                                   Image.NEAREST)) > 127
        cov = ms.reshape(grid, PATCH, grid, PATCH).mean((1, 3)).ravel()
        w = torch.tensor(cov, dtype=patch.dtype, device=patch.device)
        mean = (patch * w[:, None]).sum(0) / w.sum().clamp(min=1e-6)
        sel = w >= 0.5
        mx = patch[sel].max(0)[0] if int(sel.sum()) > 0 else patch.max(0)[0]
        return {k: (v / v.norm()).cpu().numpy() for k, v in
                (("global", g), ("masked_max", mx))}

    def feat336(img, m):
        pil, _ = prep_crop(img, m)
        if pil is None:
            return None
        x = prep336(pil).unsqueeze(0).to("cuda")
        with torch.no_grad():
            g = m336.encode_image(x)[0]
        g = g.detach()
        return g / g.norm()

    def emb_txt(model, tok, names):
        with torch.no_grad():
            f = model.encode_text(tok(names).to("cuda"))
        return (f / f.norm(dim=-1, keepdim=True))

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        T16 = emb_txt(m16, tok16, prompts)
        T336 = emb_txt(m336, tok336, prompts)
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = np.asarray(Image.open(ip).convert("RGB"))
            kept = mask_nms(list(gen.generate(img)[0]))
            MS, FG, FM, FL = [], [], [], []
            for m in kept:
                a = feat16(img, m)
                if a is None:
                    continue
                l = feat336(img, m)
                if l is None:
                    continue
                MS.append(m); FG.append(a["global"]); FM.append(a["masked_max"])
                FL.append(l.cpu().numpy())
            if len(MS) < 6:
                continue
            FG, FM, FL = np.stack(FG), np.stack(FM), np.stack(FL)
            print("[%s/%s] masks=%d" % (sc, v, len(MS)), flush=True)
            cfg = {"B16_global": (FG, T16.cpu().numpy()),
                   "L336_global": (FL, T336.cpu().numpy()),
                   "B16_maskedmax": (FM, T16.cpu().numpy())}
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
                for k, (F, T) in cfg.items():
                    s = F @ T[qi]
                    p_ = int(s.argmax())
                    row["rank_" + k] = int(np.where(np.argsort(-s) == b)[0][0]) + 1
                    row["colgap_" + k] = round(float(s[p_] - s[b]), 4)
                rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "colgap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    KEYS = ["B16_global", "L336_global", "B16_maskedmax"]
    out = dict(n=len(rows))
    base_fail = [r for r in rows if r["rank_B16_global"] > 1]
    out["n_failures_B16"] = len(base_fail)
    for k in KEYS:
        out["rank1_" + k] = round(float(np.mean([r["rank_" + k] == 1 for r in rows])), 3)
        out["medrank_" + k] = round(float(np.median([r["rank_" + k] for r in rows])), 1)
        if base_fail:
            v = [r["colgap_" + k] for r in base_fail]
            out["fail_medcolgap_" + k] = round(float(np.median(v)), 4)
            out["fail_colgap_lt_half_" + k] = round(float(np.mean([x < 0.5 * np.median(
                [r["colgap_B16_global"] for r in base_fail]) for x in v])), 3)
    with open(os.path.join(args.out, "colgap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()