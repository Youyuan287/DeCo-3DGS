"""Does CLIP separate a part from its surroundings when the crop is CLEAN?

Both pipelines build every feature as CLIP(masked pixels).  So the question
"why are parts never identified" reduces to a property of CLIP itself, and it
can be measured independently of the pipeline:

    take the GT pixels of a part, crop them tightly, ask CLIP -- does the part
    name rank above the other names, and by how much?

If CLIP already fails on a clean crop, no aggregation / hierarchy / clustering
change can recover the part.  If it succeeds, the loss is created downstream and
the pipeline is the place to look.

Two datasets, same code path:
  ScanNet : GT labels are a per-point ply -> project into a frame (verified
            convention p_cam = (p - C) @ R) -> largest connected component of
            the class -> tight crop.
  LERF-OVS: test_mask/<view>/<query>.png is already a per-query mask -> tight crop.

Text side is each dataset's own name set: NYU40-19 for ScanNet, the scene prompts
from structure.json for LERF.  No GT is used in scoring; GT only decides what to crop.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

SUBSETS19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
NYU40 = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator",
    25: "television", 26: "paper", 27: "towel", 28: "shower curtain", 29: "box",
    30: "whiteboard", 31: "person", 32: "nightstand", 33: "toilet", 34: "sink",
    35: "lamp", 36: "bathtub", 37: "bag", 38: "otherstructure",
    39: "otherfurniture", 40: "otherprop",
}
PART_LIKE = {"window", "door", "table", "picture", "blinds", "counter",
             "porcelain hand", "chopsticks", "egg", "pork belly",
             "glass of water", "wavy noodles in bowl", "onion segments",
             "corn", "door handle", "bear nose", "hooves"}


def read_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64)
    return xyz, lab


def load_cam(scene, frame):
    m = np.loadtxt(os.path.join(scene, "pose", frame + ".txt")).astype(np.float64)
    K = np.loadtxt(os.path.join(scene, "intrinsic", frame + ".txt")).astype(np.float64)
    return m[:3, :3], m[:3, 3], (K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def project(pts, R, C, K, W, H):
    p = (pts - C[None, :]) @ R          # verified ScanNet convention
    z = p[:, 2]
    ok = z > 1e-6
    x = K[0] * p[:, 0] / np.maximum(z, 1e-6) + K[2]
    y = K[1] * p[:, 1] / np.maximum(z, 1e-6) + K[3]
    return x, y, z, ok


def largest_component(pix, H, W):
    m = np.zeros((H, W), bool)
    m[pix[:, 1], pix[:, 0]] = True
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (sizes.argmax() + 1)


def crop_tight(img, mask, pad=0.25):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    dx, dy = int(w * pad), int(h * pad)
    H, W = img.shape[:2]
    x0, x1 = max(0, x0 - dx), min(W, x1 + dx)
    y0, y1 = max(0, y0 - dy), min(H, y1 + dy)
    sub = img[y0:y1, x0:x1].copy()
    sub_mask = mask[y0:y1, x0:x1]
    sub[~sub_mask] = 127                      # same convention as the pipeline crop
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannet", default="/root/rivermind-data/datasets/scannet")
    ap.add_argument("--gt", default="/root/rivermind-data/repro/deco-b123/"
                                    "scannet10_full_20261029/gt")
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-class", type=int, default=3)
    ap.add_argument("--min-px", type=int, default=300)
    ap.add_argument("--only", default="all", choices=["all", "lerf", "scannet"])
    args = ap.parse_args()

    import open_clip
    dev = "cuda"
    model, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def embed_text(names):
        with torch.no_grad():
            t = tok(names).to(dev)
            f = model.encode_text(t)
        return f / f.norm(dim=-1, keepdim=True)

    def embed_img(arr):
        im = Image.fromarray(arr)
        t = prep(im).unsqueeze(0).to(dev)
        with torch.no_grad():
            f = model.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True))[0]

    rows = []

    def score(tag, ds, name, emb, names):
        T = embed_text(names)
        cos = (T @ emb).cpu().numpy()
        i = names.index(name)
        order = np.argsort(-cos)
        rank = int(np.where(order == i)[0][0]) + 1
        others = np.delete(cos, i)
        sigma = (cos[i] - others.mean()) / max(others.std(), 1e-6)
        top = names[int(order[0])]
        rows.append(dict(dataset=ds, scene=tag, name=name, rank=rank,
                         n_names=len(names), top1=top,
                         cos_true=round(float(cos[i]), 4),
                         margin_sigma=round(float(sigma), 3),
                         part_like=int(name in PART_LIKE)))
        return cos, sigma, top

    # ---------------- ScanNet ----------------
    scenes = (["scene0000_00", "scene0347_00", "scene0070_00"]
              if args.only in ("all", "scannet") else [])
    names19 = [NYU40[c] for c in SUBSETS19]
    for sc in scenes:
        sdir = os.path.join(args.scannet, sc)
        xyz, lab = read_ply(os.path.join(args.gt, sc + "_vh_clean_2.labels.ply"))
        frames = sorted(f[:-4] for f in os.listdir(os.path.join(sdir, "images"))
                        if f.endswith(".jpg"))
        img0 = np.asarray(Image.open(os.path.join(sdir, "images", frames[0] + ".jpg")))
        H, W = img0.shape[:2]
        for c in SUBSETS19:
            nm = NYU40[c]
            hit, seen = 0, 0
            for f in frames:
                if hit >= args.frames_per_class:
                    break
                R, C, K = load_cam(sdir, f)
                x, y, z, ok = project(xyz[lab == c], R, C, K, W, H)
                xi = np.round(x).astype(int)
                yi = np.round(y).astype(int)
                sel = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                if sel.sum() < args.min_px:
                    continue
                seen += 1
                mask = largest_component(np.stack([xi[sel], yi[sel]], 1), H, W)
                if mask.sum() < args.min_px:
                    continue
                img = np.asarray(Image.open(os.path.join(sdir, "images", f + ".jpg")).convert("RGB"))
                sub = crop_tight(img, mask)
                if sub is None or sub.shape[0] < 16 or sub.shape[1] < 16:
                    continue
                emb = embed_img(sub)
                score(sc, "scannet", nm, emb, names19)
                hit += 1

    # ---------------- LERF-OVS ----------------
    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    for sc in (["ramen", "teatime", "figurines"]
               if args.only in ("all", "lerf") else []):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        for v in st[sc]["views"]:
            md = os.path.join(sdir, "test_mask", str(v))
            if not os.path.isdir(md):
                continue
            for nm in prompts:
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    continue
                m = np.asarray(Image.open(p)) > 0
                if m.sum() < args.min_px:
                    continue
                ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
                if not os.path.exists(ip):
                    continue
                img = np.asarray(Image.open(ip).convert("RGB"))
                if img.shape[:2] != m.shape[:2]:
                    m2 = np.zeros(img.shape[:2], bool)
                    hh, ww = min(m.shape[0], img.shape[0]), min(m.shape[1], img.shape[1])
                    m2[:hh, :ww] = m[:hh, :ww]
                    m = m2
                sub = crop_tight(img, m)
                if sub is None or sub.shape[0] < 16 or sub.shape[1] < 16:
                    continue
                emb = embed_img(sub)
                score(sc, "lerf", nm, emb, prompts)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "clip_part_clean.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return {}
        return dict(n=len(r),
                    rank1=round(float(np.mean([x["rank"] == 1 for x in r])), 3),
                    mean_rank=round(float(np.mean([x["rank"] for x in r])), 2),
                    med_sigma=round(float(np.median([x["margin_sigma"] for x in r])), 3),
                    frac_sigma_lt1=round(float(np.mean([x["margin_sigma"] < 1 for x in r])), 3))

    summary = dict(
        scannet_part=agg(lambda x: x["dataset"] == "scannet" and x["part_like"]),
        scannet_other=agg(lambda x: x["dataset"] == "scannet" and not x["part_like"]),
        lerf_part=agg(lambda x: x["dataset"] == "lerf" and x["part_like"]),
        lerf_other=agg(lambda x: x["dataset"] == "lerf" and not x["part_like"]),
        all_part=agg(lambda x: x["part_like"]),
        all_other=agg(lambda x: not x["part_like"]),
    )
    with open(os.path.join(args.out, "clip_part_clean.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nper-item:", flush=True)
    for x in sorted(rows, key=lambda r: r["margin_sigma"]):
        print("  %-6s %-9s %-22s rank %d/%-2d top1=%-20s sigma=%6.2f"
              % (x["dataset"], x["scene"][:9], x["name"][:22], x["rank"],
                 x["n_names"], x["top1"][:20], x["margin_sigma"]), flush=True)


if __name__ == "__main__":
    main()