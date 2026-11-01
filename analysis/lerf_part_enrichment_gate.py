"""Part-whole semantic enrichment gate (2D, no GT selection).

For every true-part query, compare text rankings of the GT-best mask under:
  A local   : existing per-mask CLIP _f
  B ctx     : max(local, bbox-context, 1.5x-context)
  C partwhole: max(local, parent, residual=local-parent)
  D all     : max(local, ctx, parent, residual)
Late interaction = max over tokens. GT is evaluation only.
"""
import os, sys, json, csv, argparse
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip

sys.path.insert(0, "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS")
from utils.vlm_utils import ClipSimMeasure

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

vlm = ClipSimMeasure()
vlm.load_model()
model, _, pre = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16")
model = model.cuda().eval()

TESTS = [
    ("figurines", "frame_00041", "pirate hat"),
    ("figurines", "frame_00105", "pirate hat"),
    ("figurines", "frame_00152", "pirate hat"),
    ("figurines", "frame_00195", "pirate hat"),
    ("teatime", "frame_00002", "bear nose"),
    ("teatime", "frame_00025", "bear nose"),
    ("teatime", "frame_00107", "bear nose"),
    ("teatime", "frame_00025", "hooves"),
    ("teatime", "frame_00043", "hooves"),
    ("teatime", "frame_00107", "dall-e brand"),
    ("teatime", "frame_00129", "dall-e brand"),
]


def poly2mask(shape, segs):
    m = np.zeros(shape, np.uint8)
    if not segs:
        return m
    if isinstance(segs[0][0], (int, float)):
        segs = [segs]
    for p in segs:
        cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 2)], 1)
    return m


def gt_of(anno, cat, h, w):
    g = np.zeros((h, w), np.uint8)
    for o in anno["objects"]:
        if o["category"] == cat:
            g = np.maximum(g, poly2mask((h, w), o["segmentation"]))
    return g.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def crop_feature(im, mask, scale=1.0, mask_bg=False):
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if scale != 1.0:
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        hh, ww = (y1 - y0) * scale / 2, (x1 - x0) * scale / 2
        y0, y1 = max(0, int(cy - hh)), min(im.height, int(cy + hh))
        x0, x1 = max(0, int(cx - ww)), min(im.width, int(cx + ww))
    sub = np.asarray(im.crop((x0, y0, x1, y1))).copy()
    if mask_bg:
        mm = mask[y0:y1, x0:x1]
        sub[~mm] = 0
    h, w = sub.shape[:2]
    l = max(h, w)
    pad = np.zeros((l, l, 3), np.uint8)
    pad[(l - h) // 2:(l - h) // 2 + h, (l - w) // 2:(l - w) // 2 + w] = sub
    return pad


@torch.no_grad()
def encode_tiles(im, masks, scale, mask_bg, bs=32):
    feats = []
    for i in range(0, len(masks), bs):
        tiles = [pre(Image.fromarray(crop_feature(im, m, scale, mask_bg)))[None]
                 for m in masks[i:i + bs]]
        x = torch.cat(tiles, 0).cuda().half()
        z = model.encode_image(x).float()
        z = z / z.norm(dim=-1, keepdim=True)
        feats.append(z.cpu())
    return torch.cat(feats, 0).numpy()


def np_norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


rows = []
for scene, img, prompt in TESTS:
    src = os.path.join(args.data_root, "VALA/dataset/3dgs/lerf_ovs", scene,
                       "language_features", img)
    s = np.load(src + "_s.npy")
    f0 = np.load(src + "_f.npy").astype(np.float32)
    ids = np.unique(s)
    ids = ids[ids >= 0]
    ms, levels = [], []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append(s[lv] == mid)
        levels.append(lv)
    local = np_norm(f0[ids])
    im = Image.open(os.path.join(args.data_root, "VALA/dataset/3dgs/lerf_ovs",
                                 scene, "images", img + ".jpg")).convert("RGB")
    ctx = encode_tiles(im, ms, 1.0, False)
    ctx15 = encode_tiles(im, ms, 1.5, False)
    # parent: smallest coarser mask containing >0.8 of the mask
    parent = np.full(len(ms), -1)
    areas = np.array([m.sum() for m in ms])
    for i, m in enumerate(ms):
        best, ba = -1, 1e18
        for j, n in enumerate(ms):
            if levels[j] <= levels[i]:
                continue
            inter = (m & n).sum()
            if inter > 0 and inter / m.sum() > 0.8 and n.sum() < ba:
                best, ba = j, n.sum()
        parent[i] = best
    pf = np.zeros_like(local)
    for i in range(len(ms)):
        pf[i] = local[parent[i]] if parent[i] >= 0 else 0.0
    resid = np_norm(local - pf)
    resid[parent < 0] = 0.0
    anno = json.load(open(os.path.join(args.data_root,
        "ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label", scene, img + ".json")))
    h, w = ms[0].shape
    gt = gt_of(anno, prompt, h, w)
    if not gt.any():
        continue
    ious = np.array([iou(m, gt) for m in ms])
    jb = int(np.argmax(ious))
    vlm.encode_text(prompt)
    t = vlm.text_feature[0].detach().cpu().numpy()
    t = t / max(float(np.linalg.norm(t)), 1e-8)

    def rank(sc):
        return int((sc > sc[jb]).sum()) + 1, int(np.argmax(sc))

    sc_local = local @ t
    sc_ctx = np.maximum(sc_local, np.maximum(ctx @ t, ctx15 @ t))
    sc_pw = np.maximum(sc_local, np.maximum(pf @ t, resid @ t))
    sc_all = np.maximum(sc_ctx, sc_pw)
    out = dict(scene=scene, image=img, prompt=prompt, best_iou=float(ious[jb]))
    for name, sc in [("local", sc_local), ("ctx", sc_ctx),
                     ("partwhole", sc_pw), ("all", sc_all)]:
        r, top = rank(sc)
        out[name + "_rank"] = r
        out[name + "_top_iou"] = float(ious[top])
    rows.append(out)
    print("DONE", out, flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "enrichment_gate.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("ALL_DONE", len(rows))