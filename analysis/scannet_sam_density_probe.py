"""Does SAM's proposal density cap the semantic unit on ScanNet?

The framework's failing classes are surface sub-regions (window / door / table /
wall).  The pipeline encodes at points_per_side=16, crop_n_layers=0.  Question:
at the current setting, does any SAM mask isolate those classes at all, and does
raising the proposal density create one?

GT is used ONLY to score masks.  Nothing here proposes, selects or weights with GT.
No CLIP, no text, no rendering, no evaluation harness.

phase A: project GT into every frame, pick the frames where the target classes are
         most visible.
phase B: run SamAutomaticMaskGenerator over (pps, crop_n_layers) variants on those
         frames and measure per-class isolation.

isolation_recall(c) = max over masks of recall(m vs class c) subject to
                      precision(m, c) >= 0.5 and |m| >= 20
i.e. the share of a class's visible pixels that can be covered by a mask which is
at least half that class.  A mask that merges window into wall has low precision
on window.
"""
import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

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
TARGET = [1, 7, 8, 9]          # wall / table / door / window -- the failing set
CONTROL = [2, 4, 5, 6, 24]     # floor / bed / chair / sofa / refrigerator
SCORE_CLASSES = [7, 8, 9]      # frame selection: table + door + window


def read_gt(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]),
                    np.asarray(v["z"])], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64)
    return xyz, lab


def load_cam(scene, frame):
    m = np.loadtxt(os.path.join(scene, "pose", frame + ".txt")).astype(np.float64)
    K = np.loadtxt(os.path.join(scene, "intrinsic", frame + ".txt")).astype(np.float64)
    return m[:3, :3], m[:3, 3], (K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def project(pts, R, C, K, W, H):
    # cameras.json / pose store the camera-to-world rotation as rows, so the
    # world->camera rotation is R.T applied on the right: p_cam = (p - C) @ R.
    # (using @ R.T instead puts everything in the wrong orientation)
    p = (pts - C[None, :]) @ R
    z = p[:, 2]
    ok = z > 1e-6
    x = K[0] * p[:, 0] / np.maximum(z, 1e-6) + K[2]
    y = K[1] * p[:, 1] / np.maximum(z, 1e-6) + K[3]
    return x, y, z, ok


def label_image(xyz, lab, R, C, K, W, H, fill_px=5.0):
    """Per-pixel GT class from z-buffered point projection, holes filled to fill_px."""
    x, y, z, ok = project(xyz, R, C, K, W, H)
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    sel = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    flat = np.full(H * W, -1, np.int32)
    if sel.sum():
        idx = yi[sel] * W + xi[sel]
        zz = z[sel]
        ll = lab[sel]
        order = np.argsort(-zz)          # nearest point wins
        flat[idx[order]] = ll[order]
    img = flat.reshape(H, W)
    empty = img < 0
    if empty.any() and not empty.all():
        dist, ind = ndimage.distance_transform_edt(empty, return_indices=True)
        src = img[ind[0], ind[1]]
        fill = empty & (dist <= fill_px)
        img[fill] = src[fill]
    return img


def score_masks(masks, lab_img, classes, min_area=20, prec_thr=0.5):
    out = {}
    for c in classes:
        Pc = lab_img == c
        nP = int(Pc.sum())
        if nP < 50:
            out[c] = dict(n_gt=nP, best_iou=0.0, iso_rec=0.0, n_prec50=0)
            continue
        best_iou = 0.0
        iso_rec = 0.0
        n_prec = 0
        for m in masks:
            n = int(m.sum())
            if n < min_area:
                continue
            inter = int(np.count_nonzero(m & Pc))
            if inter == 0:
                continue
            prec = inter / n
            rec = inter / nP
            iou = inter / (n + nP - inter)
            if iou > best_iou:
                best_iou = iou
            if prec >= prec_thr:
                n_prec += 1
                if rec > iso_rec:
                    iso_rec = rec
        out[c] = dict(n_gt=nP, best_iou=round(best_iou, 4),
                      iso_rec=round(iso_rec, 4), n_prec50=n_prec)
    return out


def select_frames(scene, xyz, lab, frames, k, min_px=150):
    """Class-balanced greedy pick: force one frame per failing class, then fill."""
    W, H = 648, 484
    rows = []
    for f in frames:
        R, C, K = load_cam(scene, f)
        img = label_image(xyz, lab, R, C, K, W, H)
        cnt = {c: int((img == c).sum()) for c in TARGET + CONTROL}
        rows.append((f, cnt))
    picked, seen = [], set()
    for c in TARGET:                       # wall, table, door, window
        cands = [r for r in rows if r[0] not in seen and r[1][c] >= min_px]
        cands.sort(key=lambda r: -r[1][c])
        if cands and len(picked) < k:
            picked.append(cands[0])
            seen.add(cands[0][0])
    rest = [r for r in rows if r[0] not in seen]
    rest.sort(key=lambda r: -sum(min(r[1][c], 3000) for c in SCORE_CLASSES))
    picked += rest[:max(0, k - len(picked))]
    picked.sort(key=lambda r: r[0])
    print("[select] scanned %d frames, picked %d (min %d px per class)"
          % (len(rows), len(picked), min_px), flush=True)
    for f, cnt in picked:
        print("  frame %s  " % f + " ".join("%s=%d" % (NYU40[c], cnt[c])
                                            for c in TARGET), flush=True)
    return [f for f, _ in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene id, e.g. scene0000_00")
    ap.add_argument("--data", default="/root/rivermind-data/datasets/scannet")
    ap.add_argument("--gt", default="/root/rivermind-data/repro/deco-b123/"
                                    "scannet10_full_20261029/gt")
    ap.add_argument("--sam-ckpt", default="/root/rivermind-data/VALA/ckpts/"
                                          "sam_vit_h_4b8939.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--variants", default="16:0,32:0,32:1,64:0")
    ap.add_argument("--pred-iou", type=float, default=0.7)
    ap.add_argument("--stability", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scene_dir = os.path.join(args.data, args.scene)
    gt_path = os.path.join(args.gt, args.scene + "_vh_clean_2.labels.ply")
    xyz, lab = read_gt(gt_path)
    print("[gt] %s  points=%d  classes=%d" % (gt_path, len(lab), len(np.unique(lab))),
          flush=True)

    frames = sorted(f[:-4] for f in os.listdir(os.path.join(scene_dir, "images"))
                    if f.endswith(".jpg"))
    print("[data] %s  frames=%d" % (scene_dir, len(frames)), flush=True)
    picked = select_frames(scene_dir, xyz, lab, frames, args.frames)
    if not picked:
        raise SystemExit("no frame has the target classes visible -- projection "
                         "convention is probably wrong")

    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).to(dev)
    print("[sam] loaded on %s" % dev, flush=True)

    variants = []
    for v in args.variants.split(","):
        pps, crop = v.split(":")
        variants.append((int(pps), int(crop)))

    W, H = 648, 484
    mask_rows, summary = [], {}
    for frame in picked:
        img = np.asarray(Image.open(os.path.join(scene_dir, "images",
                                                 frame + ".jpg")).convert("RGB"))
        H, W = img.shape[:2]
        R, C, K = load_cam(scene_dir, frame)
        lab_img = label_image(xyz, lab, R, C, K, W, H)
        counts = {c: int((lab_img == c).sum()) for c in TARGET + CONTROL}
        print("[frame %s] labelled px=%d  %s" %
              (frame, int((lab_img >= 0).sum()),
               " ".join("%s=%d" % (NYU40[c], counts[c]) for c in TARGET)), flush=True)
        for pps, crop in variants:
            tag = "pps%d_crop%d" % (pps, crop)
            gen = SamAutomaticMaskGenerator(
                sam, points_per_side=pps, pred_iou_thresh=args.pred_iou,
                stability_score_thresh=args.stability, crop_n_layers=crop,
                crop_n_points_downscale_factor=1, min_mask_region_area=100,
            )
            # the LangSplat SAM fork returns (all, small, medium, large)
            masks = gen.generate(img)[0]
            ms = [m["segmentation"] for m in masks]
            areas = np.array([int(m.sum()) for m in ms]) if ms else np.zeros(1, int)
            sc = score_masks(ms, lab_img, TARGET + CONTROL)
            print("  %-14s masks=%4d  area p50=%6.0f  " % (tag, len(ms), np.median(areas))
                  + " ".join("%s: iso=%.2f" % (NYU40[c], sc[c]["iso_rec"])
                             for c in TARGET), flush=True)
            summary.setdefault(frame, {})[tag] = dict(n_masks=len(ms),
                                                      area_p50=float(np.median(areas)),
                                                      per_class=sc)
            for c in TARGET + CONTROL:
                mask_rows.append(dict(scene=args.scene, frame=frame, variant=tag,
                                      pps=pps, crop=crop, cls=c,
                                      name=NYU40[c], n_gt_px=counts[c],
                                      best_iou=sc[c]["best_iou"],
                                      iso_rec=sc[c]["iso_rec"],
                                      n_prec50=sc[c]["n_prec50"],
                                      n_masks=len(ms),
                                      area_p50=float(np.median(areas))))

    csv_path = os.path.join(args.out, args.scene + "_density.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
        w.writeheader()
        w.writerows(mask_rows)
    with open(os.path.join(args.out, args.scene + "_density.json"), "w") as f:
        json.dump(dict(scene=args.scene, frames=picked,
                       pred_iou=args.pred_iou, stability=args.stability,
                       summary=summary), f, indent=2, default=float)
    print("[done] %s" % csv_path, flush=True)


if __name__ == "__main__":
    main()