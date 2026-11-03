"""Text-only multi-granularity mask selection on LERF-OVS.

Uses the existing per-mask CLIP features (language_features/{img}_f.npy, rows are
global mask IDs) and global multi-level mask maps ({img}_s.npy). For each query:
  - score every mask by cosine(text, mask feature);
  - top-1 mask, and top-k unions (raw and mask-NMS);
  - compare against GT (IoU). No GT is used for selection.
The GT oracle from lerf_multigranularity_gate.py is joined offline.
"""
import os, sys, json, csv
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

vlm = ClipSimMeasure()
vlm.load_model()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def masks_and_feats(img):
    s = np.load(os.path.join(args.source, "language_features", img + "_s.npy"))
    f = np.load(os.path.join(args.source, "language_features", img + "_f.npy")).astype(np.float32)
    ids = np.unique(s)
    ids = ids[ids >= 0]
    masks = []
    levels = []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        masks.append(s[lv] == mid)
        levels.append(lv)
    fn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
    return ids, masks, levels, fn[ids]


def select_union(order, masks, k, nms=0.0):
    sel = []
    for j in order:
        if nms > 0 and sel:
            if max(iou(masks[j], masks[q]) for q in sel) > nms:
                continue
        sel.append(j)
        if len(sel) >= k:
            break
    u = np.zeros_like(masks[0], bool)
    for j in sel:
        u |= masks[j]
    return u, sel


gt_dir = os.path.join(args.gt_root, args.scene)
imgs = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".jpg"))
rows = []
nq = 0
for img in imgs:
    anno = json.load(open(os.path.join(gt_dir, img + ".json")))
    ids, masks, levels, fn = masks_and_feats(img)
    h, w = masks[0].shape
    prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
    for prompt in prompts:
        nq += 1
        if args.limit_queries and nq > args.limit_queries:
            break
        gt = np.zeros((h, w), np.uint8)
        for o in anno["objects"]:
            if o["category"] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
        gt = gt.astype(bool)
        if not gt.any():
            continue
        vlm.encode_text(prompt)
        t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
        t = t / max(float(np.linalg.norm(t)), 1e-8)
        scores = fn @ t
        order = np.argsort(-scores)
        top1 = masks[order[0]]
        u1, s1 = select_union(order, masks, 1)
        u3, s3 = select_union(order, masks, 3)
        u5, s5 = select_union(order, masks, 5)
        u3n, s3n = select_union(order, masks, 3, nms=0.8)
        u5n, s5n = select_union(order, masks, 5, nms=0.8)
        rows.append(dict(
            scene=args.scene, image=img, prompt=prompt,
            top1_iou=iou(top1, gt), top1_level=levels[order[0]],
            top1_score=float(scores[order[0]]), top2_score=float(scores[order[1]]) if len(order) > 1 else float("nan"),
            top1_area=int(top1.sum()), gt_area=int(gt.sum()),
            top3_iou=iou(u3, gt), top5_iou=iou(u5, gt),
            top3n_iou=iou(u3n, gt), top5n_iou=iou(u5n, gt),
            top3_levels=";".join(str(levels[q]) for q in s3),
            top5_levels=";".join(str(levels[q]) for q in s5),
            top3n_levels=";".join(str(levels[q]) for q in s3n),
            top5n_levels=";".join(str(levels[q]) for q in s5n),
        ))
    print("IMG", img, len(prompts), flush=True)
    if args.limit_queries and nq >= args.limit_queries:
        break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_mgselect.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))