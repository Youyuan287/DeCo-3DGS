"""Leakage-free parent inference gate.

Parent candidates: class-agnostic multi-granularity SAM masks from _s.npy
(levels 2/3, area 0.3%-45% of image, IoU-NMS). No GT parent category/prompt.
For each parent candidate: crop -> GroundingDINO(part text) -> part box center ->
smallest multi-granularity mask containing the center. Select by detector score /
cross-parent support. GT is evaluation only.
"""
import os, sys, json, csv, argparse
from collections import defaultdict
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--model_dir", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

proc = GroundingDinoProcessor.from_pretrained(args.model_dir, local_files_only=True)
model = GroundingDinoForObjectDetection.from_pretrained(args.model_dir, local_files_only=True).cuda().eval()

# query -> part text (query-only parser for the gate; no GT parent)
TESTS = [
    ("figurines", "frame_00041", "pirate hat", "hat."),
    ("figurines", "frame_00105", "pirate hat", "hat."),
    ("figurines", "frame_00152", "pirate hat", "hat."),
    ("figurines", "frame_00195", "pirate hat", "hat."),
    ("teatime", "frame_00002", "bear nose", "nose."),
    ("teatime", "frame_00025", "bear nose", "nose."),
    ("teatime", "frame_00107", "bear nose", "nose."),
    ("teatime", "frame_00025", "hooves", "hooves."),
    ("teatime", "frame_00043", "hooves", "hooves."),
    ("teatime", "frame_00107", "dall-e brand", "brand. logo."),
    ("teatime", "frame_00129", "dall-e brand", "brand. logo."),
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
    return float((a & b).sum() / u) if u else 0.0


def detect(im, text, thr=0.08):
    inp = proc(images=im, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    return proc.post_process_grounded_object_detection(
        out, inp.input_ids, box_threshold=thr, text_threshold=thr,
        target_sizes=[im.size[::-1]])[0]


def load_masks(data_root, scene, img):
    p = os.path.join(data_root, "VALA/dataset/3dgs/lerf_ovs", scene,
                     "language_features", img)
    s = np.load(p + "_s.npy")
    ids = np.unique(s)
    ids = ids[ids >= 0]
    ms, levels, ids_out = [], [], []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append(s[lv] == mid)
        levels.append(lv)
        ids_out.append(int(mid))
    return ms, np.array(levels), ids_out


rows = []
for scene, img, prompt, ptext in TESTS:
    ip = os.path.join(args.data_root, "VALA/dataset/3dgs/lerf_ovs", scene,
                      "images", img + ".jpg")
    im = Image.open(ip).convert("RGB")
    w, h = im.size
    gp = os.path.join(args.data_root, "ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label",
                      scene, img + ".json")
    anno = json.load(open(gp))
    gt = gt_of(anno, prompt, h, w)
    if not gt.any():
        continue
    ms, levels, mid_list = load_masks(args.data_root, scene, img)
    n_pix = h * w
    # class-agnostic parent candidates: levels 2/3, reasonable size, NMS
    cand = []
    for j, m in enumerate(ms):
        a = int(m.sum())
        if 0.003 * n_pix <= a <= 0.45 * n_pix:
            cand.append(j)
    cand.sort(key=lambda j: -ms[j].sum())
    keep = []
    for j in cand:
        if all(iou(ms[j], ms[k]) < 0.75 for k in keep):
            keep.append(j)
        if len(keep) >= 60:
            break
    # part hypotheses over parent candidates
    support = defaultdict(int)
    maxq = defaultdict(float)
    parent_area = defaultdict(list)
    parent_id = defaultdict(list)
    for j in keep:
        m = ms[j]
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = int(0.15 * max(y1 - y0, x1 - x0))
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        crop = im.crop((x0, y0, x1, y1))
        rq = detect(crop, ptext)
        if len(rq["boxes"]) == 0:
            continue
        qo = torch.argsort(rq["scores"], descending=True)[:5]
        for qi in qo:
            b = rq["boxes"][int(qi)].tolist()
            qs = float(rq["scores"][int(qi)])
            cx = int((b[0] + b[2]) / 2 + x0)
            cy = int((b[1] + b[3]) / 2 + y0)
            cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
            cont = [k for k in range(len(ms))
                    if ms[k].sum() > 20 and ms[k][cy, cx]
                    and (ms[k] & m).sum() / ms[k].sum() > 0.5]
            if not cont:
                continue
            k = min(cont, key=lambda k: ms[k].sum())
            support[k] += 1
            maxq[k] = max(maxq[k], qs)
            parent_area[k].append(int(m.sum()))
            parent_id[k].append(j)
    if not support:
        rows.append(dict(scene=scene, image=img, prompt=prompt, n_parent=len(keep),
                         top1_iou=np.nan, top3_union_iou=np.nan, top1_support=0))
        continue
    ranked = sorted(support, key=lambda k: (support[k], maxq[k]), reverse=True)
    top1 = iou(ms[ranked[0]], gt)
    union = np.zeros((h, w), bool)
    for k in ranked[:3]:
        union |= ms[k]
    rows.append(dict(
        scene=scene, image=img, prompt=prompt, n_parent=len(keep),
        top1_iou=top1, top3_union_iou=iou(union, gt),
        top1_support=int(support[ranked[0]]),
        top1_q=float(maxq[ranked[0]]),
        top1_area=int(ms[ranked[0]].sum()),
        top1_area_ratio=float(ms[ranked[0]].sum() / max(np.median(parent_area[ranked[0]]), 1)),
    ))
    print("DONE", scene, img, prompt, rows[-1], flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "part_grounding_nogt.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("ALL_DONE", len(rows))