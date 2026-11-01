"""Part grounding gate: parent detection -> parent crop -> part detection -> smallest
multi-granularity mask containing the detected box center.

Training-free at query time; GroundingDINO weights and transformers are on the data disk.
GT is used only to (a) pick the parent category for the gate list and (b) evaluate.
"""
import os, sys, json, csv, argparse
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

# scene, image, query, parent category, part text
TESTS = [
    ("figurines", "frame_00041", "pirate hat", "rubber duck with hat", "hat."),
    ("figurines", "frame_00105", "pirate hat", "rubber duck with hat", "hat."),
    ("figurines", "frame_00152", "pirate hat", "rubber duck with hat", "hat."),
    ("figurines", "frame_00195", "pirate hat", "rubber duck with hat", "hat."),
    ("teatime", "frame_00002", "bear nose", "stuffed bear", "nose."),
    ("teatime", "frame_00025", "bear nose", "stuffed bear", "nose."),
    ("teatime", "frame_00107", "bear nose", "stuffed bear", "nose."),
    ("teatime", "frame_00025", "hooves", "sheep", "hooves."),
    ("teatime", "frame_00043", "hooves", "sheep", "hooves."),
    ("teatime", "frame_00107", "dall-e brand", "sheep", "brand. logo."),
    ("teatime", "frame_00129", "dall-e brand", "sheep", "brand. logo."),
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


def boxmask(box, h, w):
    x0, y0, x1, y1 = [int(v) for v in box]
    m = np.zeros((h, w), bool)
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
    return m


def detect(im, text, thr=0.08):
    inp = proc(images=im, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    return proc.post_process_grounded_object_detection(
        out, inp.input_ids, box_threshold=thr, text_threshold=thr,
        target_sizes=[im.size[::-1]])[0]


def masks_for(data_root, scene, img):
    p = os.path.join(data_root, "VALA/dataset/3dgs/lerf_ovs", scene,
                     "language_features", img)
    s = np.load(p + "_s.npy")
    ids = np.unique(s)
    ids = ids[ids >= 0]
    ms = []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append(s[lv] == mid)
    return ms


rows = []
for scene, img, prompt, parent, ptext in TESTS:
    ip = os.path.join(args.data_root, "VALA/dataset/3dgs/lerf_ovs", scene,
                      "images", img + ".jpg")
    im = Image.open(ip).convert("RGB")
    w, h = im.size
    gp = os.path.join(args.data_root, "ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label",
                      scene, img + ".json")
    anno = json.load(open(gp))
    gt = gt_of(anno, prompt, h, w)
    pm = gt_of(anno, parent, h, w)
    if not gt.any():
        continue
    rp = detect(im, parent + ".")
    if len(rp["boxes"]) == 0:
        rows.append(dict(scene=scene, image=img, prompt=prompt, parent=parent,
                         parent_box_iou=np.nan, parent_cov=np.nan, n_part_boxes=0,
                         top1_iou=np.nan, top3_union_iou=np.nan, oracle_box_iou=np.nan,
                         gt_area=int(gt.sum())))
        continue
    pi = int(torch.argmax(rp["scores"]))
    pbox = rp["boxes"][pi].tolist()
    pbm = boxmask(pbox, h, w)
    par_iou = iou(pbm, pm)
    par_cov = float((pbm & pm).sum() / max(pm.sum(), 1))
    x0, y0, x1, y1 = [int(v) for v in pbox]
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = im.crop((x0, y0, x1, y1))
    rq = detect(crop, ptext)
    qs = []
    for b, s in zip(rq["boxes"].tolist(), rq["scores"].tolist()):
        qs.append(([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0], float(s)))
    ms = masks_for(args.data_root, scene, img)
    # masks inside parent box, then smallest containing each part-box center
    cand = [m for m in ms if m.sum() > 20 and (m & pbm).sum() / m.sum() > 0.5]
    picks = []
    box_ious = []
    for b, s in qs:
        box_ious.append(iou(boxmask(b, h, w), gt))
        cx = int((b[0] + b[2]) / 2)
        cy = int((b[1] + b[3]) / 2)
        cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
        cont = [m for m in cand if m[cy, cx]]
        if cont:
            cont.sort(key=lambda m: m.sum())
            picks.append((cont[0], s))
    top1_iou = iou(picks[0][0], gt) if picks else np.nan
    union = np.zeros((h, w), bool)
    for m, s in picks[:3]:
        union |= m
    top3_iou = iou(union, gt) if picks else np.nan
    rows.append(dict(
        scene=scene, image=img, prompt=prompt, parent=parent,
        parent_box_iou=par_iou, parent_cov=par_cov, n_part_boxes=len(qs),
        top1_iou=top1_iou, top3_union_iou=top3_iou,
        oracle_box_iou=max(box_ious) if box_ious else np.nan,
        gt_area=int(gt.sum()),
    ))
    print("DONE", scene, img, prompt, rows[-1], flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "part_grounding_gate.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("ALL_DONE", len(rows))