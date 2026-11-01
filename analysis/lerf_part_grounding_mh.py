"""Multi-hypothesis parent-conditioned part grounding gate.

For each true-part query:
  - several parent prompts x several parent boxes per prompt;
  - crop each parent box, detect the part, take the smallest multi-granularity mask
    containing each part-box center;
  - score masks by hypothesis support and detector scores; top1 / top3 union.
GT is only used for evaluation.
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

# scene, image, query, parent prompt variants, part text
TESTS = [
    ("figurines", "frame_00041", "pirate hat", ["rubber duck with hat.", "rubber duck.", "toy duck.", "duck."], "hat."),
    ("figurines", "frame_00105", "pirate hat", ["rubber duck with hat.", "rubber duck.", "toy duck.", "duck."], "hat."),
    ("figurines", "frame_00152", "pirate hat", ["rubber duck with hat.", "rubber duck.", "toy duck.", "duck."], "hat."),
    ("figurines", "frame_00195", "pirate hat", ["rubber duck with hat.", "rubber duck.", "toy duck.", "duck."], "hat."),
    ("teatime", "frame_00002", "bear nose", ["stuffed bear.", "bear.", "teddy bear."], "nose."),
    ("teatime", "frame_00025", "bear nose", ["stuffed bear.", "bear.", "teddy bear."], "nose."),
    ("teatime", "frame_00107", "bear nose", ["stuffed bear.", "bear.", "teddy bear."], "nose."),
    ("teatime", "frame_00025", "hooves", ["sheep.", "stuffed sheep.", "lamb."], "hooves."),
    ("teatime", "frame_00043", "hooves", ["sheep.", "stuffed sheep.", "lamb."], "hooves."),
    ("teatime", "frame_00107", "dall-e brand", ["sheep.", "stuffed sheep."], "brand. logo."),
    ("teatime", "frame_00129", "dall-e brand", ["sheep.", "stuffed sheep."], "brand. logo."),
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
for scene, img, prompt, pvars, ptext in TESTS:
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
    ms = masks_for(args.data_root, scene, img)

    support = defaultdict(int)
    variants = defaultdict(set)
    qlist = defaultdict(list)
    plist = defaultdict(list)
    pareas = defaultdict(list)
    n_hyp = 0
    for pv in pvars:
        rp = detect(im, pv)
        if len(rp["boxes"]) == 0:
            continue
        order = torch.argsort(rp["scores"], descending=True)[:5]
        for pi in order:
            pbox = rp["boxes"][int(pi)].tolist()
            pscore = float(rp["scores"][int(pi)])
            pbm = boxmask(pbox, h, w)
            x0, y0, x1, y1 = [int(v) for v in pbox]
            pad = int(0.15 * max(x1 - x0, y1 - y0))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            crop = im.crop((x0, y0, x1, y1))
            rq = detect(crop, ptext)
            if len(rq["boxes"]) == 0:
                continue
            qo = torch.argsort(rq["scores"], descending=True)[:10]
            cand = [j for j, m in enumerate(ms)
                    if m.sum() > 20 and (m & pbm).sum() / m.sum() > 0.5]
            for qi in qo:
                b = rq["boxes"][int(qi)].tolist()
                qscore = float(rq["scores"][int(qi)])
                cx = int((b[0] + b[2]) / 2 + x0)
                cy = int((b[1] + b[3]) / 2 + y0)
                cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
                cont = [j for j in cand if ms[j][cy, cx]]
                if not cont:
                    continue
                j = min(cont, key=lambda j: ms[j].sum())
                support[j] += 1
                variants[j].add(pv)
                qlist[j].append(qscore)
                plist[j].append(pscore)
                pareas[j].append(int(pbm.sum()))
                n_hyp += 1

    if not support:
        rows.append(dict(scene=scene, image=img, prompt=prompt, n_hyp=0,
                         top1_iou=np.nan, top3_union_iou=np.nan, top1_support=0))
        continue
    # dump per-mask aggregate features for offline parent-hypothesis ranking
    for j in support:
        mask_rows.append(dict(
            scene=scene, image=img, prompt=prompt,
            mask_idx=j, mask_area=int(ms[j].sum()),
            gt_iou=iou(ms[j], gt),
            n_support=int(support[j]), n_variants=len(variants[j]),
            max_q=float(max(qlist[j])), mean_q=float(np.mean(qlist[j])),
            max_p=float(max(plist[j])), mean_p=float(np.mean(plist[j])),
            parent_area_med=float(np.median(pareas[j])),
            area_ratio=float(ms[j].sum() / max(np.median(pareas[j]), 1)),
        ))
    ranked = sorted(support, key=lambda j: (support[j] / (ms[j].sum() ** 0.5), support[j], -ms[j].sum()), reverse=True)
    top1 = iou(ms[ranked[0]], gt)
    union = np.zeros((h, w), bool)
    for j in ranked[:3]:
        union |= ms[j]
    rows.append(dict(
        scene=scene, image=img, prompt=prompt, n_hyp=n_hyp,
        top1_iou=top1, top3_union_iou=iou(union, gt),
        top1_support=int(support[ranked[0]]),
        top1_area=int(ms[ranked[0]].sum()),
    ))
    print("DONE", scene, img, prompt, rows[-1], flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "part_grounding_mh.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("ALL_DONE", len(rows))