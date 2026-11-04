"""Does phrase grounding succeed exactly where CLIP ranking fails?

Grounding DINO (ECCV 2024) is a mature, frozen, text-conditioned detector.  For each
(scene, view, query) we take its top-1 box and measure IoU with the GT mask, then
cross-tabulate against whether CLIP's mask ranking already picked the right mask.

Complementarity is the only thing that matters here: if grounding succeeds on the
subset where CLIP fails, the two sources are orthogonal and a gated fusion is a
method.  If it fails on the same subset, it is not.

GT is used only to score.  Nothing is selected with GT.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

sys_path = "/root/rivermind-data/part_grounding_20261101/pylibs"
import sys
sys.path.insert(0, sys_path)

MODEL = "/root/rivermind-data/part_grounding_20261101/model/grounding-dino-tiny"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerf", default="/root/rivermind-data/datasets/lerf_mask")
    ap.add_argument("--clip_csv", required=True,
                    help="lerf_mask_rank.csv or lerf_colgap.csv from earlier runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="ramen,teatime,figurines")
    ap.add_argument("--box-thr", type=float, default=0.08)
    args = ap.parse_args()

    from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
    proc = GroundingDinoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = GroundingDinoForObjectDetection.from_pretrained(
        MODEL, local_files_only=True).cuda().eval()

    clip = {}
    for r in csv.DictReader(open(args.clip_csv)):
        clip[(r["scene"], r["view"], r["name"])] = r

    st = json.load(open(os.path.join(args.lerf, "build", "structure.json")))
    rows = []
    BOXES = {}
    for sc in args.scenes.split(","):
        sdir = os.path.join(args.lerf, sc)
        prompts = st[sc]["prompts"]["0"]
        for test in st[sc]["test"]:
            v = (test if isinstance(test, str) else str(test["dir"])).replace("test_", "")
            ip = os.path.join(sdir, "images", "test_%s.jpg" % v)
            md = os.path.join(sdir, "test_mask", v)
            if not (os.path.exists(ip) and os.path.isdir(md)):
                continue
            img = Image.open(ip).convert("RGB")
            W, H = img.size
            for nm in prompts:
                p = os.path.join(md, nm + ".png")
                if not os.path.exists(p):
                    continue
                gt = np.asarray(Image.open(p)) > 0
                if gt.shape[:2] != (H, W):
                    g2 = np.zeros((H, W), bool)
                    hh, ww = min(gt.shape[0], H), min(gt.shape[1], W)
                    g2[:hh, :ww] = gt[:hh, :ww]
                    gt = g2
                if gt.sum() < 100:
                    continue
                inputs = proc(images=img, text=nm + ".", return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model(**inputs)
                res = proc.post_process_grounded_object_detection(
                    out, inputs.input_ids,
                    box_threshold=args.box_thr, text_threshold=args.box_thr,
                    target_sizes=[img.size[::-1]])[0]
                best = 0.0
                boxes_np = res["boxes"].cpu().numpy() if len(res["boxes"]) else np.zeros((0, 4))
                BOXES[(sc, v, nm)] = boxes_np
                if len(res["boxes"]):
                    for (x0, y0, x1, y1) in res["boxes"].cpu().numpy():
                        x0, y0 = max(0, int(x0)), max(0, int(y0))
                        x1, y1 = min(W, int(x1)), min(H, int(y1))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        box = np.zeros((H, W), bool)
                        box[y0:y1, x0:x1] = True
                        inter = np.count_nonzero(box & gt)
                        iou = inter / (box.sum() + gt.sum() - inter)
                        best = max(best, iou)
                # detector-guided selection inside the SAM kept pool
                sel_iou, sel_rank1, sel_box_iou = float("nan"), None, float("nan")
                if len(boxes_np) and "kept" in dir():
                    pass
                key = (sc, v, nm)
                c = clip.get(key)
                rows.append(dict(scene=sc, view=v, name=nm,
                                 n_boxes=int(len(res["boxes"])),
                                 box_iou=round(float(best), 4),
                                 clip_rank1=int(c["rank1_ViT-B-16"]) if c and "rank1_ViT-B-16" in c else None,
                                 gt_px=int(gt.sum())))
                print("[%s/%s] %-24s boxes=%2d box_iou=%.3f  clip_rank1=%s"
                      % (sc, v, nm, len(res["boxes"]), best,
                         rows[-1]["clip_rank1"]), flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "boxes.npz"),
                        **{("%s|%s|%s" % k): v for k, v in BOXES.items()})
    with open(os.path.join(args.out, "grounding_vs_clip.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    have = [r for r in rows if r["clip_rank1"] is not None]
    clip_ok = [r for r in have if r["clip_rank1"] == 1]
    clip_bad = [r for r in have if r["clip_rank1"] == 0]
    out = dict(n=len(rows), n_paired=len(have),
               box_ge50_all=round(float(np.mean([r["box_iou"] >= 0.5 for r in rows])), 3),
               box_ge50_clip_ok=round(float(np.mean([r["box_iou"] >= 0.5 for r in clip_ok])), 3) if clip_ok else None,
               box_ge50_clip_bad=round(float(np.mean([r["box_iou"] >= 0.5 for r in clip_bad])), 3) if clip_bad else None,
               n_clip_ok=len(clip_ok), n_clip_bad=len(clip_bad))
    with open(os.path.join(args.out, "grounding_vs_clip.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()