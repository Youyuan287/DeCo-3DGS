"""LERF-OVS (not LERF-Mask): CLIP mask ranking vs Grounded-SAM, on the official frames.

Data, all from the LERF-OVS assets the evaluator itself uses:
  images + GT   ref/hf_lerf/lerf_ovs/label/<scene>/<frame>.{jpg,json}
  mask pool     VALA/dataset/3dgs/lerf_ovs/<scene>/language_features/<frame>_s.npy
  mask features same dir, _f.npy  (CLIP ViT-B-16, the pipeline's own observations)

Arms, paired on every (frame, category):
  clip    argmax over the pipeline's mask pool of  <f_mask, t_category>
  gsam    Grounding DINO top-1 box -> SAM box prompt -> mask with the best SAM score

Nothing is fitted: DINO and SAM are frozen, and the category text is only used at
inference.  GT is used to score, never to select.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys
sys.path.insert(0, "/root/rivermind-data/part_grounding_20261101/pylibs")

LABEL = "/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label"
POOL = "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs"
DINO = "/root/rivermind-data/part_grounding_20261101/model/grounding-dino-tiny"
SAM_CKPT = "/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth"


def gt_masks(h, w, anno):
    out = {}
    for o in anno["objects"]:
        c = o["category"]
        im = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(im)
        seg = o["segmentation"]
        polys = seg if (seg and isinstance(seg[0], (list, tuple))
                        and seg[0] and isinstance(seg[0][0], (list, tuple))) else [seg]
        for poly in polys:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(arr) >= 3:
                dr.polygon([tuple(p) for p in arr], fill=1)
        m = np.asarray(im) > 0
        out[c] = np.logical_or(out[c], m) if c in out else m
    return out


def pool_masks(scene, frame):
    p = os.path.join(POOL, scene, "language_features", frame)
    s = np.load(p + "_s.npy")                      # (levels, H, W)
    ids = sorted({int(v) for v in np.unique(s) if v >= 0})
    ms = []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append((mid, s[lv] == mid))
    return ids, ms


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    ap.add_argument("--box-thr", type=float, default=0.08)
    args = ap.parse_args()

    from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
    from segment_anything import SamPredictor, sam_model_registry
    import open_clip

    proc = GroundingDinoProcessor.from_pretrained(DINO, local_files_only=True)
    dino = GroundingDinoForObjectDetection.from_pretrained(
        DINO, local_files_only=True).cuda().eval()
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to("cuda")
    pred = SamPredictor(sam)
    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to("cuda").eval()
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb_txt(names):
        with torch.no_grad():
            f = clip.encode_text(tok(names).to("cuda"))
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    def emb_mask(img, m):
        ys, xs = np.nonzero(m)
        if len(xs) < 8:
            return np.zeros(512, np.float32)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = np.asarray(img)[y0:y1, x0:x1].copy()
        sm = m[y0:y1, x0:x1]
        sub[~sm] = 127
        t = prep(Image.fromarray(sub)).unsqueeze(0).to("cuda")
        with torch.no_grad():
            e = clip.encode_image(t)
        e = (e / e.norm(dim=-1, keepdim=True))[0].cpu().numpy()
        return e.astype(np.float32)

    rows = []
    for sc in args.scenes.split(","):
        gdir = os.path.join(LABEL, sc)
        if not os.path.isdir(gdir):
            print("missing", gdir, flush=True)
            continue
        frames = sorted(f[:-5] for f in os.listdir(gdir) if f.endswith(".json"))
        for fr in frames:
            anno = json.load(open(os.path.join(gdir, fr + ".json")))
            img = Image.open(os.path.join(gdir, fr + ".jpg")).convert("RGB")
            W, H = img.size
            gts = gt_masks(H, W, anno)
            ids, ms = pool_masks(sc, fr)
            f = np.stack([emb_mask(img, mm) for _, mm in ms]) if ms else np.zeros((0, 512))
            T = emb_txt(list(gts.keys()))
            pred.set_image(np.asarray(img))
            print("[%s/%s] pool=%d cats=%d" % (sc, fr, len(ms), len(gts)), flush=True)
            for ci, cat in enumerate(gts):
                gt = gts[cat]
                if gt.sum() < 100:
                    continue
                # --- clip arm
                sc_ = f @ T[ci]
                k = int(sc_.argmax())
                m = next((mm for mid, mm in ms if mid == ids[k]), None)
                clip_iou = iou(m, gt) if m is not None else 0.0
                # --- grounded-sam arm
                inp = proc(images=img, text=cat + ".", return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = dino(**inp)
                res = proc.post_process_grounded_object_detection(
                    out, inp.input_ids, box_threshold=args.box_thr,
                    text_threshold=args.box_thr, target_sizes=[img.size[::-1]])[0]
                gsam_iou = 0.0
                if len(res["boxes"]):
                    x0, y0, x1, y1 = res["boxes"][0].cpu().numpy()
                    masks, scores, _ = pred.predict(
                        box=np.array([[x0, y0, x1, y1]], dtype=np.float32),
                        multimask_output=True)
                    k2 = int(np.argmax(np.asarray(scores)))
                    gsam_iou = iou(masks[k2].astype(bool), gt)
                rows.append(dict(scene=sc, frame=fr, name=cat, gt_px=int(gt.sum()),
                                 clip_iou=round(clip_iou, 4),
                                 gsam_iou=round(gsam_iou, 4),
                                 n_boxes=int(len(res["boxes"]))))
                print("   %-22s clip=%.3f  gsam=%.3f" % (cat, clip_iou, gsam_iou),
                      flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lerfovs_gsam.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    c = np.array([r["clip_iou"] for r in rows])
    g = np.array([r["gsam_iou"] for r in rows])
    out = dict(n=len(rows),
               clip_med=round(float(np.median(c)), 3), gsam_med=round(float(np.median(g)), 3),
               clip_ge50=round(float((c >= 0.5).mean()), 3),
               gsam_ge50=round(float((g >= 0.5).mean()), 3),
               gsam_wins=round(float((g > c + 0.05).mean()), 3),
               clip_wins=round(float((c > g + 0.05).mean()), 3))
    with open(os.path.join(args.out, "lerfovs_gsam.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()