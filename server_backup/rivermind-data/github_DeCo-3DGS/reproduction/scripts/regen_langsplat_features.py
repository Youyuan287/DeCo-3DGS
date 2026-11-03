#!/usr/bin/env python
# ReLaGS LERF-OVS reproduction — LangSplat feature regeneration (FAILED EXPERIMENT)
# 背景: figurines 的 langsplat/language_features 被误删, 本脚本尝试按官方 image_encoding.py
#       的参数(SamAutomaticMaskGenerator points_per_side=32 等 + CLIP ViT-B-16)重建。
# 结论: 四个层级掩膜无法严格复现官方的 mask2segmap+NMS 结构与覆盖度, 产物与非官方不等值,
#       已废弃, 不参与任何指标。保留以记录尝试过程; 正确做法是运行仓库 scripts/image_encoding.py。
"""Regenerate LangSplat-style language features (_s.npy / _f.npy) for a scene.

Format reverse-engineered from the surviving scenes:
  _s.npy : (4, H, W) int32, four disjoint id-ranges (one mask bucket per channel), -1 = unassigned
  _f.npy : (N, 512) float16, one CLIP ViT-B-16 feature per SAM mask

NOT byte-identical to the official release: SAM mask proposals and bucket assignment
are re-derived here. Use only as a fallback; validation is done on a scene whose
official features still exist.
"""
import os, sys, glob, argparse
import numpy as np
import torch
import cv2
from PIL import Image

def build_sam(ckpt):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to("cuda").eval()
    return SamAutomaticMaskGenerator(
        sam, points_per_side=64, pred_iou_thresh=0.88,
        stability_score_thresh=0.95, crop_n_layers=0, min_mask_region_area=0)

def build_clip():
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", cache_dir=None)
    m = m.cuda().eval()
    return m, pre

@torch.no_grad()
def clip_feats(model, pre, img_rgb, masks, bs=256):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1).cuda()
    out = []
    for i in range(0, len(masks), bs):
        crops = []
        for m in masks[i:i+bs]:
            ys, xs = np.where(m)
            if len(ys) == 0:
                crops.append(np.zeros((224,224,3), np.uint8)); continue
            y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            sub = img_rgb[y0:y1, x0:x1].astype(np.float32).copy()
            sub[m[y0:y1, x0:x1] == 0] = 0.0
            crops.append(cv2.resize(sub, (224,224), interpolation=cv2.INTER_LINEAR).astype(np.uint8))
        x = torch.stack([pre(Image.fromarray(c)) for c in crops]).cuda()
        f = model.encode_image(x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu())
    return torch.cat(out, 0).half()

BUCKETS = (0.295, 0.386, 0.207, 0.112)  # channel share observed in official data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen = build_sam(args.sam_ckpt)
    model, pre = build_clip()
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} images -> {args.out}", flush=True)
    for k, fp in enumerate(files):
        name = os.path.splitext(os.path.basename(fp))[0]
        img = cv2.imread(fp)
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = gen.generate(rgb)
        anns = res[0] if isinstance(res, tuple) else res
        anns = [a for a in anns if a.get("area", 0) >= 20]
        if len(anns) > 500:
            anns = sorted(anns, key=lambda a: -a["area"])[:500]
        if not anns:
            np.save(os.path.join(args.out, f"{name}_s.npy"), np.full((4,H,W), -1, np.int16))
            np.save(os.path.join(args.out, f"{name}_f.npy"), np.zeros((1,512), np.float16))
            continue
        anns = sorted(anns, key=lambda a: a["area"])
        N = len(anns)
        masks = [a["segmentation"].astype(bool) for a in anns]
        # Official data: every channel tiles ~0.85-0.98 of the image with globally
        # unique ids => four independent granularity levels over the same mask pool.
        # Level c takes, for each pixel, the mask at rank r_c among the masks covering it
        # (c=0 finest .. c=3 coarsest).
        n_cov = np.zeros((H, W), np.int32)
        for m in masks:
            n_cov += m
        rank = np.zeros((H, W), np.int32)
        s = np.full((4, H, W), -1, np.int16)
        for gid, m in enumerate(masks):
            if not m.any():
                continue
            for c in range(4):
                want = np.rint(c * np.maximum(n_cov - 1, 0) / 3.0).astype(np.int32)
                sel = m & (rank == want)
                if sel.any():
                    s[c][sel] = gid
            rank += m.astype(np.int32)
        feats = clip_feats(model, pre, rgb, masks)
        np.save(os.path.join(args.out, f"{name}_s.npy"), s)
        np.save(os.path.join(args.out, f"{name}_f.npy"), feats.numpy())
        if k % 10 == 0:
            print(f"  [{k+1}/{len(files)}] {name} masks={N}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
