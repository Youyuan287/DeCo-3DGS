#!/usr/bin/env python3
"""Generate source-matched R0 crop and R1 dense-token readouts for one LERF scene.

This intentionally reuses the project's SAM proposal code and OpenCLIP checkpoint.
It does not touch NAGs or Gaussian models. R1 differs only in how the same proposal
masks read the image: fixed crop embedding (R0) vs. dense patch-token interpolation
and fixed mask pooling (R1).
"""
import argparse, os, cv2, numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts import image_encoding as ie


def pool_by_map(dense, seg_map):
    # dense: [H,W,C], seg_map: [H,W] with -1 for background
    h, w, c = dense.shape
    flat = dense.reshape(h * w, c)
    labels = torch.from_numpy(seg_map.reshape(-1)).to(dense.device, dtype=torch.long)
    keep = labels >= 0
    if not bool(keep.any()):
        return torch.zeros((0, c), device=dense.device)
    ids = labels[keep]
    n = int(ids.max().item()) + 1
    out = torch.zeros((n, c), device=dense.device, dtype=dense.dtype)
    cnt = torch.zeros((n, 1), device=dense.device, dtype=dense.dtype)
    out.index_add_(0, ids, flat[keep])
    cnt.index_add_(0, ids, torch.ones((ids.numel(), 1), device=dense.device, dtype=dense.dtype))
    return F.normalize(out / cnt.clamp_min(1), dim=-1)


@torch.no_grad()
def dense_tokens(model, image):
    # Match image_encoding.py preprocessing (RGB float image, 224 square input).
    x = image.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    x = model.process(x).half()
    v = model.model.visual
    out = v.forward_intermediates(
        x, image_indices=[-1], normalize_intermediates=True,
        intermediates_only=True, output_fmt="NLC")
    tok = out["image_intermediates"][-1]  # [1,196,768] for ViT-B/16
    if getattr(v, "proj", None) is not None:
        tok = tok @ v.proj
    tok = F.normalize(tok.float(), dim=-1)
    side = int(tok.shape[1] ** 0.5)
    assert side * side == tok.shape[1], tok.shape
    tok = tok.transpose(1, 2).reshape(1, tok.shape[-1], side, side)
    tok = F.interpolate(tok, size=image.shape[:2], mode="bilinear", align_corners=False)
    return tok[0].permute(1, 2, 0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--sam", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    out0 = os.path.join(args.out_root, "language_features_r0")
    out1 = os.path.join(args.out_root, "language_features_r1")
    os.makedirs(out0, exist_ok=True); os.makedirs(out1, exist_ok=True)

    ie.model = ie.OpenCLIPNetwork(ie.OpenCLIPNetworkConfig())
    sam = ie.sam_model_registry["vit_h"](checkpoint=args.sam).to("cuda")
    ie.mask_generator = ie.SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.7,
        box_nms_thresh=0.7, stability_score_thresh=0.85,
        crop_n_layers=1, crop_n_points_downscale_factor=1,
        min_mask_region_area=100)

    images = sorted(f for f in os.listdir(os.path.join(args.source, "images"))
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for name in tqdm(images, desc="figurines readout"):
        stem = os.path.splitext(name)[0]
        if all(os.path.exists(os.path.join(d, stem + suf)) for d in (out0, out1)
               for suf in ("_f.npy", "_s.npy")):
            continue
        bgr = cv2.imread(os.path.join(args.source, "images", name), cv2.IMREAD_COLOR)
        if bgr is None: raise RuntimeError(name)
        h0, w0 = bgr.shape[:2]
        scale = max(h0 / 1080.0, 1.0)
        image = cv2.resize(bgr, (int(w0 / scale), int(h0 / scale)))
        image_t = torch.from_numpy(image)
        seg_images, seg_maps = ie.sam_encoder(image_t.permute(2, 0, 1)[None])
        dense = dense_tokens(ie.model, torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

        r0, r1, maps = [], [], []
        offset = 0
        for mode in ("default", "s", "m", "l"):
            if mode not in seg_maps: continue
            sm = seg_maps[mode]
            maps.append(sm.astype(np.int32) + np.where(sm >= 0, offset, 0))
            crops = seg_images[mode]
            with torch.no_grad():
                f = F.normalize(ie.model.model.encode_image(crops), dim=-1).float()
            r0.append(f.cpu())
            pooled = pool_by_map(dense, sm)
            r1.append(pooled.cpu())
            offset += int(sm.max()) + 1 if sm.size and sm.max() >= 0 else 0
        fmap0 = torch.cat(r0, dim=0).numpy().astype(np.float16)
        fmap1 = torch.cat(r1, dim=0).numpy().astype(np.float16)
        smap = np.stack(maps, axis=0).astype(np.int32)
        smap[smap < 0] = -1
        for d, f in ((out0, fmap0), (out1, fmap1)):
            np.save(os.path.join(d, stem + "_f.npy"), f)
            np.save(os.path.join(d, stem + "_s.npy"), smap)
        assert fmap0.shape == fmap1.shape and fmap0.shape[1] == 512


if __name__ == "__main__":
    main()
