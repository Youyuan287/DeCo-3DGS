#!/usr/bin/env python3
"""Build R1 dense-token readout from an existing, fixed proposal cache.

The `_s.npy` maps and images are not regenerated. Only the semantic readout is
changed; this is the controlled follow-up to the failed/too-expensive SAM path.
"""
import argparse, os, glob, cv2, numpy as np, torch
import torch.nn.functional as F
import open_clip
from tqdm import tqdm


def pool_by_maps(dense, seg_maps):
    h, w, c = dense.shape
    flat = dense.reshape(h*w, c).repeat(seg_maps.shape[0], 1)
    labels = torch.from_numpy(seg_maps.reshape(-1)).to(dense.device, dtype=torch.long)
    keep = labels >= 0
    ids = labels[keep]
    n = int(ids.max().item()) + 1
    out = torch.zeros((n, c), device=dense.device)
    cnt = torch.zeros((n, 1), device=dense.device)
    out.index_add_(0, ids, flat[keep])
    cnt.index_add_(0, ids, torch.ones((ids.numel(), 1), device=dense.device))
    return F.normalize(out / cnt.clamp_min(1), dim=-1)


@torch.no_grad()
def dense_tokens(model, process, image):
    # Do not use OpenCLIP's standard inference transform here: it performs a
    # center crop, which destroys the pixel-to-token geometry needed by the
    # fixed proposal maps.  Resize the complete image to the ViT input grid,
    # then apply only the CLIP normalization.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
    x = ((x - mean) / std).half().cuda()
    v = model.visual
    out = v.forward_intermediates(x, indices=[-1], normalize_intermediates=True,
                                  intermediates_only=True, output_fmt="NLC")
    tok = out["image_intermediates"][-1]
    if getattr(v, "proj", None) is not None:
        tok = tok @ v.proj
    tok = F.normalize(tok.float(), dim=-1)
    side = int(tok.shape[1] ** 0.5)
    assert side * side == tok.shape[1], tok.shape
    tok = tok.transpose(1,2).reshape(1, tok.shape[-1], side, side)
    tok = F.interpolate(tok, size=image.shape[:2], mode="bilinear", align_corners=False)
    return tok[0].permute(1,2,0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--existing_features', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model, _, process = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
    model.eval().cuda()
    for sp in tqdm(sorted(glob.glob(os.path.join(args.existing_features, '*_s.npy'))), desc='dense readout'):
        stem = os.path.basename(sp)[:-6]
        outp = os.path.join(args.out, stem + '_f.npy')
        if os.path.exists(outp): continue
        sm = np.load(sp)
        image = cv2.imread(os.path.join(args.images, stem + '.jpg'), cv2.IMREAD_COLOR)
        if image is None:
            image = cv2.imread(os.path.join(args.images, stem + '.png'), cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(stem)
        dense = dense_tokens(model, process, image)
        parts=[]
        feat = pool_by_maps(dense, sm).cpu().numpy().astype(np.float16)
        old = np.load(sp.replace('_s.npy','_f.npy'), mmap_mode='r')
        assert feat.shape == old.shape, (stem, feat.shape, old.shape)
        assert np.isfinite(feat).all()
        np.save(outp, feat)
        np.save(os.path.join(args.out, stem + '_s.npy'), sm)


if __name__ == '__main__': main()
