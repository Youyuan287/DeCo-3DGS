#!/usr/bin/env python3
"""R2: multi-scale masked-crop global CLIP readout on fixed proposal maps.

The proposal masks and their global ids are unchanged.  Each proposal gets
three CLIP image embeddings: tight masked crop, context-expanded masked crop,
and full-image masked crop.  The normalized embeddings are fused with fixed
weights (0.5, 0.3, 0.2), preserving the global image/text CLIP space used by
the original LangSplat cache.
"""
import argparse, glob, os, cv2, numpy as np, torch
from PIL import Image
import open_clip
from tqdm import tqdm

WEIGHTS = (0.5, 0.3, 0.2)

def proposal_masks(seg):
    ids = np.unique(seg[seg >= 0])
    return ids.tolist(), [(seg == gid).any(axis=0) for gid in ids.tolist()]

def crop_variants(rgb, mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        z = np.zeros((224, 224, 3), np.uint8)
        return [z, z, z]
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0+y1)/2, (x0+x1)/2
    hh, ww = (y1-y0)/2, (x1-x0)/2
    ey0, ey1 = max(0, int(cy-1.5*hh)), min(h, int(cy+1.5*hh))
    ex0, ex1 = max(0, int(cx-1.5*ww)), min(w, int(cx+1.5*ww))
    tight = rgb[y0:y1, x0:x1].copy()
    tight[~mask[y0:y1, x0:x1]] = 0
    context = rgb[ey0:ey1, ex0:ex1].copy()
    context[~mask[ey0:ey1, ex0:ex1]] = 0
    full = rgb.copy()
    full[~mask] = 0
    return [tight, context, full]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--existing_features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--names', default='', help='comma-separated stems to regenerate')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model, _, pre = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='laion2b_s34b_b88k', precision='fp16')
    model.eval().cuda()
    files = sorted(glob.glob(os.path.join(a.existing_features, '*_s.npy')))
    wanted = set(x.strip() for x in a.names.split(',') if x.strip())
    if wanted:
        files = [p for p in files if os.path.basename(p)[:-6] in wanted]
    for sp in tqdm(files, desc='multiscale crop'):
        stem = os.path.basename(sp)[:-6]
        outp = os.path.join(a.out, stem + '_f.npy')
        if os.path.exists(outp):
            continue
        seg = np.load(sp)
        image = cv2.imread(os.path.join(a.images, stem + '.jpg'))
        if image is None: image = cv2.imread(os.path.join(a.images, stem + '.png'))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ids, masks = proposal_masks(seg)
        feats = []
        for start in range(0, len(masks), a.batch):
            batch = []
            for m in masks[start:start+a.batch]:
                for crop in crop_variants(rgb, m):
                    batch.append(pre(Image.fromarray(cv2.resize(crop, (224,224), interpolation=cv2.INTER_LINEAR))))
            x = torch.stack(batch).cuda().half()
            z = model.encode_image(x).float()
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            z = z.view(-1, 3, z.shape[-1])
            fused = sum(w*z[:,i] for i,w in enumerate(WEIGHTS))
            fused = fused / fused.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feats.append(fused.cpu())
        compact = torch.cat(feats).numpy().astype(np.float16)
        old = np.load(sp.replace('_s.npy','_f.npy'), mmap_mode='r')
        f = np.zeros(old.shape, np.float16)
        f[np.asarray(ids, dtype=np.int64)] = compact
        assert f.shape == old.shape, (stem, f.shape, old.shape)
        np.save(outp, f); np.save(os.path.join(a.out, stem+'_s.npy'), seg)
    print('done', len(files))

if __name__ == '__main__': main()
