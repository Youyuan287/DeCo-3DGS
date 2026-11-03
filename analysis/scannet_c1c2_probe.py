#!/usr/bin/env python3
"""Do C1 (per-view image centering) and C2 (template LSE) help on the OpenSeg field?

Same partition, same masks, same aggregation; only the two calibration switches move.
C1 centres each view's mask features over that view's masks before aggregation.
C2 replaces the mean-over-templates text score with a log-sum-exp over templates.

Reports per-scene mIoU for the four combinations on each encoder, so the CLIP arm can be
checked against the known +9.8 and the OpenSeg arm answers the question directly.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

NYU40 = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair", 6: "sofa", 7: "table",
    8: "door", 9: "window", 10: "bookshelf", 11: "picture", 12: "counter", 14: "desk",
    16: "curtain", 24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub",
}
CLASSES = sorted(NYU40)
PROMPTS = ("{}", "a photo of a {}", "the {}", "a {} in a room")
TAU = 0.02


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def text_templates(model_name, pretrained, names):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    out = np.zeros((len(names), len(PROMPTS), 0), np.float32)
    rows = []
    with torch.no_grad():
        for n in names:
            t = tok([p.format(n) for p in PROMPTS])
            if torch.cuda.is_available():
                t = t.cuda()
            f = model.encode_text(t).float()
            rows.append(l2n(f.cpu().numpy()))
    return np.stack(rows)                   # (K, P, D), raw per-template embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--os-dir", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from plyfile import PlyData
    from scipy.spatial import cKDTree

    z = np.load(args.obs)
    sp, view, mask, cnt = z["sp"], z["view"], z["mask"], z["count"]
    names = [str(n) for n in z["names"]]
    n_sp = int(z["n_sp"])

    vtx = PlyData.read(args.ply)["vertex"]
    g = np.stack([vtx["x"], vtx["y"], vtx["z"]], 1).astype(np.float64)
    vtx = PlyData.read(args.gt)["vertex"]
    G = np.stack([vtx["x"], vtx["y"], vtx["z"]], 1).astype(np.float64)
    GL = np.asarray(vtx["label"]).astype(np.int64)
    gcls = GL[cKDTree(G).query(g, workers=8)[1]]
    root = torch.load(args.nag, map_location="cpu")["nag"][-1].numpy().astype(np.int64)

    key = sp.astype(np.int64) * 100000 + view
    order = np.argsort(key, kind="stable")
    key, sp_s, view_s, mask_s, cnt_s = key[order], sp[order], view[order], mask[order], cnt[order]
    b = np.flatnonzero(np.diff(key)) + 1
    starts, ends = np.r_[0, b], np.r_[b, key.size]

    banks = {}
    banks["clip"] = text_templates("ViT-B-16", "laion2b_s34b_b88k", [NYU40[c] for c in CLASSES])
    src = "/root/rivermind-data/models/clip/ViT-L-14-336-openai.bin"
    banks["openseg"] = text_templates("ViT-L-14-336", src if os.path.exists(src) else "openai",
                                      [NYU40[c] for c in CLASSES])

    results = {}
    for ch, suffix, d in (("clip", "_f.npy", args.clip_dir), ("openseg", "_f_openseg.npy", args.os_dir)):
        feats = [l2n(np.load(os.path.join(d, nm + suffix)).astype(np.float32)) for nm in names]
        for c1 in (0, 1):
            dim = feats[0].shape[1]
            sp_size = np.bincount(root, minlength=n_sp).astype(np.float64)
            vecs = [[] for _ in range(n_sp)]
            for a, e in zip(starts, ends):
                s_ = int(sp_s[a])
                v = int(view_s[a])
                m, c = mask_s[a:e], cnt_s[a:e].astype(np.float64)
                ok = m < feats[v].shape[0]
                if not ok.any():
                    continue
                m, c = m[ok], c[ok]
                q = c / c.sum()
                q[q < 0.3] = 0.0          # merge_proj RATIO_THRESHOLD, no renormalisation
                if q.sum() <= 0:
                    continue
                f = l2n(q @ feats[v][m])
                vis = min(c.sum() / max(sp_size[s_], 1.0), 1.0)   # visibility_portion
                vecs[s_].append(f * vis)
            acc = np.zeros((n_sp, dim), np.float64)
            for s_, L in enumerate(vecs):
                if not L:
                    continue
                A = np.stack(L)
                if A.shape[0] == 1:
                    acc[s_] = l2n(A[0])
                    continue
                # ROFA: drop observations below mu - tau*sigma of mean cosine
                G = A @ A.T
                ms = (G.sum(1) - 1.0) / (A.shape[0] - 1.0)
                keep = ms > (ms.mean() - 2.0 * ms.std())
                if keep.sum() == 0:
                    keep[int(ms.argmax())] = True
                acc[s_] = l2n(A[keep].mean(0))
            R = l2n(acc)
            if c1:
                # C1 (deco_ops.center_np, renorm=0): subtract the mean over regions
                R = R - R.mean(0, keepdims=True)
            T = banks[ch]                                 # (K, P, D) raw templates
            Tbar = l2n(T.mean(1))                         # eval_scannet_3d.text_features
            Tc = l2n(T - T.mean(axis=0, keepdims=True))   # C2a: centre over the class set
            for c2 in (0, 1):
                if c2:
                    Sc = np.einsum("rd,kpd->rkp", R, Tc)
                    mx = Sc.max(2, keepdims=True)
                    scc = (TAU * np.log(np.exp((Sc - mx) / TAU).mean(2, keepdims=True)) + mx)[..., 0]
                else:
                    scc = R @ Tbar.T
                pred = np.array(CLASSES)[scc.argmax(1)][root]
                ious = []
                for c in CLASSES:
                    gt = gcls == c
                    if gt.sum() == 0:
                        continue
                    pp = pred == c
                    u = np.logical_or(gt, pp).sum()
                    ious.append(float(np.logical_and(gt, pp).sum() / u) if u else 0.0)
                results[(ch, c1, c2)] = float(np.mean(ious))

    out = dict(scene=args.scene, results={"%s_c1%d_c2%d" % (k[0], k[1], k[2]): v
                                          for k, v in results.items()})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "c1c2_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()