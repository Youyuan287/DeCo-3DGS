#!/usr/bin/env python3
"""Apply our decision audit to the ScanNet field, with the encoder swapped in place.

Same partition (the released CLIP roots), same masks, same views, same certificate.
Only the feature channel changes: CLIP ViT-B-16 vs OpenSeg-768 (ViT-L/14-336 text tower).

The region/witness linkage comes from the trace dump (encoder-independent):
  scannet_mask_mixture_probe.py -> <scene>_L3_obs.npz  (root, view, mask, count)

Region feature = normalised weighted mean of the masks covering the root across views.
Witness token = the dominant mask's feature in that view.
Certificate    = half-sample majority on delta_v = s_v(c) - s_v(L0), revise if max z > tau.

This is an offline reproduction, not the certified harness: it exists to answer whether a
better encoder makes the witnesses usable, on one scene, before spending hours re-merging.
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
SUBSETS19 = CLASSES
PROMPTS = ("{}", "a photo of a {}", "the {}", "a {} in a room")


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def bank(model_name, pretrained, names):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    emb = []
    with torch.no_grad():
        for n in names:
            t = tok([p.format(n) for p in PROMPTS])
            if torch.cuda.is_available():
                t = t.cuda()
            f = model.encode_text(t).float()
            f = f / f.norm(dim=-1, keepdim=True)
            emb.append(f.mean(0))
    B = l2n(torch.stack(emb).cpu().numpy())
    return l2n(B - B.mean(0, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--clip-dir", required=True)
    ap.add_argument("--os-dir", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--channel", choices=["clip", "openseg"], required=True)
    ap.add_argument("--gates", default="0,0.2,0.35,0.5,0.65")
    ap.add_argument("--taus", default="0,1.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from plyfile import PlyData
    from scipy.spatial import cKDTree

    z = np.load(args.obs)
    sp, view, mask, cnt = z["sp"], z["view"], z["mask"], z["count"]
    names = [str(n) for n in z["names"]]
    n_sp = int(z["n_sp"])

    d = args.clip_dir if args.channel == "clip" else args.os_dir
    suffix = "_f.npy" if args.channel == "clip" else "_f_openseg.npy"
    feats = []
    for nm in names:
        f = np.load(os.path.join(d, nm + suffix)).astype(np.float32)
        feats.append(l2n(f))

    # region feature and dominant-mask witness, one pass over (root, view) groups
    key = sp.astype(np.int64) * 100000 + view
    order = np.argsort(key, kind="stable")
    key, sp_s, view_s, mask_s, cnt_s = key[order], sp[order], view[order], mask[order], cnt[order]
    b = np.flatnonzero(np.diff(key)) + 1
    starts, ends = np.r_[0, b], np.r_[b, key.size]

    dim = feats[0].shape[1]
    acc = np.zeros((n_sp, dim), np.float64)
    wsum = np.zeros(n_sp)
    wit = {}                      # root -> list of (view, feature)
    for a, e in zip(starts, ends):
        s = int(sp_s[a])
        v = int(view_s[a])
        m = mask_s[a:e]
        c = cnt_s[a:e].astype(np.float64)
        ok = m < feats[v].shape[0]
        if not ok.any():
            continue
        m, c = m[ok], c[ok]
        share = c / c.sum()
        acc[s] += share @ feats[v][m]
        wsum[s] += 1.0
        j = int(np.argmax(c))
        wit.setdefault(s, []).append((v, feats[v][m[j]]))

    R = np.zeros_like(acc)
    has = wsum > 0
    R[has] = l2n(acc[has])
    R = R - R[has].mean(0, keepdims=True)
    bank_c = bank("ViT-B-16", "laion2b_s34b_b88k", [NYU40[c] for c in CLASSES]) \
        if args.channel == "clip" else None
    if bank_c is None:
        src = "/root/rivermind-data/models/clip/ViT-L-14-336-openai.bin"
        bank_c = bank("ViT-L-14-336", src if os.path.exists(src) else "openai",
                      [NYU40[c] for c in CLASSES])
    S = R @ bank_c.T                       # (n_sp, 19)

    # per-root witness statistics, computed once
    base = S.argmax(1).copy()
    agree, bestc, bestz, zs = {}, {}, {}, []
    for r in range(n_sp):
        w = wit.get(r)
        if not w or len(w) < 4:
            continue
        Z = np.stack([f for _, f in w]) @ bank_c.T     # (n_views, 19)
        agree[r] = float((Z.argmax(1) == base[r]).mean())
        L0 = int(base[r])
        bc, bz = -1, -np.inf
        for c in range(len(CLASSES)):
            if c == L0:
                continue
            dlt = Z[:, c] - Z[:, L0]
            sd = dlt.std(ddof=0)
            if sd < 1e-9:
                continue
            zc = dlt.mean() / (sd / np.sqrt(dlt.size / 2.0))
            if zc > bz:
                bz, bc = zc, c
        bestc[r], bestz[r] = bc, bz
        zs.append(float(bz))

    # evaluate per-Gaussian, matching metrics()
    vtx = PlyData.read(args.ply)["vertex"]
    g = np.stack([vtx["x"], vtx["y"], vtx["z"]], 1).astype(np.float64)
    vtx = PlyData.read(args.gt)["vertex"]
    G = np.stack([vtx["x"], vtx["y"], vtx["z"]], 1).astype(np.float64)
    GL = np.asarray(vtx["label"]).astype(np.int64)
    gcls = GL[cKDTree(G).query(g, workers=8)[1]]

    root = torch.load(args.nag, map_location="cpu")["nag"][-1].numpy().astype(np.int64)
    assert root.shape[0] == g.shape[0], (root.shape, g.shape)

    def miou(p_):
        pr = np.array(CLASSES)[np.clip(p_, 0, len(CLASSES) - 1)][root]
        ious, per = [], {}
        for c in CLASSES:
            gt = gcls == c
            if gt.sum() == 0:
                continue
            pp = pr == c
            u = np.logical_or(gt, pp).sum()
            val = float(np.logical_and(gt, pp).sum() / u) if u else 0.0
            ious.append(val)
            per[NYU40[c]] = val
        return float(np.mean(ious)), per

    # witness informativeness: does the per-view evidence rank classes better than the region?
    _cnt = np.bincount(root * 41 + gcls, minlength=n_sp * 41).reshape(n_sp, 41)
    gt_dom = _cnt.argmax(1)
    idx_of = {c: i for i, c in enumerate(CLASSES)}
    ok = np.array([gt_dom[r] in idx_of for r in range(n_sp)])
    gt_idx = np.array([idx_of.get(int(gt_dom[r]), -1) for r in range(n_sp)])
    acc_base = float((base[ok] == gt_idx[ok]).mean())
    m_acc = np.full(n_sp, -1, np.int64)
    v_acc = np.full(n_sp, -1, np.int64)
    for r, w in wit.items():
        if len(w) < 4:
            continue
        Z = np.stack([f for _, f in w]) @ bank_c.T
        m_acc[r] = int(Z.mean(0).argmax())
        v_acc[r] = int(np.bincount(Z.argmax(1), minlength=len(CLASSES)).argmax())
    sel = ok & (m_acc >= 0)
    acc_wit_mean = float((m_acc[sel] == gt_idx[sel]).mean())
    acc_wit_vote = float((v_acc[sel] == gt_idx[sel]).mean())
    # GT-free reliability: do the views agree with the region's own label?
    ag = []
    for r, w in wit.items():
        if len(w) < 4:
            continue
        Z = np.stack([f for _, f in w]) @ bank_c.T
        ag.append(float((Z.argmax(1) == base[r]).mean()))
    agree_scalar = float(np.mean(ag)) if ag else 0.0

    m0, pc0 = miou(base)
    gates = [float(x) for x in args.gates.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    curve = []
    for g in gates:
        for t in taus:
            pred = base.copy()
            rev = 0
            for r in list(agree):
                if agree[r] < g:
                    continue
                if bestc[r] >= 0 and bestz[r] > t:
                    pred[r] = bestc[r]
                    rev += 1
            m1, pc1 = miou(pred)
            curve.append(dict(gate=g, tau=t, revised=rev,
                              frac_revised=rev / max(len(agree), 1),
                              mIoU=m1, delta=m1 - m0))
    out = dict(scene=args.scene, channel=args.channel, n_roots=int(has.sum()),
               n_with_witness=len(agree), mIoU_base=m0,
               mean_z=float(np.mean(zs)) if zs else 0.0,
               agree_gt_free=float(np.mean(list(agree.values()))) if agree else 0.0,
               acc_base=acc_base, acc_witness_mean=acc_wit_mean,
               acc_witness_vote=acc_wit_vote, curve=curve,
               per_class_base=pc0)
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "gate_%s_%s.json" % (args.scene, args.channel))
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("scene", "channel", "n_with_witness",
                                         "mIoU_base", "agree_gt_free", "acc_base",
                                         "acc_witness_mean")}))
    for c in curve:
        print("  gate=%.2f tau=%.2f  rev=%3d (%4.0f%%)  mIoU=%.4f  d=%+.4f"
              % (c["gate"], c["tau"], c["revised"], 100 * c["frac_revised"],
                 c["mIoU"], c["delta"]))


if __name__ == "__main__":
    main()
