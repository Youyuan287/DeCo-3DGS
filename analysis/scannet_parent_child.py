#!/usr/bin/env python3
"""Does the hierarchy already reveal parts that the root cluster cannot express?

Every root cluster is a union of finer clusters, and their text features are stored in
`sai_nag.pt` (nag_feat = levels 1,2,3). This asks, with no training and no GPU forward:

  Q1  Do the children of an impure root cluster disagree more than the children of a
      pure one? -> AUC of child-score dispersion for predicting parent impurity.
  Q2  When a root cluster genuinely contains two classes, does the minority class show
      up among its children's assignments? -> that is the "part detector" hypothesis.

Also runs a permutation control (children reassigned to random parents) so a high AUC
cannot come from the metric itself.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

NYU40 = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop",
}
SUBSETS19 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
PROMPTS = ("{}", "a photo of a {}", "the {}", "a {} in a room")
NAMES = [NYU40[c] for c in SUBSETS19]


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def auc(y, score):
    """Probability that a positive outranks a negative (ties count 0.5)."""
    y = np.asarray(y, bool)
    s = np.asarray(score, float)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    uniq, inv = np.unique(s, return_inverse=True)
    for u in np.flatnonzero(np.bincount(inv) > 1):
        m = inv == u
        ranks[m] = ranks[m].mean()
    n1 = y.sum()
    n0 = (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bank_from_clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tok = open_clip.get_tokenizer("ViT-B-16")

    def emb(words):
        with torch.no_grad():
            f = model.encode_text(tok(list(words)))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    tpl = np.stack([emb([p.format(n) for n in NAMES]) for p in PROMPTS])
    tpl_c = l2n(tpl - tpl.mean(1, keepdims=True))
    b = l2n(tpl_c.mean(0))
    return l2n(b - b.mean(0, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--child-level", type=int, default=0,
                    help="index into nag_feat for the child features (0=L1, 1=L2)")
    args = ap.parse_args()

    from plyfile import PlyData
    from scipy.spatial import cKDTree

    d = torch.load(args.nag, map_location="cpu")
    nag = [t.numpy().astype(np.int64).reshape(-1) for t in d["nag"]]
    feats = [f.numpy().astype(np.float32) for f in d["nag_feat"]]
    parent = nag[3]
    child_lab_full = nag[args.child_level + 1]   # nag_feat[0] is level 1 -> nag[1]
    child_feat = feats[args.child_level]
    parent_feat = feats[2]
    n_child = child_feat.shape[0]
    assert child_lab_full.max() + 1 == n_child, (child_lab_full.max() + 1, n_child)

    v = PlyData.read(args.ply)["vertex"]
    g = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    v = PlyData.read(args.gt)["vertex"]
    gt = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    gl = np.asarray(v["label"]).astype(np.int64)
    gcls = gl[cKDTree(gt).query(g, workers=8)[1]]

    # nesting: one parent per child
    pair = np.unique(np.stack([child_lab_full, parent], 1), axis=0)
    cnt = np.bincount(pair[:, 0], minlength=n_child)
    assert cnt.max() == 1, "child cluster spans several parents"

    n_par = parent.max() + 1
    bank = bank_from_clip()
    Xc = child_feat - child_feat.mean(0, keepdims=True)
    S = Xc @ bank.T                                   # (n_child, 19)
    Pc = parent_feat - parent_feat.mean(0, keepdims=True)
    Sp = Pc @ bank.T                                  # (n_par, 19)
    clsz = np.bincount(child_lab_full, minlength=n_child).astype(float)

    rows = []
    for p in range(n_par):
        sel = np.flatnonzero(pair[:, 1] == p)
        if sel.size < 2:
            continue
        kids = pair[sel, 0]
        Sc = S[kids]
        w = clsz[kids]
        a = Sc.argmax(1)
        pa = int(Sp[p].argmax())
        share = float(np.average(a == pa, weights=w))
        share_raw = float((a == pa).mean())
        ncls = int(len(np.unique(a)))
        col = Sc[:, pa]
        std = float(np.average(np.abs(col - np.average(col, weights=w)), weights=w))
        rng = float(col.max() - col.min())
        # entropy over the children's class distribution (mass weighted)
        h = 0.0
        for c in np.unique(a):
            q = float(w[a == c].sum() / w.sum())
            h -= q * np.log(q)
        # GT: parent purity and second class
        gp = gcls[parent == p]
        cnts = np.bincount(gp, minlength=41)
        pur = float(cnts.max() / max(cnts.sum(), 1))
        dom = int(cnts.argmax())
        tmp = cnts.copy()
        tmp[dom] = 0
        sec = int(tmp.argmax())
        sec_mass = float(tmp.max() / max(cnts.sum(), 1))
        # is the second GT class present among children, and where does it rank?
        if sec in SUBSETS19:
            si = SUBSETS19.index(sec)
            sec_in_children = float(w[a == si].sum() / w.sum())
            mw = np.average(Sc, axis=0, weights=w)
            rank = int((mw[si] > mw).sum()) + 1
            mx = Sc.max(0)
            max_rank = int((mx[si] > mx).sum()) + 1
        else:
            sec_in_children = float("nan")
            rank = max_rank = -1
        rows.append(dict(scene=args.scene, parent=p, n_child=int(kids.size),
                         w_child=float(w.sum()), purity=pur, gt_dom=dom, gt_sec=sec,
                         sec_mass=sec_mass, share=share, share_raw=share_raw,
                         ncls=ncls, disp_std=std, disp_range=rng, entropy=h,
                         sec_share_children=sec_in_children,
                         sec_mean_rank=rank, sec_max_rank=max_rank))

    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "pc_%s_L%d.csv" % (args.scene, args.child_level))
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    pur = np.array([r["purity"] for r in rows])
    imp = pur < 0.9

    # permutation control: identical metric, children reassigned to random parents
    rng = np.random.default_rng(0)
    par_ids = np.array([r["parent"] for r in rows])
    kid_lists = [pair[np.flatnonzero(pair[:, 1] == g), 0] for g in par_ids]
    ctrl = []
    for _ in range(10):
        perm = rng.permutation(len(kid_lists))
        sh, nc = [], []
        for i, src in enumerate(perm):
            kids = kid_lists[src]
            Sc = S[kids]
            w2 = clsz[kids]
            a2 = Sc.argmax(1)
            pa = int(Sp[par_ids[i]].argmax())
            sh.append(float(np.average(a2 == pa, weights=w2)))
            nc.append(int(len(np.unique(a2))))
        ctrl.append((auc(imp, [-x for x in sh]), auc(imp, nc)))

    print(json.dumps(dict(scene=args.scene, level=args.child_level, n_parent=len(rows),
                          impure=int(imp.sum()),
                          auc_share=auc(imp, [-r["share"] for r in rows]),
                          auc_ncls=auc(imp, [r["ncls"] for r in rows]),
                          auc_std=auc(imp, [r["disp_std"] for r in rows]),
                          auc_entropy=auc(imp, [r["entropy"] for r in rows]),
                          ctrl_auc_share=float(np.mean([c[0] for c in ctrl])),
                          ctrl_auc_ncls=float(np.mean([c[1] for c in ctrl])),
                          pure_share=float(np.mean([r["share"] for r in rows if r["purity"] >= 0.9])),
                          imp_share=float(np.mean([r["share"] for r in rows if r["purity"] < 0.9]))
                          if imp.any() else 0.0)))

    two = [r for r in rows if r["sec_mass"] >= 0.2 and r["sec_mean_rank"] > 0]
    if two:
        print(json.dumps(dict(scene=args.scene, level=args.child_level,
                              n_two_class=len(two),
                              sec_in_top3=float(np.mean([r["sec_mean_rank"] <= 3 for r in two])),
                              sec_in_top1=float(np.mean([r["sec_mean_rank"] == 1 for r in two])),
                              sec_appears_as_child=float(np.mean([r["sec_share_children"] > 0 for r in two])),
                              sec_child_share=float(np.mean([r["sec_share_children"] for r in two])))))


if __name__ == "__main__":
    main()