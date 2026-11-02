"""Local-contrast variants (B0-B5) on the ReLaGS ScanNet field.

Keeps candidates / CLIP features / views / hierarchy / fusion fixed and changes
ONLY how the per-view semantic evidence is computed. Every variant uses the same
top-3 view aggregation, the same per-class z-score of released, the same beta.

B0 E0   : s_v(c,t)                                  (no contrast; baseline zR+E0)
B1 VC   : s_v - alpha * mean_view s_v                (global per-view offset = VCLS)
B2 RAND : s_v - mean_{n in random K} s_v(n)          (same K/weights as B3)
B3 KNN  : s_v - mean_{n in spatial-kNN(c) & vis(v)} s_v(n)
B4 SIB  : s_v - mean_{n in siblings(c) & vis(v)}    s_v(n)
B5 HARD : s_v - tau*logmeanexp_{n in kNN & vis(v)}(s_v(n)/tau)
B6 PAIR : paired cross-view statistic on (c, n) with a shared co-visibility set
          delta_v = s_v(c) - s_v(n),  med = median_v delta_v,  q = mean_v [delta_v>0]
          A_c(t) = mean_n  med_n(t) * (2 q_n(t) - 1)

B6 is the one that tests the actual mechanism claim. B2-B5 subtract a per-view
neighbourhood mean, so an additive per-view offset shared by c and its referents
cancels; B6 compares the SAME two regions across views, so a view offset that
differs between them does NOT cancel, and the sign-consistency q is what carries
"is this advantage stable or a single-view accident".

Co-visibility is required: a referent enters only if it is observed in the SAME
view. If a candidate has no co-visible referent in a view, that view falls back
to s_v (no contrast) and the share is reported as `cov`.

Per variant: mIoU, mAcc, root selection accuracy, margin between true class and
strongest wrong class (mean/median).
"""
import os, sys, glob, json
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import load_gt, load_gaussians, metrics, NYU40, SUBSETS, text_features
import score_scannet as published


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def lme(x, tau):
    m = x.max()
    return m + tau * np.log(np.exp((x - m) / tau).mean())


def main():
    ap = ArgumentParser()
    for k in ["scene", "model", "input", "gt_root", "released_npz", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0, help="tau = gamma * sd(S)")
    args = ap.parse_args()

    gt_path = glob.glob(os.path.join(args.gt_root, "*_vh_clean_2.labels.ply"))
    gtxyz, gtlabel = load_gt(gt_path[0])
    xyz, opacity, keep, ply = load_gaussians(args.model, 0)
    _, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    levels = [t.numpy().astype(np.int64) for t in nag["nag"]]
    labels_root = levels[-1]
    n_root = int(labels_root.max()) + 1
    assert len(labels_root) == len(xyz), (len(labels_root), len(xyz))

    # levels[i] assigns a region id to every Gaussian; a root's parent is the
    # level-(L-2) region of any Gaussian in it (the hierarchy is nested).
    parent = np.full(n_root, -1, np.int64)
    if len(levels) >= 2:
        order = np.argsort(labels_root, kind="stable")
        lr_sorted = labels_root[order]
        starts = np.r_[0, np.flatnonzero(np.diff(lr_sorted)) + 1]
        first_g = order[starts]
        parent[lr_sorted[starts]] = levels[-2][first_g]
    by_parent = {}
    for r in range(n_root):
        if parent[r] >= 0:
            by_parent.setdefault(int(parent[r]), []).append(r)
    sib = [np.array([q for q in by_parent.get(int(parent[r]), []) if q != r], np.int64)
           for r in range(n_root)]

    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    T = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                      prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    T = l2n(T)
    R = np.load(args.released_npz)["scores_off"].astype(np.float64)
    n_cls = len(cls)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    S = F @ T.T

    key = reg * 100000 + view
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), n_cls))
    np.add.at(sums, inv, S)
    cnt = np.bincount(inv).astype(np.float64)
    sv = sums / cnt[:, None]
    vr = (uk // 100000).astype(np.int64)
    vv = (uk % 100000).astype(np.int64)

    # per root: list of (view, index-into-sv)
    rv = {}
    for i in range(len(sv)):
        rv.setdefault(int(vr[i]), []).append((int(vv[i]), i))
    # per view: set of roots observed
    vis = {}
    for i in range(len(sv)):
        vis.setdefault(int(vv[i]), set()).add(int(vr[i]))

    view_mean = {}
    for i in range(len(sv)):
        view_mean.setdefault(int(vv[i]), []).append(i)
    view_mean = {v: sv[np.array(ix)].mean(0) for v, ix in view_mean.items()}
    rv_map = {r: {v: j for v, j in rv_r} for r, rv_r in rv.items()}

    cen = np.zeros((n_root, 3))
    n_g = np.bincount(labels_root, minlength=n_root).astype(np.float64)
    for k in range(3):
        cen[:, k] = np.bincount(labels_root, weights=xyz[:, k],
                                minlength=n_root) / np.maximum(n_g, 1)
    kk = min(args.knn + 1, n_root)
    _, nn = cKDTree(cen).query(cen, k=kk, workers=8)
    nn = np.atleast_2d(nn)
    knn = [nn[r][nn[r] != r][: args.knn] for r in range(n_root)]

    rng = np.random.default_rng(0)
    rand_ref = {}
    for r in range(n_root):
        q = rng.choice(n_root - 1, size=min(args.knn, n_root - 1), replace=False)
        rand_ref[r] = np.where(q >= r, q + 1, q)

    tau = max(args.gamma * float(S.std()), 1e-6)

    def refs(kind, r):
        if kind == "knn" or kind == "hard":
            return knn[r]
        if kind == "sib":
            return sib[r]
        if kind == "rand":
            return rand_ref[r]
        return np.array([], np.int64)

    def evidence(kind):
        E = np.zeros((n_root, n_cls))
        hit = tot = 0
        for r, rv_r in rv.items():
            per_view = np.zeros((len(rv_r), n_cls))
            for t, (v, j) in enumerate(rv_r):
                tot += 1
                if kind == "e0":
                    per_view[t] = sv[j]
                    continue
                if kind == "vc":
                    per_view[t] = sv[j] - args.alpha * view_mean[v]
                    continue
                visr = vis[v]
                idxs = [rv_map[rr].get(v) for rr in refs(kind, r) if rr in visr]
                idxs = [q for q in idxs if q is not None]
                if not idxs:
                    per_view[t] = sv[j]
                    continue
                hit += 1
                M = sv[np.array(idxs)]
                if kind == "hard":
                    per_view[t] = sv[j] - np.array([lme(M[:, c], tau) for c in range(n_cls)])
                else:
                    per_view[t] = sv[j] - M.mean(0)
            k = min(args.topk, len(rv_r))
            per_view = np.partition(per_view, len(rv_r) - k, axis=0)[-k:]
            E[r] = per_view.mean(0)
        return E, hit / max(tot, 1)

    def pair_evidence(kind):
        """Paired cross-view statistic: same (c, n) evaluated on shared views."""
        E = np.zeros((n_root, n_cls))
        hit = tot = 0
        vcount = []
        for r, rv_r in rv.items():
            tot += 1
            acc = np.zeros(n_cls)
            wsum = 0.0
            for nb in refs(kind, r):
                nb = int(nb)
                if nb not in rv_map:
                    continue
                common = [v for v in rv_map[r] if v in rv_map[nb]]
                if len(common) < 2:
                    continue
                jc = np.array([rv_map[r][v] for v in common])
                jn = np.array([rv_map[nb][v] for v in common])
                D = sv[jc] - sv[jn]
                acc += np.median(D, axis=0) * (2.0 * (D > 0).mean(0) - 1.0)
                wsum += 1
                vcount.append(len(common))
            if wsum:
                E[r] = acc / wsum
                hit += 1
        return E, hit / max(tot, 1), vcount

    zR = (R - R.mean(0)) / (R.std(0) + 1e-8)

    def z(x):
        """Per-class standardization over ALL roots, matching the certified
        pipeline: roots without observations carry 0 and are therefore pushed
        far below the mean, so they cannot be selected on evidence alone.
        Keeping this convention is what makes E0 reproduce the known 48.77.
        """
        out = (x - x.mean(0)) / (x.std(0) + 1e-8)
        return np.where(np.isfinite(out), out, 0.0)

    lr_keep, gk = labels_root[keep], gt[keep]
    # majority GT class per root (gk < 1000 by NYU40 convention)
    hz = np.bincount(lr_keep.astype(np.int64) * 1000 + gk,
                     minlength=n_root * 1000).reshape(n_root, 1000)
    gt_major = np.where(hz.sum(1) > 0, hz.argmax(1), -1).astype(np.int64)
    gt_idx = np.full(n_root, -1, np.int64)
    for r in range(n_root):
        if gt_major[r] >= 0:
            q = np.flatnonzero(cls == gt_major[r])
            gt_idx[r] = int(q[0]) if len(q) else -1
    ev_r = np.flatnonzero(gt_idx >= 0)

    def evaluate(sc, tag, cov=None):
        pred = np.array(cls)[sc.argmax(1)][labels_root]
        m = metrics(gk, pred[keep], cls)
        am = sc.argmax(1)
        racc = 100 * float((am[ev_r] == gt_idx[ev_r]).mean())
        mg = []
        for r in ev_r:
            row = sc[r].copy()
            tv = row[gt_idx[r]]
            row[gt_idx[r]] = -1e18
            mg.append(tv - row.max())
        mg = np.array(mg)
        d = dict(tag=tag, miou=round(100 * m["mIoU"], 2), macc=round(100 * m["mAcc"], 2),
                 racc=round(racc, 2), mg_mean=float(np.round(mg.mean(), 5)),
                 mg_med=float(np.round(np.median(mg), 5)))
        if cov is not None:
            d["cov"] = round(100 * cov, 1)
        return d

    out = [evaluate(R, "raw"), evaluate(zR, "zR")]
    for kind, tag in [("e0", "E0"), ("vc", "VC"), ("rand", "Erand"),
                      ("knn", "Eknn"), ("sib", "Esib"), ("hard", "Ehard")]:
        E, cov = evidence(kind)
        out.append(evaluate(zR + args.beta * z(E), "zR+" + tag,
                            None if kind in ("e0", "vc") else cov))

    for kind, tag in [("knn", "Epair_knn"), ("sib", "Epair_sib")]:
        E, cov, vcount = pair_evidence(kind)
        d = evaluate(zR + args.beta * z(E), "zR+" + tag, cov)
        if vcount:
            d["views_med"] = int(np.median(vcount))
            d["views_min"] = int(np.min(vcount))
        out.append(d)

    summary = dict(scene=args.scene, n_root=n_root, n_cls=n_cls,
                   knn_deg=round(float(np.mean([len(k) for k in knn])), 2),
                   sib_deg=round(float(np.mean([len(s) for s in sib])), 2),
                   sib_zero=round(float(np.mean([len(s) == 0 for s in sib])), 3),
                   tau=round(tau, 5), variants=out)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s.json" % args.scene), "w") as f:
        json.dump(summary, f, indent=2)
    print("RESULT", args.scene, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()