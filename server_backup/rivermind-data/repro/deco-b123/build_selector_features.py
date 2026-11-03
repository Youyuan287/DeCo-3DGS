"""Build per-(root, class) candidate features for the ScanNet step-2 selector.

Candidates per root = main-score top-5 classes unioned with every eligible mode's
argmax class. Features are purely evidence-based (no GT, no scene priors):
main score / rank / deficit, mode votes and mode scores, M2 reliability-weighted
advantage, and multi-view footprint statistics from the observation sidecar
(coverage of the root's Gaussians, distinct views, mean reconstructed visibility).
Labels: class == GT majority of the root (for training / evaluation only).
"""
import os, json
import numpy as np
import pandas as pd

B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
OUT = "/root/rivermind-data/repro/deco-b123/step2_selector_20261030/features"
S = ("scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 "
     "scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00").split()
CL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
os.makedirs(OUT, exist_ok=True)

for s in S:
    R = f"{B}/ablation/v2_on_on/{s}"
    E = f"{B}/{s}/export"
    z = np.load(R + "/predictions_and_scores.npz")
    gt, lab, off = z["gt"], z["labels"], z["pred_off"]
    sc = z["scores_off"].astype(np.float64)
    xyz = z["xyz"]
    cl = np.array(CL)
    cls2i = {int(c): i for i, c in enumerate(CL)}
    nr = len(sc)
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))

    maj = np.full(nr, -1, dtype=np.int64)
    for r in range(nr):
        g = order[bounds[r]:bounds[r + 1]]
        if len(g) == 0:
            continue
        gg = gt[g]
        gg = gg[np.isin(gg, CL)]
        if len(gg):
            v_, c_ = np.unique(gg, return_counts=True)
            maj[r] = int(v_[c_.argmax()])

    t = pd.read_csv(R + "/SCANNET_M3_WEIGHT_TRACE.csv",
                    usecols=["arm", "class_id", "root_id", "mode_id",
                             "valid", "eligible", "mode_score", "advantage",
                             "reliability"])
    t1 = t[(t.arm == "m1_fixed") & t.valid & t.eligible]
    t2 = t[(t.arm == "m1_m2_fixed") & t.valid & t.eligible]
    bm = t1.loc[t1.groupby(["root_id", "mode_id"]).mode_score.idxmax(),
                ["root_id", "mode_id", "class_id", "mode_score", "advantage"]].copy()
    rel2 = t2.set_index(["root_id", "mode_id", "class_id"]).reliability
    bm = bm.join(rel2.rename("rel"), on=["root_id", "mode_id", "class_id"])
    bm["rel"] = bm["rel"].fillna(0.0)
    v = bm.groupby(["root_id", "class_id"]).agg(
        n_vote=("mode_id", "size"), adv_max=("advantage", "max"),
        ms_max=("mode_score", "max"), rel_max=("rel", "max")).reset_index()
    t2i = t2[["root_id", "class_id", "advantage", "reliability"]].copy()
    t2i["wadv"] = t2i.reliability * t2i.advantage.clip(lower=0)
    wa = t2i.groupby(["root_id", "class_id"]).wadv.sum().reset_index()
    v = v.merge(wa, on=["root_id", "class_id"], how="left")
    v["wadv"] = v["wadv"].fillna(0.0)
    vd = {(int(a.root_id), int(a.class_id)): (int(a.n_vote), float(a.adv_max),
                                              float(a.ms_max), float(a.rel_max),
                                              float(a.wadv))
          for a in v.itertuples()}

    ms = json.load(open(E + "/B1.members.json"))["memberships"]
    sd = np.load(E + "/observation_semantic_sidecar.npz")
    so, sid, sw = sd["support_offsets"], sd["support_gaussian_ids"], sd["support_weights"]
    vis = sd["reconstructed_visibility"]
    views = sd["view_id"]
    mode_agm = {(int(b.root_id), int(b.mode_id)): int(b.class_id) for b in bm.itertuples()}

    fp = {}   # (root, class) -> [cov, nviews, vis, n_modes_with_fp]
    for mbl in ms:
        r = int(mbl["region_id"])
        slot = int(mbl["slot"])
        c = mode_agm.get((r, slot))
        if c is None:
            continue
        obs = mbl["observation_indices"]
        if not obs:
            continue
        arrs = []
        for o in obs:
            a, b = so[o], so[o + 1]
            m_ = sw[a:b] > 0
            if m_.any():
                arrs.append(sid[a:b][m_])
        if not arrs:
            continue
        u = np.unique(np.concatenate(arrs))
        rg = order[bounds[r]:bounds[r + 1]]
        cov = float(np.isin(u, rg).mean()) if len(rg) else 0.0
        nv = len(set(views[obs].tolist()))
        mv = float(np.mean(vis[obs]))
        cur = fp.get((r, c))
        if cur is None:
            fp[(r, c)] = [cov, nv, mv, 1]
        else:
            cur[0] = max(cur[0], cov)
            cur[1] = max(cur[1], nv)
            cur[2] = max(cur[2], mv)
            cur[3] += 1

    rows = []
    top_idx = np.argsort(-sc, 1)[:, :5]
    for r in range(nr):
        rg = order[bounds[r]:bounds[r + 1]]
        n_g = len(rg)
        mean_z = float(xyz[rg, 2].mean()) if n_g else 0.0
        s_r = sc[r]
        s_top1 = s_r[top_idx[r, 0]]
        top1 = int(cl[top_idx[r, 0]])
        mode_classes = [mode_agm[(r, md)] for (rr, md) in mode_agm if rr == r]
        cands = list(dict.fromkeys([int(cl[i]) for i in top_idx[r]] + mode_classes))
        pos = {int(c): k + 1 for k, c in enumerate(cl[np.argsort(-s_r)])}
        for c in cands:
            i = cls2i[c]
            s_c = float(s_r[i])
            nv_, adv_, msm_, relm_, wadv_ = vd.get((r, c), (0, 0.0, 0.0, 0.0, 0.0))
            fpc = fp.get((r, c), [0.0, 0, 0.0, 0])
            rows.append(dict(
                scene=s, root=r, cls=c, gt_majority=int(maj[r]), pred_root=int(off[rg[0]]) if n_g else -1,
                s=s_c, deficit=s_c - float(s_top1), rank=pos[c], is_top1=int(c == top1),
                n_vote=nv_, adv_max=adv_, ms_max=msm_, rel_max=relm_, wadv=wadv_,
                cov=fpc[0], fp_views=fpc[1], fp_vis=fpc[2], fp_modes=fpc[3],
                root_n_gauss=n_g, root_mean_z=mean_z,
                y=int(c == maj[r]) if maj[r] >= 0 else -1))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "%s_features.csv" % s), index=False)
    print("done", s, len(rows), flush=True)