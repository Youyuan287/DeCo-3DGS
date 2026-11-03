"""Query-level audit on ScanNet-10.

The per-root classification audit failed because it compared two different
class queries. Here we restore the same-query structure: for each class query c,
the incumbent is the released root set I_c = {r : argmax S0[r] == c}, and the
challenger is the same number of roots ranked by per-view mask evidence for c.
The certificate compares the two sets on the same witness views, so the class
prior (which is common to both) cancels.

GT is only used for evaluation of per-class IoU.
"""
import os, sys, json, argparse
import numpy as np

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [REL, os.path.join(REL, "ext"),
                "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"]
from eval_scannet_3d import SUBSETS, metrics


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--center", default="none", choices=["none", "image", "view"])
    args = ap.parse_args()

    p = os.path.join(args.abl, args.scene)
    d = np.load(os.path.join(p, "predictions_and_scores.npz"))
    gt = d["gt"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    S0 = d["scores_off"].astype(np.float64)
    cls = SUBSETS[19]
    cls_arr = np.array(cls)
    base_idx = S0.argmax(1)
    n_root = S0.shape[0]

    tb = np.load(os.path.join(p, "text_bank.npz"))
    one_word = l2n(tb["textbank"].astype(np.float32))
    if one_word.ndim == 3:
        one_word = one_word.reshape(-1, one_word.shape[-1])
    if one_word.shape[0] % 19 == 0 and one_word.shape[0] != 19:
        banks = one_word.reshape(one_word.shape[0] // 19, 19, -1)
    else:
        banks = one_word[None]

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    if args.center == "image":
        F = l2n(F - F.mean(0, keepdims=True))   # C1: subtract the shared mean
    elif args.center == "view":
        vv = obs["view_ids"].astype(np.int64)
        for v in np.unique(vv):
            m = vv == v
            F[m] = l2n(F[m] - F[m].mean(0, keepdims=True))  # view-conditional nuisance removal
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    order = np.argsort(reg, kind="stable")
    reg_s = reg[order]

    def slice_of(rid):
        lo = np.searchsorted(reg_s, rid, "left")
        hi = np.searchsorted(reg_s, rid, "right")
        return order[lo:hi]

    # per-root per-view sims for every class (only roots with >=2 obs)
    root_obs = {rid: slice_of(rid) for rid in range(n_root)}
    root_obs = {r: oi for r, oi in root_obs.items() if len(oi) >= 2}

    def sims_of_class(c):
        best = None
        for b in banks:
            t = b[c]
            s = F @ t
            best = s if best is None else np.maximum(best, s)
        return best

    def evidence(rid, svec):
        oi = root_obs[rid]
        v = svec[oi]
        k = min(args.topk, len(v))
        return float(np.sort(v)[-k:].mean())

    def set_evidence(S, svec):
        ev = {}
        for r in S:
            oi = root_obs.get(r)
            if oi is None:
                continue
            vv = view[oi]
            ss = svec[oi]
            for v, s in zip(vv.tolist(), ss.tolist()):
                if s > ev.get(v, -1e9):
                    ev[v] = s
        return ev

    def certify(Sa, Sb, svec, seed=0):
        ea, eb = set_evidence(Sa, svec), set_evidence(Sb, svec)
        vs = sorted(set(ea) & set(eb))
        n = len(vs)
        if n < 5:
            return 0.0
        dd = np.array([eb[v] - ea[v] for v in vs])
        k = max(2, n // 2)
        rng = np.random.default_rng(seed)
        means = np.array([dd[rng.choice(n, k, replace=False)].mean() for _ in range(300)])
        return float((means > 0).mean())

    present = [c for c in range(19) if (gt == cls[c]).any()]
    results = {"baseline": {}, "always_chal": {}, "audit": {}, "audit95": {}}
    stats = {"n_present": len(present), "audit_fire": 0, "audit95_fire": 0,
             "inc_size": 0, "chal_size": 0}
    preds = {k: {} for k in results}
    for c in present:
        svec = sims_of_class(c)
        inc = [r for r in root_obs if base_idx[r] == c]
        stats["inc_size"] += len(inc)
        if len(inc) == 0:
            for k in preds:
                preds[k][c] = np.zeros_like(gt, bool)
            continue
        cand = [(evidence(r, svec), r) for r in root_obs]
        cand.sort(reverse=True)
        chal = [r for _, r in cand[:len(inc)]]
        stats["chal_size"] += len(chal)
        p = certify(inc, chal, svec)
        if p >= 0.5:
            stats["audit_fire"] += 1
        if p >= 0.95:
            stats["audit95_fire"] += 1

        def members(S):
            m = np.zeros(n_root, bool)
            m[S] = True
            return m

        for tag, S in (("baseline", inc), ("always_chal", chal),
                       ("audit", chal if p >= 0.5 else inc),
                       ("audit95", chal if p >= 0.95 else inc)):
            gmask = np.zeros(len(gt), bool)
            gmask[members(S)[labels]] = True
            preds[tag][c] = gmask

    valid = np.isin(gt, cls)   # official protocol: GT outside the subset is ignored

    def miou(predmap):
        ious = []
        for c in present:
            g = gt == cls[c]
            pp = predmap[c] & valid
            u = np.logical_or(g, pp).sum()
            ious.append(float(np.logical_and(g, pp).sum() / u) if u else 0.0)
        return float(np.mean(ious)), ious

    # released argmax baseline in the same per-class form
    base = {c: (base_idx[labels] == c) & valid for c in present}
    out = {"scene": args.scene, "stats": stats}
    for tag, pm in (("baseline", base), ("always_chal", preds["always_chal"]),
                    ("audit", preds["audit"]), ("audit95", preds["audit95"])):
        out[tag], _ = miou(pm)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "queryaudit_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("SCENE", args.scene, json.dumps({k: round(v, 4) for k, v in out.items() if isinstance(v, float)}),
          "stats", stats, flush=True)


if __name__ == "__main__":
    main()