"""Audit on the DeCo C1+C2 (on) arm of ScanNet-10.

Incumbent: argmax of the on-arm per-root scores (V2 modules off), which is the
certified 50.18 macro arm. Witnesses: per-view observations (unchanged by C1/C2).
Same certificate as scannet_audit_eval.py, with a class-prior centered variant.
GT comes from the npz; no GT is used by the method.
"""
import os, sys, json, argparse
import numpy as np

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [REL, os.path.join(REL, "ext"),
                "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"]
from eval_scannet_3d import SUBSETS, metrics


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def macro_iou(gt, pred, cls):
    return metrics(gt, pred, cls)["mIoU"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(os.path.join(args.abl, args.scene, "predictions_and_scores.npz"))
    gt = d["gt"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    S0 = d["scores_off"]                     # (n_root, 19), C1+C2 on, V2 off
    cls = SUBSETS[19]
    cls_arr = np.array(cls)
    base_idx = S0.argmax(1)
    base_pred = cls_arr[base_idx][labels]
    m_base = macro_iou(gt, base_pred, cls)

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    order = np.argsort(reg, kind="stable")
    reg_s = reg[order]
    F_s = F[order]

    one_word = None
    tb = np.load(os.path.join(args.abl, args.scene, "text_bank.npz"))
    one_word = l2n(tb["textbank"].astype(np.float32))   # (10,19,512) or (19,512)
    if one_word.ndim == 3:
        one_word = l2n(one_word.reshape(-1, one_word.shape[-1]))
    # textbank may be multi-scale; reduce to a per-class score by max over banks
    if one_word.shape[0] % 19 == 0 and one_word.shape[0] != 19:
        banks = one_word.reshape(one_word.shape[0] // 19, 19, -1)
    else:
        banks = one_word[None, :, :]

    def scores_of(X):
        best = None
        for b in banks:
            s = X @ b.T
            best = s if best is None else np.maximum(best, s)
        return best

    class_mean = scores_of(F_s).mean(0)

    tags = ("audit_majority", "audit95", "cal_majority", "cal95")
    pred_idx = {t: base_idx.copy() for t in tags}
    n_fire = {t: 0 for t in tags}
    n_roots = S0.shape[0]
    srt = np.sort(S0, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    cal = {}
    for rid in range(n_roots):
        lo = np.searchsorted(reg_s, rid, "left")
        hi = np.searchsorted(reg_s, rid, "right")
        if hi - lo < 3:
            continue
        X = F_s[order[lo:hi]]
        S = scores_of(X)
        l0 = int(base_idx[rid])
        D = S - S[:, [l0]]
        mu = D.mean(0); sd = D.std(0); k = max(2, len(X) // 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = mu / (sd / np.sqrt(k))
        z = np.where(np.isfinite(z), z, 0.0); z[l0] = -1e9
        c = int(np.argmax(z))
        if z[c] > 0:
            pred_idx["audit_majority"][rid] = c
            if c != l0: n_fire["audit_majority"] += 1
        if z[c] > 1.64:
            pred_idx["audit95"][rid] = c
            if c != l0: n_fire["audit95"] += 1

        Sc = S - class_mean[None, :]
        Dc = Sc - Sc[:, [l0]]
        muc = Dc.mean(0); sdc = Dc.std(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = muc / (sdc / np.sqrt(k))
        zc = np.where(np.isfinite(zc), zc, 0.0); zc[l0] = -1e9
        c2 = int(np.argmax(zc))
        if zc[c2] > 0:
            pred_idx["cal_majority"][rid] = c2
            if c2 != l0: n_fire["cal_majority"] += 1
        if zc[c2] > 1.64:
            pred_idx["cal95"][rid] = c2
            if c2 != l0: n_fire["cal95"] += 1
        cal[rid] = (c2, float(zc[c2]))

    # margin-gated audit: revise only inside the field's own uncertainty region
    gate_out = {}
    for q in (1.0, 0.5, 0.2, 0.1):
        thr = float(np.quantile(margin, q))
        pidx = base_idx.copy()
        nf = 0
        for rid, (c2, zc2) in cal.items():
            if margin[rid] <= thr and zc2 > 1.64 and c2 != base_idx[rid]:
                pidx[rid] = c2
                nf += 1
        p = cls_arr[pidx][labels]
        gate_out["cal95_q%g" % q] = macro_iou(gt, p, cls)
        gate_out["fires_q%g" % q] = nf

    out = {"scene": args.scene, "baseline": m_base, "fires": n_fire, "gate": gate_out}
    for t, pidx in pred_idx.items():
        p = cls_arr[pidx][labels]
        out[t] = macro_iou(gt, p, cls)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "onarm_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("SCENE", args.scene, "baseline %.4f" % m_base,
          " ".join("%s %.4f" % (t, out[t]) for t in tags),
          "| gate " + " ".join("%s %.4f(f%d)" % (k, gate_out[k], gate_out["fires_" + k.split("_", 1)[1]])
                               for k in gate_out if not k.startswith("fires")),
          flush=True)


if __name__ == "__main__":
    main()