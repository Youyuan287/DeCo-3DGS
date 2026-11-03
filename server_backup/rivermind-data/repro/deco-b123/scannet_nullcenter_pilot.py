#!/usr/bin/env python3
"""Null-centered audit pilot on ScanNet-10 (on-arm, C1+C2, V2 off).

Why: the ScanNet audit fails because the certificate is a sign test on
delta_v(c, l0) = s_v(c) - s_v(l0), and a sign test is valid only when the
paired advantage is centered at zero under the null.  On LERF the pairing is
"two answers to the same query", so the common bias cancels by construction.
On a closed set the witness carries a class-dependent bias b, so delta is
centered at b != 0 and every root looks "significant".

Pilot: estimate b from *placebo* roots -- roots on which the released field is
most confident (largest top-1 minus top-2 margin).  If the field is right there,
any stable witness advantage is bias, not signal.  Subtract the estimate and run
the same certificate.  No ground truth enters the estimator; GT is used only to
score the result.

Variants (all chosen a priori, no GT tuning):
  audit95 / audit_maj      : existing certificate, no centering
  cal95                   : global view mean over all observations (existing)
  plac95_q{0.1,0.25,0.5}  : placebo mean over top-q confident roots
  placmaj_q0.25           : placebo centering, majority threshold (z > 0)
  plac_gate95             : placebo centering + revise only below median margin
  incb_q0.25              : placebo bias conditioned on the incumbent class
  resid95_q0.25           : per-class affine calibration to the released score

Diagnostics (GT only for evaluation):
  witness top-1 macro, detection / false-alarm rates of the witness,
  per-fire net corrected Gaussians, fire precision.
"""
import argparse
import json
import os
import sys

import numpy as np

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import SUBSETS, metrics  # noqa: E402


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def zstat(D, k):
    mu = D.mean(0)
    sd = D.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = mu / (sd / np.sqrt(k))
    return np.where(np.isfinite(z), z, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--abl", required=True, help="dir with <scene>/predictions_and_scores.npz")
    ap.add_argument("--input", required=True, help="dir with observations_level3.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scene = args.scene
    d = np.load(os.path.join(args.abl, scene, "predictions_and_scores.npz"))
    gt = d["gt"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    S0 = d["scores_off"].astype(np.float64)          # (n_root, 19), C1+C2 on, V2 off
    cls = SUBSETS[19]
    cls_arr = np.array(cls)
    n_root = S0.shape[0]
    base_idx = S0.argmax(1)
    srt = np.sort(S0, axis=1)
    margin = srt[:, -1] - srt[:, -2]

    tb = np.load(os.path.join(args.abl, scene, "text_bank.npz"))
    bank = l2n(tb["textbank"].astype(np.float32))
    if bank.ndim == 3:
        banks = bank.reshape(-1, bank.shape[-2], bank.shape[-1])
    else:
        banks = bank[None]

    def score_views(X):
        best = None
        for b in banks:
            s = X @ b.T
            best = s if best is None else np.maximum(best, s)
        return best

    obs = np.load(os.path.join(args.input, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float32))
    reg = obs["region_ids"].astype(np.int64)
    w_all = obs["weights"].astype(np.float64)
    order = np.argsort(reg, kind="stable")
    reg_s = reg[order]
    F_s = F[order]
    w_s = w_all[order]

    Sroot = {}
    Wroot = {}
    for rid in range(n_root):
        lo = np.searchsorted(reg_s, rid, "left")
        hi = np.searchsorted(reg_s, rid, "right")
        if hi - lo >= 3:
            Sroot[rid] = score_views(F_s[order[lo:hi]])
            Wroot[rid] = w_s[order[lo:hi]]

    auditable = np.array(sorted(Sroot.keys()))
    gauss_valid = np.zeros(len(gt), dtype=bool)
    gt_idx = np.full(len(gt), -1, dtype=np.int64)
    for k, c in enumerate(cls_arr):
        m = gt == c
        gt_idx[m] = k
        gauss_valid |= m
    inter = np.bincount(labels[gauss_valid] * 19 + gt_idx[gauss_valid],
                        minlength=n_root * 19).reshape(n_root, 19).astype(np.int64)
    root_total = inter.sum(1)
    base_k = S0.argmax(1)
    base_correct = inter[np.arange(n_root), base_k]

    def macro(pred_idx):
        p = cls_arr[pred_idx][labels]
        return float(metrics(gt, p, cls)["mIoU"])

    def fire_stats(pred_idx):
        fired = np.where(pred_idx != base_idx)[0]
        net = 0
        good = bad = 0
        for rid in fired:
            dnet = int(inter[rid, pred_idx[rid]] - inter[rid, base_idx[rid]])
            net += dnet
            if dnet > 0:
                good += 1
            elif dnet < 0:
                bad += 1
        return dict(fires=int(len(fired)), net_corrected=int(net),
                    fire_precision=(good / len(fired)) if len(fired) else None,
                    good=good, bad=bad, ties=len(fired) - good - bad)

    out = {"scene": scene, "n_root": int(n_root), "n_auditable": int(len(auditable)),
           "variants": {}, "diagnostics": {}}

    def add(name, pred_idx):
        out["variants"][name] = dict(macro=macro(pred_idx), **fire_stats(pred_idx))

    # --- existing certificates -------------------------------------------
    class_mean = score_views(F_s).mean(0)   # matches the existing cal95 baseline

    def run_cert(center_fn, tau, gate=None):
        pred = base_idx.copy()
        for rid in auditable:
            S = Sroot[rid]
            l0 = int(base_idx[rid])
            center = center_fn(rid, l0)
            Sc = S if center is None else S - center[None, :]
            k = max(2, len(S) // 2)
            z = zstat(Sc - Sc[:, [l0]], k)
            z[l0] = -1e9
            c = int(np.argmax(z))
            if z[c] > tau and (gate is None or margin[rid] <= gate):
                pred[rid] = c
        return pred

    add("audit_maj", run_cert(lambda rid, l0: None, 0.0))
    add("audit95", run_cert(lambda rid, l0: None, 1.64))
    add("cal95", run_cert(lambda rid, l0: class_mean, 1.64))
    add("cal_maj", run_cert(lambda rid, l0: class_mean, 0.0))

    # --- placebo bias estimators -----------------------------------------
    qs = (0.10, 0.25, 0.50)
    placebo = {}
    for q in qs:
        thr = float(np.quantile(margin[auditable], 1.0 - q))
        placebo[q] = [rid for rid in auditable if margin[rid] >= thr]
    # global per-class bias: mean view score over placebo roots
    beta = {}
    for q in qs:
        acc = np.zeros(19)
        nv = 0
        for rid in placebo[q]:
            acc += Sroot[rid].sum(0)
            nv += len(Sroot[rid])
        beta[q] = acc / nv
    # per-incumbent bias: mean delta_v(c, l0) over placebo roots with the same l0
    incb = {q: {} for q in qs}
    for q in qs:
        by_l0 = {}
        for rid in placebo[q]:
            by_l0.setdefault(int(base_idx[rid]), []).append(rid)
        for l0, rids in by_l0.items():
            if len(rids) < 5:
                continue
            acc = np.zeros(19)
            nv = 0
            for rid in rids:
                S = Sroot[rid]
                acc += (S - S[:, [l0]]).sum(0)
                nv += len(S)
            incb[q][l0] = acc / nv

    for q in qs:
        tag = "plac95_q%g" % q
        add(tag, run_cert(lambda rid, l0, q=q: beta[q], 1.64))
        if q == 0.25:
            add("placmaj_q0.25", run_cert(lambda rid, l0, q=q: beta[q], 0.0))
            med = float(np.median(margin[auditable]))
            add("plac_gate95_q0.25", run_cert(lambda rid, l0, q=q: beta[q], 1.64, gate=med))
    add("incb95_q0.25", run_cert(
        lambda rid, l0: incb[0.25].get(l0, beta[0.25]), 1.64))

    # --- affine calibration to the released score ------------------------
    # per class c: mean_v s_v(c) ~ a_c * S0_c + b_c, fitted on placebo roots
    a = np.ones(19)
    b = np.zeros(19)
    for c in range(19):
        xs, ys = [], []
        for rid in placebo[0.25]:
            xs.append(S0[rid, c])
            ys.append(Sroot[rid][:, c].mean())
        if len(xs) >= 20 and np.std(xs) > 1e-6:
            a[c], b[c] = np.polyfit(np.array(xs), np.array(ys), 1)
    add("resid95_q0.25", run_cert(
        lambda rid, l0: a * S0[rid] + b, 1.64))

    # --- diagnostics: does the witness carry any signal? -----------------
    def aggregate(S, w, mode):
        if mode == "mean":
            return S.mean(0)
        if mode == "max":
            return S.max(0)
        if mode == "top3":
            k = min(3, len(S))
            return np.sort(S, axis=0)[-k:].mean(0)
        if mode == "wmean":
            ww = np.maximum(w, 1e-8)
            return (S * ww[:, None]).sum(0) / ww.sum()
        raise ValueError(mode)

    modes = ("mean", "max", "top3", "wmean")
    det_all = {}
    for mode in modes:
        w_pred = base_idx.copy()
        d = {"auditable": int(len(auditable)), "mode": mode}
        agree = improve = right = wrong = 0
        for rid in auditable:
            S, w = Sroot[rid], Wroot[rid]
            l0 = int(base_idx[rid])
            a = aggregate(S, w, mode)
            wi = int(np.argmax(a))
            w_pred[rid] = wi
            rtot = root_total[rid]
            if rtot == 0:
                continue
            if base_correct[rid] > 0.5 * rtot:
                right += 1
                agree += int(wi != l0)
            else:
                wrong += 1
                improve += int(inter[rid, wi] > inter[rid, l0])
        d.update(baseline_correct_roots=int(right), baseline_wrong_roots=int(wrong),
                 witness_disagree_on_correct=int(agree), witness_improves_on_wrong=int(improve),
                 false_alarm_rate=(agree / max(1, right)),
                 detection_rate=(improve / max(1, wrong)))
        det_all[mode] = d
        out["variants"]["witness_top1_" + mode] = dict(macro=macro(w_pred), **fire_stats(w_pred))
    out["diagnostics"] = det_all

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "nullcenter_%s.json" % scene), "w") as f:
        json.dump(out, f, indent=2)
    print("SCENE", scene,
          " ".join("%s %.4f(f%d)" % (k, v["macro"], v["fires"])
                   for k, v in out["variants"].items()),
          "| " + " ".join("%s det %.2f fa %.2f" % (
              m, det_all[m]["detection_rate"], det_all[m]["false_alarm_rate"])
              for m in modes),
          flush=True)


if __name__ == "__main__":
    main()