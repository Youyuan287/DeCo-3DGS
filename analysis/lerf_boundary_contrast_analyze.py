#!/usr/bin/env python
"""Analysis for the pre-registered boundary-contrast pilot (20_BOUNDARY_CONTRAST_PREREG.md).

Reads results/lerf_boundary_contrast_20261104/<scene>_units.csv and reports B1/B2/B3
plus the pre-registered diagnostics. No parameter selection here.
"""
import csv, glob, os, sys
import numpy as np
from scipy.stats import spearmanr

D = sys.argv[1] if len(sys.argv) > 1 else "results/lerf_boundary_contrast_20261104"
rng = np.random.default_rng(0)


def num(r, c):
    try:
        v = float(r[c])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, "*_units.csv"))):
        rows += list(csv.DictReader(open(f)))
    return rows


def group(rows):
    q = {}
    for r in rows:
        q.setdefault((r["scene"], r["image"], r["prompt"]), []).append(r)
    return q


def nested(sel):
    per = {}
    for k, v in sel.items():
        per.setdefault(k[0], {}).setdefault(k[1], []).append(v)
    return float(np.mean([np.mean([np.mean(x) for x in d.values()]) for d in per.values()]))


def zs(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    o = np.zeros_like(x)
    if ok.sum() > 1:
        o[ok] = (x[ok] - x[ok].mean()) / (x[ok].std() + 1e-12)
    return o


def pick(q, score_fn):
    sel = {}
    for k, v in q.items():
        s = score_fn(v)
        sel[k] = num(v[int(np.argmax(s))], "iou")
    return sel


def rho_mean(q, field, only_nonempty=False):
    out = []
    for k, v in q.items():
        j = np.array([num(r, field) for r in v])
        i = np.array([num(r, "iou") for r in v])
        ok = np.isfinite(j) & np.isfinite(i)
        if only_nonempty:
            ok &= np.array([num(r, "area_ratio") for r in v]) > 0
        if ok.sum() >= 3 and np.std(j[ok]) > 1e-9 and np.std(i[ok]) > 1e-9:
            out.append(spearmanr(j[ok], i[ok])[0])
    return np.array(out), [k for k in q if True][:0]


def boot_mean(x, n=2000):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 5:
        return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    rows = load()
    if not rows:
        print("no rows in", D)
        return
    q = group(rows)
    print("rows %d  units %d  scenes %d" % (len(rows), len(q), len({k[0] for k in q})))
    n_empty = sum(1 for r in rows if num(r, "area_ratio") == 0)
    n_jbad = sum(1 for r in rows if not np.isfinite(num(r, "J")))
    print("area==0 rows %d (%.1f%%)   J invalid rows %d (%.1f%%)"
          % (n_empty, 100 * n_empty / len(rows), n_jbad, 100 * n_jbad / len(rows)))

    # ---- macro comparison (B2) ----
    sc = lambda v: zs([num(r, "score") for r in v])
    jz = lambda v: zs([num(r, "J") for r in v])
    ar = lambda v: zs([np.log(max(num(r, "area_ratio"), 1e-9)) for r in v])
    ne = lambda v: zs([1.0 if num(r, "area_ratio") > 0 else 0.0 for r in v])
    rules = {
        "rank1": None,
        "max score": lambda v: [num(r, "score") for r in v],
        "non-empty flag": lambda v: ne(v) + 1e-6 * sc(v),
        "score+logarea": lambda v: sc(v) + ar(v),
        "score+J (eta=1)": lambda v: sc(v) + jz(v),
        "J alone": lambda v: jz(v),
    }
    macro = {}
    sel_all = {}
    for name, f in rules.items():
        if name == "rank1":
            sel = {k: num(v[0], "iou") for k, v in q.items()}
        else:
            sel = pick(q, f)
        sel_all[name] = sel
        macro[name] = nested(sel)
        print("  %-18s %.4f" % (name, macro[name]))
    print("  %-18s %.4f" % ("oracle", float(np.mean([max(num(r, "iou") for r in v)
                                                    for v in q.values()]))))

    # paired bootstrap: score+J minus each control, over units
    keys = sorted(q)
    def diff(ctrl):
        a = np.array([sel_all["score+J (eta=1)"][k] for k in keys])
        b = np.array([sel_all[ctrl][k] for k in keys])
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        if len(d) < 5:
            return np.nan, (np.nan, np.nan)
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
        return d.mean(), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    print("\npaired diff over %d units (per-unit mean IoU):" % len(keys))
    for ctrl in ("rank1", "non-empty flag", "score+logarea"):
        m, ci = diff(ctrl)
        print("  score+J - %-14s %+.4f  CI [%+.4f,%+.4f]" % (ctrl, m, ci[0], ci[1]))

    # ---- B1: within-unit rank correlation ----
    print("\nB1 within-unit spearman(field, IoU), mean [95%% CI]:")
    for field, only in (("J", True), ("score", True), ("J", False), ("score", False)):
        a, _ = rho_mean(q, field, only)
        lo, hi = boot_mean(a)
        print("  %-6s %-9s n=%3d  mean %+.3f  CI [%+.3f,%+.3f]"
              % (field, "area>0" if only else "all", len(a), a.mean(), lo, hi))

    # ---- B3: partial correlation removing log-area and emptiness ----
    print("\nB3 partial correlation (area>0 rows only):")
    xj, yi, xa = [], [], []
    for k, v in q.items():
        for r in v:
            if num(r, "area_ratio") <= 0:
                continue
            j, i, a_ = num(r, "J"), num(r, "iou"), np.log(max(num(r, "area_ratio"), 1e-12))
            if np.isfinite(j) and np.isfinite(i):
                xj.append(j); yi.append(i); xa.append(a_)
    xj, yi, xa = map(np.array, (xj, yi, xa))
    if len(xj) > 10:
        from numpy.linalg import lstsq
        def resid(y, x):
            A = np.c_[np.ones_like(x), x]
            return y - A @ lstsq(A, y, rcond=None)[0]
        pr = spearmanr(resid(xj, xa), resid(yi, xa))[0]
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(xj), len(xj))
            if np.std(xa[idx]) < 1e-9:
                continue
            bs.append(spearmanr(resid(xj[idx], xa[idx]), resid(yi[idx], xa[idx]))[0])
        print("  rho(J, IoU | log area) = %+.3f  CI [%+.3f,%+.3f]  n=%d"
              % (pr, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(xj)))

    # ---- pre-registered diagnostics (prereg section 3) ----
    print("\ndiagnostics:")
    for f1, f2 in (("J", "score"), ("J", "a_ring"), ("J", "a_in"), ("a_ring", "score")):
        x = np.array([num(r, f1) for r in rows]); y = np.array([num(r, f2) for r in rows])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 10:
            print("  rho(%s, %s) = %+.3f  n=%d" % (f1, f2, spearmanr(x[ok], y[ok])[0], ok.sum()))

    print("\nw sensitivity (within-unit rho(J_w, IoU), area>0):")
    for f in ("J_w3", "J", "J_w9"):
        a, _ = rho_mean(q, f, True)
        print("  %-5s mean %+.3f  n=%d" % (f, a.mean() if len(a) else np.nan, len(a)))


if __name__ == "__main__":
    main()