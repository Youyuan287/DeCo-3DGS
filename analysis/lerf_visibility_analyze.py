#!/usr/bin/env python
"""Does the '+4.2 from avoiding empty candidates' transfer to the official readout?"""
import csv, glob, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results/lerf_visibility_20261104"
rng = np.random.default_rng(0)


def num(r, c):
    try:
        v = float(r[c])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


rows = []
for f in sorted(glob.glob(os.path.join(D, "*_official.csv"))):
    rows += list(csv.DictReader(open(f)))
if not rows:
    print("no rows in", D)
    sys.exit()

q = {}
for r in rows:
    q.setdefault((r["scene"], r["image"], r["prompt"]), []).append(r)
unit = {k: v[0] for k, v in q.items()}
print("units %d  scenes %d" % (len(unit), len({k[0] for k in unit})))


def nested(vals):
    per = {}
    for k, v in vals.items():
        per.setdefault(k[0], {}).setdefault(k[1], []).append(v)
    return float(np.mean([np.mean([np.mean(x) for x in d.values()]) for d in per.values()]))


print("\nmacro IoU (official readout):")
res = {}
for v in ("base", "V1", "V2", "V3"):
    res[v] = {k: num(r, v + "_iou") for k, r in unit.items()}
    print("  %-5s %.4f" % (v, nested(res[v])))

keys = sorted(unit)
print("\npaired diff over %d units:" % len(keys))
for v in ("V1", "V2", "V3"):
    a = np.array([res[v][k] for k in keys])
    b = np.array([res["base"][k] for k in keys])
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if len(d) < 5:
        continue
    bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(4000)]
    print("  %-3s - base  %+.4f  CI [%+.4f,%+.4f]   changed units %d/%d"
          % (v, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
             int((np.abs(d) > 1e-9).sum()), len(d)))

# ---- how much room does the rule actually have? ----
n = len(unit)
empty = sum(1 for r in unit.values() if num(r, "base_px") == 0)
allinvis = sum(1 for r in unit.values()
               if num(r, "n_prefix_visible") == 0 and num(r, "base_px") == 0)
some_vis_pool = sum(1 for r in unit.values()
                    if num(r, "n_prefix_visible") == 0 and num(r, "base_px") == 0
                    and num(r, "V3_px") > 0)
print("\nroom for the rule:")
print("  base renders empty            %4d / %d  (%.1f%%)" % (empty, n, 100 * empty / n))
print("  prefix fully invisible        %4d" % allinvis)
print("  ...pool has a visible alt     %4d  <- the only cases V3 can act on" % some_vis_pool)

# per-scene macro for the record
print("\nper-scene macro (base / V3):")
for sc in sorted({k[0] for k in unit}):
    sub = {k: v for k, v in unit.items() if k[0] == sc}
    print("  %-14s %.4f / %.4f  (n=%d)" % (sc, nested({k: num(r, 'base_iou') for k, r in sub.items()}),
                                           nested({k: num(r, 'V3_iou') for k, r in sub.items()}), len(sub)))