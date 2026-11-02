"""Aggregate LERF local-contrast JSONs (per-image -> per-scene -> macro)."""
import glob, json, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "."
recs = {}
for p in sorted(glob.glob(os.path.join(d, "*_local.json"))):
    r = json.load(open(p))
    recs[r["scene"]] = r
scenes = sorted(recs)
keys = [k for k in recs[scenes[0]]["summary"] if k not in ("n",)]
keys = [k for k in keys if k in ("main", "E0", "Eknn", "Erand", "Ehard", "Epair")]

print("scenes:", ", ".join("%s(%d)" % (s, recs[s]["summary"]["n"]) for s in scenes))
print("%-8s" % "" + "".join("%10s" % s[:9] for s in scenes) + "%10s" % "macro")
for k in keys:
    vals = [recs[s]["summary"][k] for s in scenes]
    print("%-8s" % k + "".join("%10.4f" % v for v in vals) + "%10.4f" % np.mean(vals))

# per-query paired bootstrap of each variant against E0, scene-stratified
rows = {s: recs[s]["rows"] for s in scenes}

def paired(a, b, seed=0):
    diffs = []
    for s in scenes:
        for r in rows[s]:
            if a in r and b in r:
                diffs.append(r[a] - r[b])
    diffs = np.array(diffs, float)
    rng = np.random.default_rng(seed)
    bs = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(4000)])
    return diffs.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(diffs)

print("\npaired per-query deltas vs E0")
for k in keys:
    if k in ("main", "E0"):
        continue
    m, lo, hi, n = paired(k, "E0")
    win = sum(1 for s in scenes for r in rows[s] if r.get(k, 0) > r.get("E0", 0))
    print("%-8s %+7.4f  CI[%+7.4f,%+7.4f]  win %3d/%d" % (k, m, lo, hi, win, n))

print("\npaired per-query deltas vs main (released)")
for k in keys:
    if k == "main":
        continue
    m, lo, hi, n = paired(k, "main")
    print("%-8s %+7.4f  CI[%+7.4f,%+7.4f]" % (k, m, lo, hi))