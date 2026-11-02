"""Aggregate local-contrast per-scene JSONs into a macro table + paired tests."""
import glob, json, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "."
recs = {}
for p in sorted(glob.glob(os.path.join(d, "scene*.json"))):
    r = json.load(open(p))
    recs[r["scene"]] = r
scenes = sorted(recs)
tags = [v["tag"] for v in recs[scenes[0]]["variants"]]

def get(s, t, k):
    for v in recs[s]["variants"]:
        if v["tag"] == t:
            return v.get(k, float("nan"))
    return float("nan")

print("scenes:", len(scenes))
print("%-14s" % "" + "".join("%8s" % s[5:9] for s in scenes) + "     macro")
for t in tags:
    vals = [get(s, t, "miou") for s in scenes]
    print("%-14s" % t + "".join("%8.1f" % v for v in vals) + "  %8.2f" % np.nanmean(vals))
print()
print("%-14s" % "tag" + "%8s%8s%8s%8s" % ("mIoU", "mAcc", "racc", "mg_med"))
for t in tags:
    print("%-14s" % t + "%8.2f%8.2f%8.2f%8.4f" % (
        np.nanmean([get(s, t, "miou") for s in scenes]),
        np.nanmean([get(s, t, "macc") for s in scenes]),
        np.nanmean([get(s, t, "racc") for s in scenes]),
        np.nanmean([get(s, t, "mg_med") for s in scenes])))

def ci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)])
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("\npaired deltas vs zR+E0 (scene level, n=%d)" % len(scenes))
for t in tags:
    if t in ("zR+E0", "zR", "raw"):
        continue
    d_ = [get(s, t, "miou") - get(s, "zR+E0", "miou") for s in scenes]
    m, lo, hi = ci(d_)
    print("%-14s %+7.2f  CI[%+7.2f,%+7.2f]  win %2d/%d" % (
        t, m, lo, hi, sum(1 for x in d_ if x > 0), len(scenes)))

print("\ncoverage / structure")
for s in scenes:
    r = recs[s]
    print("  %s knn_deg=%.2f sib_deg=%.2f sib_zero=%.2f" % (
        s[5:9], r["knn_deg"], r["sib_deg"], r["sib_zero"]))