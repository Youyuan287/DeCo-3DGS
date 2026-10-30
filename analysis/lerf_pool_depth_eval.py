"""Pool-depth evaluation (PREREG_pool_depth.md).

Compares the released gap rule over top-5 (reference), top-10 and top-20 pools.
Scene-stratified paired bootstrap (B=2000, seed 20261030); per-family and per-scene
reporting; frozen gate decision.
"""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
PD = os.path.join(R, "lerf_pooldepth_20261030")
AT = os.path.join(R, "lerf_attribution_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


fam = {}
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fam[(r["scene"], r["image"], r["prompt"])] = r["cat"]

rows = []
with open(os.path.join(PD, "all_pooldepth.csv"), "w", newline="", encoding="utf-8") as out:
    wr = None
    for s in SC:
        with open(os.path.join(PD, "%s_pooldepth.csv" % s), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = dict(scene=r["scene"], image=r["image"], prompt=r["prompt"],
                         cat=fam.get((r["scene"], r["image"], r["prompt"]), "?"))
                for k in ("n5", "n10", "n20", "len5", "len10", "len20"):
                    d[k] = num(r[k])
                for k in ("gap5", "gap10", "gap20"):
                    d[k] = num(r[k])
                rows.append(d)
                if wr is None:
                    wr = csv.DictWriter(out, fieldnames=list(d))
                    wr.writeheader()
                wr.writerow(d)

Q = rows
print("queries:", len(Q))
# parity check vs attribution main (gap5 == main)
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    m = {(r["scene"], r["image"], r["prompt"]): num(r["main"]) for r in csv.DictReader(f)}
dmax = max(abs(q["gap5"] - m.get((q["scene"], q["image"], q["prompt"]), float("nan"))) for q in Q)
print("parity max |gap5 - main| =", dmax)


def scene_mean(key):
    per = {}
    for q in Q:
        per.setdefault(q["scene"], []).append(q[key])
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def paired_delta(key):
    vals, sc = [], []
    for q in Q:
        vals.append(q[key] - q["gap5"])
        sc.append(q["scene"])
    # stratified bootstrap
    by = {}
    for v, s in zip(vals, sc):
        by.setdefault(s, []).append(v)
    arrs = {s: np.array(v) for s, v in by.items()}
    base = float(np.mean([a.mean() for a in arrs.values()]))
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(B_BOOT):
        acc = []
        for s, a in arrs.items():
            acc.extend(a[rng.integers(0, len(a), len(a))].tolist())
        out.append(np.mean(acc))
    return base, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), vals


for key in ("gap5", "gap10", "gap20"):
    sm, per = scene_mean(key)
    print("%-6s scene-mean %.4f  per-scene %s" % (key, sm, {k: round(v, 3) for k, v in per.items()}))

for key in ("gap10", "gap20"):
    base, lo, hi, vals = paired_delta(key)
    pos = sum(1 for v in vals if v > 1e-9)
    neg = sum(1 for v in vals if v < -1e-9)
    print("%s - gap5: delta %+.4f [%.4f, %.4f], wins %d losses %d" % (key, base, lo, hi, pos, neg))

print("\nper family (mean IoU):")
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
for f_, sub in sorted(fams.items()):
    print("  %-10s n=%3d gap5 %.3f gap10 %.3f gap20 %.3f" % (
        f_, len(sub), np.mean([x["gap5"] for x in sub]),
        np.mean([x["gap10"] for x in sub]), np.mean([x["gap20"] for x in sub])))

sm20, per20 = scene_mean("gap20")
base_d, lo_d, hi_d, _ = paired_delta("gap20")
min_scene = min(per20[s] - scene_mean("gap5")[1][s] for s in SC)
deep = [x for x in Q if x["cat"] == "deep"]
deep_delta = np.mean([x["gap20"] - x["gap5"] for x in deep]) if deep else 0.0
gates = [
    ("gap20 delta >= +0.01 & CI_low > 0", base_d >= 0.01 and lo_d > 0),
    ("no scene worse than -0.01", min_scene >= -0.01),
    ("deep family not worse by >0.02", deep_delta >= -0.02),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")