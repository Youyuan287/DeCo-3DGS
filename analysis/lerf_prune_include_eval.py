"""Evaluation of the prune-and-include module (PREREG_prune_include.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
PI = os.path.join(R, "lerf_pruneinclude_20261030")
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


main = {}
acat = {}
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        k = (r["scene"], r["image"], r["prompt"])
        main[k] = num(r["main"])
        acat[k] = r["cat"]
fam = {}
with open(os.path.join(R, "lerf_subset_20261030", "preds.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fam[(r["scene"], r["image"], r["prompt"])] = r["cat"]

Q = []
for s in SC:
    with open(os.path.join(PI, "%s_pruneinclude.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            Q.append(dict(scene=r["scene"], key=k, cat=fam.get(k, "?"), acat=acat.get(k, "?"),
                          m=num(r["s0_iou"]), c=num(r["c_iou"]), c0=num(r["c0_iou"]),
                          inc=num(r["include"]), prn=num(r["pruned"])))

print("queries:", len(Q))
print("parity max |s0-main| =", max(abs(q["m"] - main.get(q["key"], float("nan"))) for q in Q))
print("include rate %.1f%% | prune rate %.1f%%"
      % (100 * np.mean([q["inc"] for q in Q]), 100 * np.mean([q["prn"] for q in Q])))


def scene_mean(key):
    per = {}
    for q in Q:
        per.setdefault(q["scene"], []).append(q[key])
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def paired(key):
    by = {}
    for q in Q:
        by.setdefault(q["scene"], []).append(q[key] - q["m"])
    arrs = {s: np.array(v) for s, v in by.items()}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(B_BOOT):
        acc = []
        for s, a in arrs.items():
            acc.extend(a[rng.integers(0, len(a), len(a))].tolist())
        boot.append(np.mean(acc))
    allv = np.concatenate(list(arrs.values()))
    return float(allv.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), allv


sm, perm = scene_mean("m")
print("main scene-mean %.4f %s" % (sm, {k: round(v, 3) for k, v in perm.items()}))
for tag in ("c", "c0"):
    sm2, per2 = scene_mean(tag)
    m, lo, hi, v = paired(tag)
    print("%s scene-mean %.4f delta %+.4f [%.4f, %.4f] wins %d losses %d"
          % (tag, sm2, m, lo, hi, int((v > 1e-9).sum()), int((v < -1e-9).sum())))

for tag in ("c", "c0"):
    print("\nper family (%s vs main):" % tag)
    fams = {}
    for q in Q:
        fams.setdefault(q["cat"], []).append(q)
    for f_, sub in sorted(fams.items()):
        d = float(np.mean([x[tag] for x in sub]) - np.mean([x["m"] for x in sub]))
        print("  %-11s n=%3d %.3f -> %.3f (%+.3f)"
              % (f_, len(sub), np.mean([x["m"] for x in sub]), np.mean([x[tag] for x in sub]), d))

print("\nby gap-error category (c vs main):")
cats = {}
for q in Q:
    cats.setdefault(q["acat"], []).append(q)
for c_, sub in sorted(cats.items()):
    print("  %-10s n=%3d %.3f -> %.3f (%+.3f)"
          % (c_, len(sub), np.mean([x["m"] for x in sub]), np.mean([x["c"] for x in sub]),
             np.mean([x["c"] for x in sub]) - np.mean([x["m"] for x in sub])))

smc, perc = scene_mean("c")
m, lo, hi, _ = paired("c")
per_scene_delta = {s: perc[s] - perm[s] for s in SC}
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {f_: float(np.mean([x["c"] for x in sub]) - np.mean([x["m"] for x in sub]))
             for f_, sub in fams.items()}
gates = [
    ("C delta >= +0.01 & CI_low > 0", (smc - sm) >= 0.01 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found family not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")