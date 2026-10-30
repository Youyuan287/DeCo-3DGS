"""Evaluation of the distribution-informed truncation module (PREREG_dist_trunc.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
DT = os.path.join(R, "lerf_disttrunc_20261030")
AT = os.path.join(R, "lerf_attribution_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030
RULES = ["d1", "d2", "d3", "d4"]


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


main = {}
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        main[(r["scene"], r["image"], r["prompt"])] = num(r["main"])

fam = {}
pf = os.path.join(R, "lerf_subset_20261030", "preds.csv")
if os.path.exists(pf):
    with open(pf, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fam[(r["scene"], r["image"], r["prompt"])] = r["cat"]

Q = []
for s in SC:
    with open(os.path.join(DT, "%s_disttrunc.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            d = dict(scene=r["scene"], key=k, cat=fam.get(k, "?"),
                     m=num(r["main_iou"]), ml=num(r["main_len"]))
            for tag in RULES:
                d[tag] = num(r[tag + "_iou"])
                d[tag + "l"] = num(r[tag + "_len"])
            Q.append(d)

print("queries:", len(Q))
dmax = max(abs(q["m"] - main.get(q["key"], float("nan"))) for q in Q)
print("parity max |main - released| =", dmax)


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
for tag in RULES:
    sm2, per2 = scene_mean(tag)
    m, lo, hi, v = paired(tag)
    print("%s scene-mean %.4f  delta %+.4f [%.4f, %.4f] wins %d losses %d  mean len %.2f (main %.2f)"
          % (tag, sm2, m, lo, hi, int((v > 1e-9).sum()), int((v < -1e-9).sum()),
             np.mean([q[tag + "l"] for q in Q]), np.mean([q["ml"] for q in Q])))

print("\nper family (main -> d1):")
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {}
for f_, sub in sorted(fams.items()):
    d = float(np.mean([x["d1"] for x in sub]) - np.mean([x["m"] for x in sub]))
    fam_delta[f_] = d
    print("  %-11s n=%3d %.3f -> %.3f (%+.3f)" % (
        f_, len(sub), np.mean([x["m"] for x in sub]), np.mean([x["d1"] for x in sub]), d))

sm1, per1 = scene_mean("d1")
m, lo, hi, _ = paired("d1")
per_scene_delta = {s: per1[s] - perm[s] for s in SC}
gates = [
    ("D1 delta >= +0.01 & CI_low > 0", (sm1 - sm) >= 0.01 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found family not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")