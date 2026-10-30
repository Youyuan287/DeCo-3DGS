"""Evaluation of the learned subset selector (PREREG_subset_selector.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
SUB = os.path.join(R, "lerf_subset_20261030")
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
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        main[(r["scene"], r["image"], r["prompt"])] = num(r["main"])

fam = {}
with open(os.path.join(SUB, "preds.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fam[(r["scene"], r["image"], r["prompt"])] = r["cat"]

Q = []
for s in SC:
    with open(os.path.join(SUB, "%s_subset.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            Q.append(dict(scene=r["scene"], key=k, cat=fam.get(k, "?"),
                          gap_len=num(r["gap_len"]),
                          s0=num(r["s0_iou"]), s1=num(r["s1_iou"]),
                          s2=num(r["s2_iou"]), s3=num(r["s3_iou"]),
                          l0=num(r["s0_len"]), l1=num(r["s1_len"]),
                          l2=num(r["s2_len"]), l3=num(r["s3_len"])))

print("queries:", len(Q))
dmax = max(abs(q["s0"] - main.get(q["key"], float("nan"))) for q in Q)
print("parity max |s0 - released main| =", dmax)


def scene_mean(key):
    per = {}
    for q in Q:
        per.setdefault(q["scene"], []).append(q[key])
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def paired(key):
    by = {}
    for q in Q:
        by.setdefault(q["scene"], []).append(q[key] - q["s0"])
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


for key in ("s0", "s1", "s2", "s3"):
    sm, per = scene_mean(key)
    print("%-3s scene-mean %.4f per-scene %s" % (key, sm, {k: round(v, 3) for k, v in per.items()}))
for key in ("s1", "s2", "s3"):
    m, lo, hi, v = paired(key)
    print("%s - s0: %+.4f [%.4f, %.4f] wins %d losses %d"
          % (key, m, lo, hi, int((v > 1e-9).sum()), int((v < -1e-9).sum())))

print("\nper family (s0 -> s1):")
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {}
for f_, sub in sorted(fams.items()):
    d = float(np.mean([x["s1"] for x in sub]) - np.mean([x["s0"] for x in sub]))
    fam_delta[f_] = d
    print("  %-10s n=%3d %.3f -> %.3f  (%+.3f)" % (
        f_, len(sub), np.mean([x["s0"] for x in sub]), np.mean([x["s1"] for x in sub]), d))

print("\nselection sizes: gap %.2f -> s1 %.2f (s2 %.2f, s3 %.2f)" % (
    np.mean([q["l0"] for q in Q]), np.mean([q["l1"] for q in Q]),
    np.mean([q["l2"] for q in Q]), np.mean([q["l3"] for q in Q])))

sm0, per0 = scene_mean("s0")
sm1, per1 = scene_mean("s1")
m, lo, hi, _ = paired("s1")
per_scene_delta = {}
for s in SC:
    per_scene_delta[s] = per1[s] - per0[s]
gates = [
    ("S1 delta >= +0.01 & CI_low > 0", (sm1 - sm0) >= 0.01 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found family not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")