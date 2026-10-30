"""Evaluation of F1 fused score (PREREG_f1_fused.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
F1 = os.path.join(R, "lerf_f1_20261031")
AT = os.path.join(R, "lerf_attribution_20261030")
PI = os.path.join(R, "lerf_pruneinclude_20261030")
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
with open(os.path.join(R, "lerf_subset_20261030", "preds.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fam[(r["scene"], r["image"], r["prompt"])] = r["cat"]
curC = {}
for s in SC:
    with open(os.path.join(PI, "%s_pruneinclude.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            curC[(r["scene"], r["image"], r["prompt"])] = num(r["c_iou"])

Q = []
for s in SC:
    with open(os.path.join(F1, "%s_f1.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            Q.append(dict(scene=r["scene"], key=k, cat=fam.get(k, "?"), c=curC.get(k, float("nan")),
                          m=num(r["main_iou"]), f=num(r["fused_iou"]),
                          lf=num(r["fused_len"]), lm=num(r["gap_len"])))

print("queries:", len(Q))
print("parity max |main - released| =", max(abs(q["m"] - main.get(q["key"], float("nan"))) for q in Q))


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
smC, _ = scene_mean("c")
smF, perF = scene_mean("f")
d, lo, hi, v = paired("f")
print("main %.4f | C %.4f | F1 %.4f" % (sm, smC, smF))
print("F1 delta %+.4f [%.4f, %.4f] wins %d losses %d | len %.2f (main %.2f)"
      % (d, lo, hi, int((v > 1e-9).sum()), int((v < -1e-9).sum()),
         np.mean([q["lf"] for q in Q]), np.mean([q["lm"] for q in Q])))

print("\nper family (F1 vs main):")
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {}
for f_, sub in sorted(fams.items()):
    dd = float(np.mean([x["f"] for x in sub]) - np.mean([x["m"] for x in sub]))
    fam_delta[f_] = dd
    print("  %-11s n=%3d %.3f -> %.3f (%+.3f)"
          % (f_, len(sub), np.mean([x["m"] for x in sub]), np.mean([x["f"] for x in sub]), dd))

per_scene_delta = {s: perF[s] - perm[s] for s in SC}
gates = [
    ("F1 delta >= +0.01 & CI_low > 0", d >= 0.01 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")