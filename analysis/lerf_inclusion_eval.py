"""Evaluation of inclusion v1 (PREREG_inclusion_v1.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
INC = os.path.join(R, "lerf_inclusion_20261030")
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
    with open(os.path.join(INC, "%s_inclusion.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            Q.append(dict(scene=r["scene"], key=k, cat=fam.get(k, "?"), c=curC.get(k, float("nan")),
                          m=num(r["s0_iou"]), inc=num(r["inc_iou"]), ip=num(r["ip_iou"]),
                          li=num(r["inc_len"]), lp=num(r["ip_len"]), l0=num(r["s0_len"])))

print("queries:", len(Q))
print("parity max |s0-main| =", max(abs(q["m"] - main.get(q["key"], float("nan"))) for q in Q))


def scene_mean(key):
    per = {}
    for q in Q:
        per.setdefault(q["scene"], []).append(q[key])
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def paired(key, base="m"):
    by = {}
    for q in Q:
        by.setdefault(q["scene"], []).append(q[key] - q[base])
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
smC, perC = scene_mean("c")
print("main %.4f | C %.4f" % (sm, smC))
for tag in ("inc", "ip"):
    sm2, per2 = scene_mean(tag)
    d, lo, hi, v = paired(tag)
    print("%-3s scene-mean %.4f delta %+.4f [%.4f, %.4f] wins %d losses %d len %.2f"
          % (tag, sm2, d, lo, hi, int((v > 1e-9).sum()), int((v < -1e-9).sum()),
             np.mean([q["l" + ("i" if tag == "inc" else "p")] for q in Q])))

print("\nper family (inc vs main):")
fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {}
for f_, sub in sorted(fams.items()):
    d = float(np.mean([x["inc"] for x in sub]) - np.mean([x["m"] for x in sub]))
    fam_delta[f_] = d
    print("  %-11s n=%3d %.3f -> %.3f (%+.3f)"
          % (f_, len(sub), np.mean([x["m"] for x in sub]), np.mean([x["inc"] for x in sub]), d))

smI, perI = scene_mean("inc")
d, lo, hi, _ = paired("inc")
per_scene_delta = {s: perI[s] - perm[s] for s in SC}
gates = [
    ("inc delta >= +0.02 & CI_low > 0", d >= 0.02 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
    ("inc beats C by >= +0.005", (smI - smC) >= 0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")