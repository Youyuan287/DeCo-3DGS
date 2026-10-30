"""Candidate-generation grid evaluation (PREREG_poolgen.md).

LOSO selection over the 12 frozen configurations; held-out scene-mean delta vs the
released configuration (mp50_lr001_n5, verified against the released main); family
analysis; frozen gate decision.
"""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
PG = os.path.join(R, "lerf_poolgen_20261030")
AT = os.path.join(R, "lerf_attribution_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030
CONFIGS = ["mp%d_lr%s_n%d" % (mp, str(lr).replace(".", ""), nr)
           for mp in (50, 20, 10) for lr in (0.01, 0.005) for nr in (5, 10)]
BASE = "mp50_lr001_n5"


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


fam = {}
main = {}
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        k = (r["scene"], r["image"], r["prompt"])
        fam[k] = r["cat"]
        main[k] = num(r["main"])

Q = []
for s in SC:
    with open(os.path.join(PG, "%s_poolgen.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            d = dict(scene=r["scene"], key=k, cat=fam.get(k, "?"), main=main.get(k, float("nan")))
            for c in CONFIGS:
                d[c] = num(r["gap_" + c])
                d["n_" + c] = num(r["n_" + c])
            Q.append(d)

print("queries:", len(Q))
dmax = max(abs(q[BASE] - q["main"]) for q in Q)
print("parity max |grid(base) - released main| =", dmax)


def scene_mean(key):
    per = {}
    for q in Q:
        v = q[key] if isinstance(key, str) else key(q)
        per.setdefault(q["scene"], []).append(v)
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


print("\nfull grid (scene-mean):")
for c in CONFIGS:
    sm, _ = scene_mean(c)
    print("  %-16s %.4f" % (c, sm))

# LOSO selection
chosen = {}
for s in SC:
    best, bestv = None, -1
    for c in CONFIGS:
        vals = [q[c] for q in Q if q["scene"] != s]
        v = float(np.mean(vals))
        if v > bestv + 1e-12:
            best, bestv = c, v
    chosen[s] = best
print("\nLOSO choices:", chosen)

los = [q for q in Q]
delta_vals, base_vals, chosen_vals = [], [], []
for q in los:
    c = chosen[q["scene"]]
    chosen_vals.append(q[c])
    base_vals.append(q[BASE])
    delta_vals.append(q[c] - q[BASE])

per_scene = {}
for q, c in zip(los, chosen_vals):
    per_scene.setdefault(q["scene"], []).append(c)
sm_chosen = float(np.mean([np.mean(v) for v in per_scene.values()]))
sm_base = scene_mean(BASE)[0]
print("LOSO held-out scene-mean: base %.4f -> chosen %.4f (delta %+.4f)"
      % (sm_base, sm_chosen, sm_chosen - sm_base))

by = {}
for q, d in zip(los, delta_vals):
    by.setdefault(q["scene"], []).append(d)
arrs = {s: np.array(v) for s, v in by.items()}
rng = np.random.default_rng(SEED)
boot = []
for _ in range(B_BOOT):
    acc = []
    for s, a in arrs.items():
        acc.extend(a[rng.integers(0, len(a), len(a))].tolist())
    boot.append(np.mean(acc))
lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
print("paired stratified bootstrap delta %+.4f [%.4f, %.4f]" % (np.mean(delta_vals), lo, hi))

per_scene_delta = {s: float(np.mean(v)) for s, v in by.items()}
print("per-scene delta:", {k: round(v, 4) for k, v in per_scene_delta.items()})

print("\nper family (chosen vs base):")
fams = {}
for q, c in zip(los, chosen_vals):
    fams.setdefault(q["cat"], ([], []))
    fams[q["cat"]][0].append(c)
    fams[q["cat"]][1].append(q[BASE])
fam_delta = {}
for f_, (cs, bs) in sorted(fams.items()):
    d = float(np.mean(cs) - np.mean(bs))
    fam_delta[f_] = d
    print("  %-10s n=%3d base %.3f chosen %.3f delta %+.3f"
          % (f_, len(cs), np.mean(bs), np.mean(cs), d))

print("\npool sizes (mean n per config, chosen folds):")
for c in set(chosen.values()):
    vals = [q["n_" + c] for q in Q if chosen[q["scene"]] == c]
    print("  %-16s mean pool %.1f" % (c, np.mean(vals) if vals else float("nan")))

gates = [
    ("LOSO delta >= +0.01 & CI_low > 0", (sm_chosen - sm_base) >= 0.01 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("recall_fail improves & deep not worse by >0.02",
     fam_delta.get("recall_fail", -1) > 0 and fam_delta.get("deep", 0) >= -0.02),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")