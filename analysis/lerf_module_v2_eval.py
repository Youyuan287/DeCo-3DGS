"""Evaluation of module v2 grid (PREREG_module_v2.md): LOSO selection over 9 configs."""
import os, csv, re
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
V2 = os.path.join(R, "lerf_modv2_20261030")
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


main, acat = {}, {}
with open(os.path.join(AT, "per_query_joined.csv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        k = (r["scene"], r["image"], r["prompt"])
        main[k] = num(r["main"])
        acat[k] = r["cat"]
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
cfg_names = None
for s in SC:
    with open(os.path.join(V2, "%s_modv2.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (r["scene"], r["image"], r["prompt"])
            if cfg_names is None:
                cfg_names = sorted({c[:-4] for c in r if re.match(r"^a\d\d_(b\d+|off)_iou$", c)})
            d = dict(scene=r["scene"], key=k, cat=fam.get(k, "?"), acat=acat.get(k, "?"),
                     m=num(r["s0_iou"]), c=curC.get(k, float("nan")))
            for c in cfg_names:
                d[c] = num(r[c + "_iou"])
            Q.append(d)

print("queries:", len(Q), "configs:", cfg_names)
print("parity max |s0-main| =", max(abs(q["m"] - main.get(q["key"], float("nan"))) for q in Q))
if "a05_off" in cfg_names:
    print("parity max |a05_off - C| =", max(abs(q["a05_off"] - q["c"]) for q in Q))


def scene_mean(key):
    per = {}
    for q in Q:
        per.setdefault(q["scene"], []).append(q[key])
    return float(np.mean([np.mean(v) for v in per.values()])), {s: float(np.mean(v)) for s, v in per.items()}


def cfg_order(c):
    m = re.match(r"a(\d\d)_(b(\d\d)|off)", c)
    alpha = int(m.group(1)) / 100.0
    beta = 0.0 if m.group(2) == "off" else int(m.group(3)) / 100.0
    return (beta, abs(alpha - 0.5))


sm_main, per_main = scene_mean("m")
print("main scene-mean %.4f" % sm_main)
for c in cfg_names:
    sm, _ = scene_mean(c)
    print("  %-8s %.4f" % (c, sm))

chosen = {}
for s in SC:
    cands = []
    for c in cfg_names:
        v = float(np.mean([q[c] for q in Q if q["scene"] != s]))
        cands.append((v, c))
    cands.sort(key=lambda x: (-x[0], cfg_order(x[1])))
    chosen[s] = cands[0][1]
print("\nLOSO choices:", chosen)

by = {}
for q in Q:
    by.setdefault(q["scene"], []).append(q[chosen[q["scene"]]] - q["m"])
arrs = {s: np.array(v) for s, v in by.items()}
rng = np.random.default_rng(SEED)
boot = []
for _ in range(B_BOOT):
    acc = []
    for s, a in arrs.items():
        acc.extend(a[rng.integers(0, len(a), len(a))].tolist())
    boot.append(np.mean(acc))
allv = np.concatenate(list(arrs.values()))
delta = float(allv.mean())
lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

per_chosen = {}
for q in Q:
    per_chosen.setdefault(q["scene"], []).append(q[chosen[q["scene"]]])
sm_chosen = float(np.mean([np.mean(v) for v in per_chosen.values()]))
per_scene_delta = {s: float(np.mean(per_chosen[s]) - per_main[s]) for s in SC}
print("LOSO held-out: main %.4f -> v2 %.4f, delta %+.4f [%.4f, %.4f]"
      % (sm_main, sm_chosen, delta, lo, hi))
print("per-scene delta:", {k: round(v, 4) for k, v in per_scene_delta.items()})

smC, perC = scene_mean("c")
print("current C: %.4f" % smC)

fams = {}
for q in Q:
    fams.setdefault(q["cat"], []).append(q)
fam_delta = {}
for f_, sub in sorted(fams.items()):
    d = float(np.mean([x[chosen[x["scene"]]] for x in sub]) - np.mean([x["m"] for x in sub]))
    fam_delta[f_] = d
    print("  %-11s n=%3d %.3f -> %.3f (%+.3f)"
          % (f_, len(sub), np.mean([x["m"] for x in sub]),
             np.mean([x[chosen[x["scene"]]] for x in sub]), d))

gates = [
    ("delta >= +0.03 & CI_low > 0", delta >= 0.03 and lo > 0),
    ("no scene worse than -0.01", min(per_scene_delta.values()) >= -0.01),
    ("found not worse by >0.005", fam_delta.get("found", 0) >= -0.005),
    ("beats current C by >= +0.005", (sm_chosen - smC) >= 0.005),
]
print("\ngate:")
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")