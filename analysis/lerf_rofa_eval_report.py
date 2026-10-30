"""Evaluate the ROFA end-to-end experiment (PREREG_rofa.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
VAR = ["main", "raw", "rofa", "rofac"]

iou = {}
for s in SC:
    with open(os.path.join(R, "lerf_identity_20261030", "%s_identity.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iou[(s, r["image"], r["prompt"], int(r["rank"]))] = float(r["iou"])
s0 = {}
for s in SC:
    with open(os.path.join(R, "lerf_pruneinclude_20261030", "%s_pruneinclude.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s0[(s, r["image"], r["prompt"])] = float(r["s0_iou"])

Q = {}
for s in SC:
    with open(os.path.join(R, "lerf_rofa_20261031", "%s_rofa.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            Q[(s, r["image"], r["prompt"])] = r

keys = [k for k in Q if k in s0]
print("queries", len(keys))
bad = [k for k in keys if abs(float(Q[k]["main_iou"]) - s0[k]) > 1e-6]
print("main parity mismatches", len(bad), bad[:3])

fam, best_iou, best_rank = {}, {}, {}
for k in keys:
    rows = [(r, iou.get((k[0], k[1], k[2], r))) for r in range(1, 40)]
    rows = [(r, v) for r, v in rows if v is not None]
    if not rows:
        continue
    bi = max(rows, key=lambda x: x[1])
    best_rank[k], best_iou[k] = bi[0], bi[1]
    fam[k] = "recall" if bi[1] < 0.3 else ("found" if bi[0] == 1 else ("rank2" if bi[0] == 2 else "deep"))

keys = [k for k in keys if k in fam]
print("labelled", len(keys), {f: sum(1 for k in keys if fam[k] == f) for f in ["found", "rank2", "deep", "recall"]})

val = {v: {k: float(Q[k][v + "_iou"]) for k in keys} for v in VAR}
per_scene = {v: {s: np.array([val[v][k] for k in keys if k[0] == s]) for s in SC} for v in VAR}

print("\n%-6s %9s %9s %9s %9s %9s %9s" % ("var", "scene-m", "found", "rank2", "deep", "recall", "wins_vs_main"))
print("%-6s %9.4f %9.3f %9.3f %9.3f %9.3f %9s" % ("main", np.mean([per_scene["main"][s].mean() for s in SC]),
      np.mean([val["main"][k] for k in keys if fam[k] == "found"]),
      np.mean([val["main"][k] for k in keys if fam[k] == "rank2"]),
      np.mean([val["main"][k] for k in keys if fam[k] == "deep"]),
      np.mean([val["main"][k] for k in keys if fam[k] == "recall"]), "-"))
for v in VAR[1:]:
    w = sum(1 for k in keys if val[v][k] > val["main"][k] + 1e-9)
    l = sum(1 for k in keys if val[v][k] < val["main"][k] - 1e-9)
    print("%-6s %9.4f %9.3f %9.3f %9.3f %9.3f %5d/%d" % (v,
          np.mean([per_scene[v][s].mean() for s in SC]),
          np.mean([val[v][k] for k in keys if fam[k] == "found"]),
          np.mean([val[v][k] for k in keys if fam[k] == "rank2"]),
          np.mean([val[v][k] for k in keys if fam[k] == "deep"]),
          np.mean([val[v][k] for k in keys if fam[k] == "recall"]), w, l))

print("\nper-scene")
print("%-14s %8s %8s %8s %8s" % ("scene", "main", "raw", "rofa", "rofac"))
for s in SC:
    print("%-14s %8.4f %8.4f %8.4f %8.4f" % (s, per_scene["main"][s].mean(), per_scene["raw"][s].mean(),
          per_scene["rofa"][s].mean(), per_scene["rofac"][s].mean()))

rng = np.random.default_rng(20261030)
B = 2000
boot = {}
for v in VAR[1:]:
    dm = []
    for bb in range(B):
        ds = []
        for s in SC:
            ks = [k for k in keys if k[0] == s]
            pick = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
            ds.append(np.mean([val[v][k] - val["main"][k] for k in pick]))
        dm.append(np.mean(ds))
    dm = np.array(dm)
    boot[v] = (dm.mean(), np.percentile(dm, 2.5), np.percentile(dm, 97.5))

print("\npaired scene-stratified bootstrap delta vs main (B=%d, seed=20261030)" % B)
for v in VAR[1:]:
    print("  %-6s delta=%+0.4f  95%%CI [%+0.4f, %+0.4f]" % (v, *boot[v]))

print("\nROFA diagnostics")
for v in ("rofa", "rofac"):
    denom_views = sum(int(Q[k]["total_views"]) for k in keys)
    rem = sum(int(Q[k]["rem_" + v]) for k in keys)
    cand = sum(int(Q[k]["n"]) for k in keys)
    qrem = sum(int(Q[k]["qv_rem_" + v]) for k in keys)
    print("  %-6s views=%d removed=%.3f%%  query-view-removed=%d/%d candidates (%.2f%%)" %
          (v, denom_views, 100.0 * rem / denom_views, qrem, cand, 100.0 * qrem / cand))

print("\nGate (primary = rofac)")
d, lo, hi = boot["rofac"]
scene_ok = all(per_scene["rofac"][s].mean() >= per_scene["main"][s].mean() - 0.01 for s in SC)
found_ok = np.mean([val["rofac"][k] for k in keys if fam[k] == "found"]) >= np.mean(
    [val["main"][k] for k in keys if fam[k] == "found"]) - 0.005
gates = [("delta >= +0.01", d >= 0.01), ("CI lower > 0", lo > 0), ("no scene worse than -0.01", scene_ok),
         ("found not worse by >0.005", found_ok)]
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")