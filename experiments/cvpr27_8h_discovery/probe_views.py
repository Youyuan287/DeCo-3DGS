"""Does the per-view DOMINANCE PATTERN pick the better candidate? (training-free)

Mean evidence was already tried and failed. This tests statistics of the
candidate-vs-rival margin per view: how often a candidate wins each view, and by
how much, rather than its average similarity.
"""
import csv, glob, os
import numpy as np

W = "analysis/results/20261101/witness"
G = "analysis/results/20261101/cnone_gate_tf"

wit = {}
for f in glob.glob(os.path.join(W, "*_witness.csv")):
    sc = os.path.basename(f).split("_witness")[0]
    for r in csv.DictReader(open(f)):
        k = (sc, r["image"], r["prompt"])
        wit.setdefault(k, {}).setdefault(int(r["rank"]), {})[int(r["view_id"])] = float(r["sim"])

iou = {}
for f in glob.glob(os.path.join(G, "*_segidentity.csv")):
    for r in csv.DictReader(open(f)):
        iou[(r["scene"], r["image"], r["prompt"], int(r["rank"]))] = float(r["iou"])

def score(sel):
    per = {}
    for (s, im, pr), rk in sel.items():
        per.setdefault(s, {}).setdefault(im, []).append(iou.get((s, im, pr, rk), np.nan))
    return float(np.nanmean([np.nanmean([np.nanmean(v) for v in d.values()]) for d in per.values()]))

RULES = {
    "rank1": lambda st: min(st),
    "max win-views": lambda st: max(st, key=lambda r: st[r]["win"]),
    "max win-rate>0.1": lambda st: max(st, key=lambda r: st[r]["win10"]),
    "max mean margin": lambda st: max(st, key=lambda r: st[r]["mean"]),
    "max median margin": lambda st: max(st, key=lambda r: st[r]["med"]),
    "max min margin": lambda st: max(st, key=lambda r: st[r]["min"]),
    "max lower-quartile margin": lambda st: max(st, key=lambda r: st[r]["q25"]),
    "max margin x win-rate": lambda st: max(st, key=lambda r: st[r]["mean"] * st[r]["rate"]),
}
sel = {k: {} for k in RULES}
for k, per_rank in wit.items():
    ranks = sorted(per_rank)
    if len(ranks) < 2:
        continue
    views = sorted(set().union(*[set(per_rank[r]) for r in ranks]))
    st = {}
    for r in ranks:
        wins, mg = 0, []
        for v in views:
            s = per_rank[r].get(v)
            if s is None:
                continue
            other = max([per_rank[o].get(v, -9) for o in ranks if o != r] or [-9])
            mg.append(s - other)
            if s > other:
                wins += 1
        mg = np.array(mg) if mg else np.array([-9.0])
        st[r] = dict(win=wins, win10=float((mg > 0.1).sum()),
                     mean=float(mg.mean()), med=float(np.median(mg)),
                     min=float(mg.min()), q25=float(np.percentile(mg, 25)),
                     rate=float((mg > 0).mean()))
    for name, fn in RULES.items():
        try:
            sel[name][k] = fn(st)
        except Exception:
            sel[name][k] = 1
print("queries with witness: %d" % len(sel["rank1"]))
for name in RULES:
    print("%-26s %.4f" % (name, score(sel[name])))
