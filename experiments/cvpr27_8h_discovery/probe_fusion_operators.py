"""E14: do the three fusion operators differ as SELECTORS on identical data?

a) feature-mean score          (NAG score: query posed on the merged feature)
b) per-view mean evidence      (top-3 mean of per-view sims)      -- early fusion
c) paired per-view margin      (mean of c-vs-strongest-rival)     -- decision fusion
d) per-view win rate           (fraction of views where c is argmax)
Same candidates, single-candidate macro, no rendering.
"""
import csv, glob, os
import numpy as np

W = "analysis/results/20261101/witness"
G = "analysis/results/20261101/cnone_gate_tf"

wit = {}
for f in glob.glob(os.path.join(W, "*_witness.csv")):
    sc = os.path.basename(f).split("_witness")[0]
    for r in csv.DictReader(open(f)):
        wit.setdefault((sc, r["image"], r["prompt"]), {}).setdefault(
            int(r["rank"]), {})[int(r["view_id"])] = float(r["sim"])

iou, scr = {}, {}
for f in glob.glob(os.path.join(G, "*_segidentity.csv")):
    for r in csv.DictReader(open(f)):
        k = (r["scene"], r["image"], r["prompt"], int(r["rank"]))
        iou[k] = float(r["iou"])
        scr[k] = float(r["score"])

def macro(sel):
    per = {}
    for k, rk in sel.items():
        per.setdefault(k[0], {}).setdefault(k[1], []).append(iou[(k[0], k[1], k[2], rk)])
    return float(np.mean([np.mean([np.mean(v) for v in d.values()]) for d in per.values()]))

qs = []
for k, pr in wit.items():
    ranks = sorted(pr)
    if len(ranks) >= 2:
        qs.append((k, ranks, pr))
print("queries:", len(qs))

def top3(d):
    v = sorted(d.values(), reverse=True)[:3]
    return float(np.mean(v)) if v else -9.0

def margin(r, pr, ranks):
    mg = []
    for v, s in pr[r].items():
        other = max([pr[o].get(v, -9) for o in ranks if o != r] or [-9])
        mg.append(s - other)
    return float(np.mean(mg)) if mg else -9.0

def winrate(r, pr, ranks):
    w = n = 0
    for v, s in pr[r].items():
        n += 1
        if s >= max([pr[o].get(v, -9) for o in ranks if o != r] or [-9]):
            w += 1
    return w / n if n else 0.0

fn = {
    "a) feature-mean score": lambda r, pr, rs, k: scr.get((k[0], k[1], k[2], r), -9),
    "b) per-view mean (top3)": lambda r, pr, rs, k: top3(pr[r]),
    "c) paired margin": lambda r, pr, rs, k: margin(r, pr, rs),
    "d) per-view win rate": lambda r, pr, rs, k: winrate(r, pr, rs),
    "oracle": None,
}
for name, f in fn.items():
    if f is None:
        print("%-26s %.4f" % (name, float(np.mean([max(iou[(k[0],k[1],k[2],r)] for r in rs)
                                                   for k, rs, _ in qs]))))
        continue
    sel = {k: max(rs, key=lambda r: f(r, pr, rs, k)) for k, rs, pr in qs}
    print("%-26s %.4f" % (name, macro(sel)))
