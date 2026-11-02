"""Build a_g-shifted witness / segidentity dirs for the LERF certificate (N1).

shift_q(r) = alpha * sd_q * a1z[root(r)]
    sd_q  = std of the query's witness sims (same scale convention as E3)
    root  = level-3 region id from segidentity.csv

Writes:
  <out>/witness_ag/<scene>_witness.csv          sim  -> sim - shift
  <out>/seg_ag/<scene>_segidentity.csv          score -> score - shift
"""
import csv, os, sys, glob
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "analysis/results/20261101"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis/results/20261103_axis/n1"
AG = sys.argv[3] if len(sys.argv) > 3 else "analysis/results/20261103_axis/ag"
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
KEY = sys.argv[5] if len(sys.argv) > 5 else "a1z"

os.makedirs(os.path.join(OUT, "witness_ag"), exist_ok=True)
os.makedirs(os.path.join(OUT, "seg_ag"), exist_ok=True)

for sc in ["figurines", "ramen", "teatime", "waldo_kitchen"]:
    ag = np.load(os.path.join(AG, "%s_ag.npz" % sc))[KEY]
    seg_p = os.path.join(SRC, "cnone_gate_tf", "%s_segidentity.csv" % sc)
    wit_p = os.path.join(SRC, "witness", "%s_witness.csv" % sc)

    root_of, score = {}, {}
    seg_rows = list(csv.DictReader(open(seg_p)))
    for r in seg_rows:
        k = (r["image"], r["prompt"], int(r["rank"]))
        root_of[k] = int(r["root_id"])
        score[k] = float(r["score"])
    bad = sum(1 for v in root_of.values() if v >= len(ag))
    print("%-14s ranks %4d  max_root %3d  n_regions %4d  out_of_range %d" % (
        sc, len(root_of), max(root_of.values()), len(ag), bad))

    wit_rows = list(csv.DictReader(open(wit_p)))
    per_q = {}
    for r in wit_rows:
        per_q.setdefault((r["image"], r["prompt"]), []).append(float(r["sim"]))
    sd = {k: float(np.std(v)) for k, v in per_q.items()}
    shift = {k: ALPHA * sd.get((k[0], k[1]), 0.0) * ag[root_of[k]]
             for k in root_of if root_of[k] < len(ag)}

    with open(os.path.join(OUT, "witness_ag", "%s_witness.csv" % sc), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "prompt", "rank", "view_id", "sim"])
        for r in wit_rows:
            k = (r["image"], r["prompt"], int(r["rank"]))
            w.writerow([r["image"], r["prompt"], r["rank"], r["view_id"],
                        "%.10f" % (float(r["sim"]) - shift.get(k, 0.0))])

    with open(os.path.join(OUT, "seg_ag", "%s_segidentity.csv" % sc), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seg_rows[0].keys())
        w.writeheader()
        for r in seg_rows:
            k = (r["image"], r["prompt"], int(r["rank"]))
            r = dict(r)
            r["score"] = "%.10f" % (score[k] - shift.get(k, 0.0))
            w.writerow(r)
    print("   wrote shifted witness + seg for", sc)
print("DONE")