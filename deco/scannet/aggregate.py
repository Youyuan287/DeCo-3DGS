import glob
import json
import os
from collections import defaultdict

ROOT = os.environ.get("RIVERMIND_RESULTS", "results")
files = sorted(glob.glob(ROOT + "/**/*.json", recursive=True))
arms = defaultdict(list)
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if isinstance(d, dict) and "center" in d and "subsets" in d:
        arms[d["center"]].append((f, d))
for k, v in arms.items():
    print("arm %-10s files %d  e.g. %s" % (k, len(v), os.path.basename(v[0][0])))


def agg(items, tag):
    inter = defaultdict(float)
    union = defaultdict(float)
    gt = defaultdict(float)
    for _, d in items:
        pc = d["subsets"][tag]["per_class"]
        for c, r in pc.items():
            inter[c] += r["inter"]
            union[c] += r["union"]
            gt[c] += r["gt"]
    rows = {c: (inter[c] / union[c], inter[c] / gt[c] if gt[c] else 0.0, gt[c]) for c in union}
    present = [c for c in rows if gt[c] > 0]
    miou = sum(rows[c][0] for c in present) / len(present)
    macc = sum(rows[c][1] for c in present) / len(present)
    return rows, miou, macc, present


for arm in ("both_lse", "none", "mean", "lse"):
    if arm not in arms:
        continue
    rows, miou, macc, present = agg(arms[arm], "19")
    print("\n=== arm %s  K19 dataset mIoU %.4f  mAcc %.4f  present %d ===" % (arm, miou * 100, macc * 100, len(present)))
    for c in present:
        i, a, g = rows[c]
        print("  %-16s iou %6.2f  acc %6.2f  gt %d" % (c, i * 100, a * 100, int(g)))
