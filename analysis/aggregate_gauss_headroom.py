"""Aggregate Gaussian-level headroom inside ReLaGS selections."""
import csv, glob, os, collections
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "20261101", "gauss_headroom3")
C = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro\lerf_selector_20261030"
rows = []
for f in glob.glob(os.path.join(D, "*_gaussheadroom.csv")):
    rows += list(csv.DictReader(open(f)))

best = {}
for f in glob.glob(os.path.join(C, "*_candidates.csv")):
    g = collections.defaultdict(list)
    for r in csv.DictReader(open(f)):
        g[(r["scene"], r["image"], r["prompt"])].append(r)
    for k, v in g.items():
        j = max(range(len(v)), key=lambda i: float(v[i]["iou"]))
        best[k] = (float(v[j]["iou"]), int(v[j]["rank"]))


def fam(r):
    bi, br = best[(r["scene"], r["image"], r["prompt"])]
    if bi < 0.3:
        return "recall"
    if br == 1:
        return "found"
    if br == 2:
        return "rank2"
    return "deep"


for r in rows:
    r["fam"] = fam(r)


def col(k):
    return np.array([float(r[k]) for r in rows if r.get(k) not in ("", "nan")])


def stats(name, x, fmt=".4f"):
    x = x[np.isfinite(x)]
    q = np.percentile(x, [10, 50, 90])
    print(f"{name:30s} n={len(x):3d} mean={x.mean():{fmt}} med={q[1]:{fmt}} "
          f"q10={q[0]:{fmt}} q90={q[2]:{fmt}}")


print("== overall ==")
stats("selected superpoints", col("n_sel_super"))
stats("selected root units", col("n_sel_root"))
stats("selected leaf units", col("n_sel_leaf"))
stats("selected Gaussians", col("n_sel_gauss"))
stats("GT Gaussians", col("n_gt_gauss"))
stats("precision", col("precision"))
stats("recall", col("recall"))
stats("IoU3D", col("iou3d"))
stats("IoU after boundary prune", col("iou_after_boundary_prune"))
stats("IoU leaf split", col("iou_leaf_split"))
stats("root GT recall", col("n_gt_in_sel_roots") / col("n_gt_gauss"))
stats("GT outside selected roots", col("n_gt_out_sel_roots") / col("n_gt_gauss"))
stats("boundary selected", col("n_boundary_sel"))
stats("boundary FP selected", col("n_boundary_sel_fp"))
stats("boundary FP / all FP", col("n_boundary_sel_fp") / np.maximum(col("n_fp"), 1))
stats("all-pool precision", col("all_precision"))
stats("all-pool recall", col("all_recall"))
stats("all-pool IoU3D", col("all_iou"))
stats("best-prefix IoU3D", col("best_pref_iou"))
stats("best-single IoU3D", col("best_single_iou"))

g = np.array([int(r["n_sel_super"]) for r in rows])
print("selected-superpoint dist", dict(zip(*np.unique(g, return_counts=True))),
      "<=3", f"{(g<=3).mean()*100:.1f}%")

# paired oracle gains
iou = np.array([float(r["iou3d"]) for r in rows])
bp = np.array([float(r["iou_after_boundary_prune"]) for r in rows])
ls = np.array([float(r["iou_leaf_split"]) for r in rows])
ok = np.isfinite(iou) & np.isfinite(bp)
print("\nboundary prune gain mean %.4f med %.4f >.01 %.1f%%" %
      ((bp[ok] - iou[ok]).mean(), np.median(bp[ok] - iou[ok]),
       100 * np.mean((bp[ok] - iou[ok]) > .01)))
ok = np.isfinite(iou) & np.isfinite(ls)
print("leaf split gain     mean %.4f med %.4f >.01 %.1f%% (n=%d)" %
      ((ls[ok] - iou[ok]).mean(), np.median(ls[ok] - iou[ok]),
       100 * np.mean((ls[ok] - iou[ok]) > .01), ok.sum()))

# perfect Gaussian selection inside selected roots: IoU == root recall
rr = np.array([float(r["n_gt_in_sel_roots"]) / float(r["n_gt_gauss"]) for r in rows])
print("\nheadroom decomposition (mean):")
print("  current IoU3D                 %.4f" % iou.mean())
print("  perfect within selected roots %.4f  (+%.4f)" % (rr.mean(), rr.mean() - iou.mean()))
print("  perfect all Gaussians         1.0000  (+%.4f from roots)" % (1 - rr.mean()))

all_iou = col("all_iou")
bp_iou = col("best_pref_iou")
bs_iou = col("best_single_iou")
print("\nexpanding the pool (mean IoU3D):")
print("  current gap prefix            %.4f" % iou.mean())
print("  all top-5/leaf candidates     %.4f  (%+.4f)" % (all_iou.mean(), all_iou.mean() - iou.mean()))
print("  best prefix (GT oracle)       %.4f  (%+.4f)" % (bp_iou.mean(), bp_iou.mean() - iou.mean()))
print("  best single (GT oracle)       %.4f  (%+.4f)" % (bs_iou.mean(), bs_iou.mean() - iou.mean()))
gap = np.array([int(r["gap_len"]) for r in rows])
bpk = np.array([int(r["best_pref_k"]) for r in rows])
print("  best prefix k > gap           %.1f%%; k == gap %.1f%%; k < gap %.1f%%" %
      (100 * np.mean(bpk > gap), 100 * np.mean(bpk == gap), 100 * np.mean(bpk < gap)))

print("\n== per family (mean) ==")
print(f"{'fam':8s} {'n':>3s} {'sups':>6s} {'prec':>6s} {'rec':>6s} {'iou3d':>6s} "
      f"{'rootrec':>7s} {'all5':>6s} {'bestpf':>7s} {'bndgain':>7s} {'leafgain':>8s} {'bndFP/FP':>8s}")
for f in ["found", "rank2", "deep", "recall"]:
    sub = [r for r in rows if r["fam"] == f]
    def m(k):
        x = np.array([float(r[k]) for r in sub])
        return np.nanmean(x)
    b = np.array([float(r["iou3d"]) - float(r["iou_after_boundary_prune"]) for r in sub])
    l = np.array([float(r["iou3d"]) - float(r["iou_leaf_split"]) for r in sub])
    print(f"{f:8s} {len(sub):3d} {m('n_sel_super'):6.2f} {m('precision'):6.3f} "
          f"{m('recall'):6.3f} {m('iou3d'):6.3f} "
          f"{np.nanmean([float(r['n_gt_in_sel_roots'])/float(r['n_gt_gauss']) for r in sub]):7.3f} "
          f"{m('all_iou'):6.3f} {m('best_pref_iou'):7.3f} "
          f"{np.nanmean(b):7.4f} {np.nanmean(l):8.4f} "
          f"{np.nanmean([float(r['n_boundary_sel_fp'])/max(float(r['n_fp']),1) for r in sub]):8.3f}")