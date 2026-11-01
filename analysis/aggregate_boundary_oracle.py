"""Aggregate boundary probe CSVs: IoU-selected mask vs BIoU-selected oracle."""
import csv, glob, os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "20261101", "boundary_oracle")
rows = []
for f in sorted(glob.glob(os.path.join(D, "*_liftprobe.csv"))):
    with open(f, newline="") as fh:
        rows += list(csv.DictReader(fh))
print("queries", len(rows))


def col(k):
    return np.array([float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")])


def stats(name, x):
    x = x[np.isfinite(x)]
    q = np.percentile(x, [10, 25, 50, 75, 90])
    print(f"{name:22s} n={len(x):3d} mean={x.mean():.4f} med={q[2]:.4f} "
          f"q10={q[0]:.3f} q25={q[1]:.3f} q75={q[3]:.3f} q90={q[4]:.3f} "
          f">=.95={np.mean(x>=.95)*100:5.1f}% >=.90={np.mean(x>=.90)*100:5.1f}%")


for k in ["cov2d", "cov2d_biou", "cov2d_best_biou", "cov2d_best_biou1",
          "iou_at_best_biou", "lifted_iou", "lifted_biou", "lifted_biou1"]:
    stats(k, col(k))

g = col("cov2d_best_biou") - col("cov2d_biou")
print(f"gain(best_biou - biou_of_iou_best): mean={g.mean():.4f} med={np.median(g):.4f} "
      f"q90={np.percentile(g,90):.4f} frac>0.01={np.mean(g>0.01)*100:.1f}%")

# how often the BIoU-best mask differs from the IoU-best mask (proxy: IoU gap)
di = col("cov2d") - col("iou_at_best_biou")
print(f"iou(iou_best)-iou(biou_best): mean={di.mean():.4f} med={np.median(di):.4f}")

# stratified by IoU-selected mask quality
iou_sel = col("cov2d")
b_best = col("cov2d_best_biou")
b_iou = col("cov2d_biou")
for lo in [0.0, 0.25, 0.5, 0.75]:
    m = iou_sel >= lo
    if m.sum():
        print(f"IoU>={lo:.2f}: n={m.sum():3d} biou(iou-best)={b_iou[m].mean():.4f} "
              f"biou(oracle)={b_best[m].mean():.4f} med_oracle={np.median(b_best[m]):.4f}")

# per scene
print("\nPER SCENE (mean)")
for s in sorted(set(r["scene"] for r in rows)):
    sub = [r for r in rows if r["scene"] == s]
    print(s, len(sub),
          "iou", np.mean([float(r["cov2d"]) for r in sub]),
          "biou_ioubest", np.mean([float(r["cov2d_biou"]) for r in sub]),
          "biou_oracle", np.mean([float(r["cov2d_best_biou"]) for r in sub]))