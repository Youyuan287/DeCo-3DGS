"""Dataset-level aggregation for the ScanNet run.

Reads the per-scene JSON written by `score_scannet.py`, sums intersection / union per class over the
ten scenes, and averages over the classes that have ground truth.

`bookshelf` (NYU40 id 10) has zero annotated points in all ten scenes, so the K19 divisor is **18**.
Pass `--divisor 19` to count the absent class as a zero instead (58.55 -> 55.47, 40.79 -> 38.64).

Both protocols are printed, because the literature uses both:
    pooled   sum inter/union over scenes, then average over present classes   (ReLaGS-style, 40.79)
    macro    per scene average over that scene's present classes, then mean   (OpenGaussian-style, 39.94)

Usage:
    python aggregate.py --results "$SCANNET_OUT"
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load(results):
    """[(renorm, center, scene, dict)] for every per-scene JSON in the directory."""
    out = []
    for f in sorted(glob.glob(os.path.join(results, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not (isinstance(d, dict) and "subsets" in d):
            continue
        out.append((int(d.get("renorm", -1)), d.get("center", "?"), d.get("scene", "?"), d))
    return out


def pooled(items, tag):
    inter, union, gt = defaultdict(float), defaultdict(float), defaultdict(float)
    for _, _, _, d in items:
        for c, r in d["subsets"][tag]["per_class"].items():
            inter[c] += r["inter"]
            union[c] += r["union"]
            gt[c] += r["gt"]
    return {c: (inter[c] / union[c] if union[c] else 0.0,
                inter[c] / gt[c] if gt[c] else 0.0,
                gt[c]) for c in union}


def macro(items, tag):
    """Mean over scenes of the per-scene mean over that scene's present classes."""
    per_scene = []
    for _, _, _, d in items:
        pc = d["subsets"][tag]["per_class"]
        ious = [r["inter"] / r["union"] for r in pc.values() if r["union"] and r["gt"] > 0]
        if ious:
            per_scene.append(sum(ious) / len(ious))
    return sum(per_scene) / len(per_scene) if per_scene else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default=os.environ.get("SCANNET_OUT",
                                           os.path.join(os.environ.get("RIVERMIND_RESULTS", "results"),
                                                        "scannet_k19")),
                    help="directory written by score_scannet.py")
    ap.add_argument("--subset", default="19", help="SUBSETS key: 19 / 15 / 10")
    ap.add_argument("--divisor", type=int, default=0,
                    help="0 = classes that actually have GT (K19 -> 18); or 19")
    ap.add_argument("--scenes", default="", help="comma list to restrict (sanity checks)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    items = load(a.results)
    if a.scenes:
        keep = {s.strip() for s in a.scenes.split(",") if s.strip()}
        items = [x for x in items if x[2] in keep]
    if not items:
        raise SystemExit("no per-scene JSON under %s" % a.results)

    groups = defaultdict(list)
    for rn, center, _, d in items:
        groups[(center, rn)].append((rn, center, d.get("scene", "?"), d))

    for (center, rn), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows = pooled(g, a.subset)
        present = [c for c in rows if rows[c][2] > 0]
        div = a.divisor or len(present)
        miou_p = sum(rows[c][0] for c in present) / div
        macc_p = sum(rows[c][1] for c in present) / div
        miou_macro = macro(g, a.subset)
        flag = {0: "   <- unified (58.55)", 1: "   <- renormalised (57.85)"}.get(rn, "")
        print("\n=== center=%s  renorm=%d  scenes=%d  K%s  divisor=%d%s"
              % (center, rn, len(g), a.subset, div, flag))
        print("    pooled  mIoU %6.2f   mAcc %6.2f" % (100 * miou_p, 100 * macc_p))
        print("    macro   mIoU %6.2f" % (100 * miou_macro,))
        if a.verbose:
            for c in sorted(present, key=lambda c: -rows[c][2]):
                i, acc, gt = rows[c]
                print("    %-16s iou %6.2f  acc %6.2f  gt %d" % (c, 100 * i, 100 * acc, int(gt)))


if __name__ == "__main__":
    main()