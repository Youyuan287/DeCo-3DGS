#!/usr/bin/env python3
"""Summarize the 67.60 package: macro / per-scene / per-family / bootstrap CI."""
import argparse, csv, os
import numpy as np

SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gatee")
    ap.add_argument("--identity", default="inputs/identity")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    Q = {}
    for sc in SC:
        path = os.path.join(args.results, "%s_cmaskgate_image.csv" % sc)
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            Q[(sc, r["image"], r["prompt"])] = {
                t: float(r[t + "_iou"]) for t in ["main", "c", "mask", "gatee"]}

    best = {}
    for sc in SC:
        path = os.path.join(args.identity, "%s_identity.csv" % sc)
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            k = (sc, r["image"], r["prompt"]); v = float(r["iou"]); rk = int(r["rank"])
            if k not in best or v > best[k][1]:
                best[k] = (rk, v)
    fam = {k: ("recall" if v[1] < 0.3 else ("found" if v[0] == 1 else
              ("rank2" if v[0] == 2 else "deep"))) for k, v in best.items()}

    def per_image(tag):
        out = {}
        for sc in SC:
            by = {}
            for k, v in Q.items():
                if k[0] == sc:
                    by.setdefault(k[1], []).append(v[tag])
            out[sc] = float(np.mean([np.mean(x) for x in by.values()]))
        return out

    def macro(tag):
        return float(np.mean(list(per_image(tag).values())))

    print("%-8s %8s %8s %8s %8s" % ("scene", "main", "C", "mask", "gateE"))
    for sc in SC:
        vals = {t: float(np.mean([v[t] for k, v in Q.items() if k[0] == sc])) for t in
                ["main", "c", "mask", "gatee"]}
        # per-image scene means
        pi = {t: per_image(t)[sc] for t in ["main", "c", "mask", "gatee"]}
        print("%-8s %8.4f %8.4f %8.4f %8.4f" % (sc, pi["main"], pi["c"], pi["mask"], pi["gatee"]))
    print("%-8s %8.4f %8.4f %8.4f %8.4f" % ("macro", macro("main"), macro("c"), macro("mask"), macro("gatee")))

    rng = np.random.default_rng(args.seed)
    def boot(a, b):
        # paired scene-stratified bootstrap over images (protocol is per-image mean)
        by_img = {}
        for sc in SC:
            d = {}
            for (s, im, pr), v in Q.items():
                if s == sc:
                    d.setdefault(im, {t: [] for t in ["main", "c", "mask", "gatee"]})
                    for t in d[im]:
                        d[im][t].append(v[t])
            by_img[sc] = {im: {t: float(np.mean(x)) for t, x in tv.items()}
                          for im, tv in d.items()}
        dm = []
        for _ in range(args.B):
            per = []
            for sc in SC:
                ks = list(by_img[sc].keys())
                pick = [ks[i] for i in rng.integers(0, len(ks), len(ks))]
                per.append(np.mean([by_img[sc][k][a] - by_img[sc][k][b] for k in pick]))
            dm.append(np.mean(per))
        return float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))

    for a, b in [("gatee", "main"), ("gatee", "c"), ("gatee", "mask")]:
        lo, hi = boot(a, b)
        print("%s - %s: %+.4f  CI [%+.4f, %+.4f]" % (a, b, macro(a) - macro(b), lo, hi))

    print("\nfamily (query-level mean IoU):")
    for f in ["found", "rank2", "deep", "recall"]:
        ks = [k for k in Q if fam.get(k) == f]
        print("  %-6s n=%3d " % (f, len(ks)) + " ".join(
            "%s=%.4f" % (t, np.mean([Q[k][t] for k in ks])) for t in ["main", "c", "mask", "gatee"]))


if __name__ == "__main__":
    main()