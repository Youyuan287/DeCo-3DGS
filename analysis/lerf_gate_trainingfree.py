"""Offline training-free gate audit on the existing gateE CSVs.

Reconstructs S0 (released gap prefix) and S_M (mask rerank) from the segidentity
dump, then evaluates:
  gateF : S_M if (S_M != S0 and e_M > e_0) else S0    (training-free; e_* = max id_top3)
  gateE : as run on the server (uses learned S_C)      (verification only)

IoUs are taken from the gate CSV's main_iou / mask_iou columns, so no rendering.
Sanity check: reconstructed e_m must equal the CSV's e_m for every row.
"""
import csv, glob, os, statistics, sys

GATE = sys.argv[1] if len(sys.argv) > 1 else "analysis/results/20261031/gatee"
SEG = sys.argv[2] if len(sys.argv) > 2 else "analysis/results/20261031/diagnostics"
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else "image"
LAM = 1.0


def zs(x):
    m = sum(x) / len(x)
    v = sum((a - m) ** 2 for a in x) / len(x)
    s = v ** 0.5
    return [(a - m) / s for a in x] if s > 1e-12 else [0.0] * len(x)


def load_seg(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["image"], r["prompt"], int(r["rank"]))] = r
    return rows


def main():
    per_scene = {}
    bad = 0
    trig = {"gateF": 0}
    for gf in sorted(glob.glob(os.path.join(GATE, "*_cmaskgate_%s.csv" % SUFFIX))):
        sc = os.path.basename(gf).split("_cmaskgate")[0]
        seg = load_seg(os.path.join(SEG, "%s_segidentity.csv" % sc))
        rows = list(csv.DictReader(open(gf)))
        per_img = {}
        for r in rows:
            name, prompt = r["image"], r["prompt"]
            n = int(r["n"])
            g = int(r["gap_len"])
            sc_l, mm_l = [], []
            for i in range(n):
                d = seg[(name, prompt, i + 1)]
                sc_l.append(float(d["score"]))
                v = d.get("id_top3", "")
                mm_l.append(float(v) if v != "" else float("nan"))
            fin = [x for x in mm_l if x == x]
            fill = sum(fin) / len(fin) if fin else 0.0
            mm = [x if x == x else fill for x in mm_l]
            S0 = list(range(g)) if g > 0 else []
            S = [a + LAM * b for a, b in zip(zs(sc_l), zs(mm))]
            order = sorted(range(n), key=lambda i: (-S[i], i))
            SM = order[:g] if g > 0 else []
            e0 = max([mm[i] for i in S0], default=-1e9) if S0 else -1e9
            eM = max([mm[i] for i in SM], default=-1e9) if SM else -1e9
            # sanity: reconstructed e_m vs server row
            if abs(eM - float(r["e_m"])) > 1e-6:
                bad += 1
            cond = set(SM) != set(S0) and eM > e0
            if cond:
                trig["gateF"] += 1
            iou = float(r["mask_iou"]) if cond else float(r["main_iou"])
            per_img.setdefault(name, []).append((iou, float(r["main_iou"]), float(r["mask_iou"])))
        scene_scores = {}
        for t, idx in (("main", 1), ("mask", 2), ("gateF", 0)):
            scene_scores[t] = statistics.mean(
                statistics.mean(x[idx] for x in xs) for xs in per_img.values())
        scene_scores["n"] = sum(len(v) for v in per_img.values())
        per_scene[sc] = scene_scores
    print("| scene | n | main | mask | gateF | gateF-main |")
    for sc, d in per_scene.items():
        print("| %s | %d | %.4f | %.4f | %.4f | %+.4f |" % (
            sc, d["n"], d["main"], d["mask"], d["gateF"], d["gateF"] - d["main"]))
    for t in ("main", "mask", "gateF"):
        m = statistics.mean(d[t] for d in per_scene.values())
        print("MACRO %-6s %.4f" % (t, m))
    print("e_m reconstruction mismatches:", bad, "/", sum(d["n"] for d in per_scene.values()))
    print("gateF triggered on %d/208 queries" % trig["gateF"])


if __name__ == "__main__":
    main()