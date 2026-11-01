"""Join text-only multi-granularity selection with the GT oracle gate."""
import csv, glob, os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "20261101", "mg_gate2")
S = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "20261101", "mg_select")
orac = {}
for f in glob.glob(os.path.join(D, "*_mg.csv")):
    for r in csv.DictReader(open(f)):
        orac[(r["scene"], r["image"], r["prompt"])] = r
sel = []
for f in glob.glob(os.path.join(S, "*_mgselect.csv")):
    sel += list(csv.DictReader(open(f)))
for r in sel:
    o = orac[(r["scene"], r["image"], r["prompt"])]
    for k in ["best3", "best_all", "best_union", "best_level"]:
        r["o_" + k] = o[k]

PART = {"pirate hat", "tesla door handle", "bear nose", "hooves", "onion segments",
        "hand", "dall-e brand", "porcelain hand", "ottolenghi"}
SET = {"three cookies", "chopsticks", "hooves", "onion segments",
       "Stainless steel pots", "rubber duck with hat", "rubber duck with buoy",
       "glass of water", "tea in a glass", "bag of cookies"}
ATTR = {"green apple", "red apple", "green toy chair", "red toy chair",
        "pink ice cream", "dark cup", "red cup", "frog cup", "yellow desk",
        "yellow pouf", "Stainless steel pots"}


def group(r):
    p = r["prompt"]
    if p in PART and p in SET:
        return "part+set"
    if p in PART:
        return "part"
    if p in SET:
        return "set"
    if p in ATTR:
        return "attr"
    return "whole"


for r in sel:
    r["grp"] = group(r)


def stats(name, sub):
    if not sub:
        return
    def m(k):
        return np.mean([float(r[k]) for r in sub])
    t1, t3, t5, t3n, t5n = m("top1_iou"), m("top3_iou"), m("top5_iou"), m("top3n_iou"), m("top5n_iou")
    b3, ba, bu = m("o_best3"), m("o_best_all"), m("o_best_union")
    print(f"{name:10s} n={len(sub):3d}  top1={t1:.3f} top3={t3:.3f} top5={t5:.3f} "
          f"top3n={t3n:.3f} top5n={t5n:.3f} | oracle L3={b3:.3f} all={ba:.3f} union={bu:.3f} | "
          f"top5>=.5={np.mean([float(r['top5_iou'])>=.5 for r in sub])*100:5.1f}% "
          f"top5>=.8={np.mean([float(r['top5_iou'])>=.8 for r in sub])*100:5.1f}%")


print("== text-only selection vs oracle ==")
for g in ["part", "set", "part+set", "attr", "whole", "ALL"]:
    sub = sel if g == "ALL" else [r for r in sel if r["grp"] == g]
    stats(g, sub)

print("\n== top1 level match ==")
for g in ["part+set", "whole", "ALL"]:
    sub = sel if g == "ALL" else [r for r in sel if r["grp"] == g]
    same = sum(1 for r in sub if int(r["top1_level"]) == int(r["o_best_level"]))
    print(g, f"{same}/{len(sub)} = {same/len(sub)*100:.1f}%")

print("\n== worst part+set ==")
for r in sorted([r for r in sel if r["grp"] in ("part", "set", "part+set")],
                key=lambda x: float(x["top5_iou"]))[:15]:
    print(f"{r['scene']:13s} {r['prompt']:22s} top1={float(r['top1_iou']):.3f} "
          f"top5={float(r['top5_iou']):.3f} oracle={float(r['o_best_all']):.3f} "
          f"union={float(r['o_best_union']):.3f}")