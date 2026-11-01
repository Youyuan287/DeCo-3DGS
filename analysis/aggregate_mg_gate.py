"""Aggregate the 2D multi-granularity gate, split by query type."""
import csv, glob, os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "20261101", "mg_gate2")
rows = []
for f in glob.glob(os.path.join(D, "*_mg.csv")):
    rows += list(csv.DictReader(open(f)))

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


for r in rows:
    r["grp"] = group(r)
    for k in ["best0", "best1", "best2", "best3", "best_all", "best_union"]:
        r[k] = float(r[k])


def stats(name, sub):
    n = len(sub)
    if not n:
        return
    b3 = np.array([r["best3"] for r in sub])
    ba = np.array([r["best_all"] for r in sub])
    bu = np.array([r["best_union"] for r in sub])
    print(f"{name:10s} n={n:3d}  best3={b3.mean():.3f} best_all={ba.mean():.3f} "
          f"union={bu.mean():.3f}  gain_all={np.mean(ba-b3):+.3f} gain_union={np.mean(bu-ba):+.3f}  "
          f"best3>=.5={np.mean(b3>=.5)*100:5.1f}% all>=.5={np.mean(ba>=.5)*100:5.1f}% "
          f"union>=.5={np.mean(bu>=.5)*100:5.1f}%  union>=.8={np.mean(bu>=.8)*100:5.1f}%  "
          f"n_union={np.mean([int(r['n_union']) for r in sub]):.2f}")


print("== by query group ==")
for g in ["part", "set", "part+set", "attr", "whole", "ALL"]:
    sub = rows if g == "ALL" else [r for r in rows if r["grp"] == g]
    stats(g, sub)

print("\n== per-level best (mean) ==")
for g in ["part", "set", "part+set", "whole", "ALL"]:
    sub = rows if g == "ALL" else [r for r in rows if r["grp"] == g]
    print(g, " ".join(f"L{k}={np.mean([r['best'+str(k)] for r in sub]):.3f}" for k in range(4)))

print("\n== best level distribution (ALL / part+set) ==")
for g in ["ALL", "part+set"]:
    sub = rows if g == "ALL" else [r for r in rows if r["grp"] == g]
    cnt = {k: sum(1 for r in sub if int(r["best_level"]) == k) for k in range(4)}
    print(g, cnt)

# hard part queries detail
print("\n== part detail ==")
for r in sorted([r for r in rows if r["grp"] == "part"], key=lambda x: x["prompt"]):
    print(f"{r['scene']:13s} {r['prompt']:22s} best3={r['best3']:.3f} all={r['best_all']:.3f} "
          f"union={r['best_union']:.3f} level={r['best_level']} n_union={r['n_union']}")

print("\n== set detail ==")
for r in sorted([r for r in rows if r["grp"] == "set"], key=lambda x: x["prompt"]):
    print(f"{r['scene']:13s} {r['prompt']:22s} best3={r['best3']:.3f} all={r['best_all']:.3f} "
          f"union={r['best_union']:.3f} level={r['best_level']} n_union={r['n_union']}")