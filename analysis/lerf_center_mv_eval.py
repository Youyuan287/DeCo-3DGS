"""Stage-1 evaluation of centered multi-view candidate scoring (PREREG_center_mv.md)."""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
TAGS = ["S1_raw", "S2_raw", "S1_c", "S2_c"]

iou = {}
for s in SC:
    with open(os.path.join(R, "lerf_identity_20261030", "%s_identity.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iou[(s, r["image"], r["prompt"], int(r["rank"]))] = float(r["iou"])

Q = {}
for s in SC:
    with open(os.path.join(R, "lerf_center_mv_20261031", "%s_center_mv.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (s, r["image"], r["prompt"])
            Q.setdefault(key, []).append(dict(rank=int(r["rank"]),
                                              **{t: float(r[t]) for t in TAGS}))

res = {}
for tag in TAGS:
    per_scene, found, r2, deep, recall = {}, [], [], [], []
    for key, rows in Q.items():
        rows = sorted(rows, key=lambda x: x["rank"])
        ious = np.array([iou.get((key[0], key[1], key[2], x["rank"]), np.nan) for x in rows])
        if not np.isfinite(ious).all():
            continue
        best = int(np.argmax(ious))
        bi = float(ious[best])
        fam = "recall" if bi < 0.3 else ("found" if best == 0 else ("rank2" if best == 1 else "deep"))
        scores = np.array([x[tag] for x in rows])
        pick = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
        per_scene.setdefault(key[0], []).append(float(ious[pick]))
        if fam == "found":
            found.append(float(pick == 0))
        elif fam == "rank2":
            r2.append(float(pick == best))
        elif fam == "deep":
            deep.append(float(pick == best))
        else:
            recall.append(float(pick == best))
    sm = float(np.mean([np.mean(v) for v in per_scene.values()]))
    res[tag] = dict(scene_mean=sm, found=np.mean(found) if found else np.nan,
                    rank2=np.mean(r2) if r2 else np.nan,
                    deep=np.mean(deep) if deep else np.nan,
                    recall=np.mean(recall) if recall else np.nan)

print("%-8s %10s %10s %8s %8s %8s" % ("signal", "scene-mean", "found ret", "deep", "rank2", "recall"))
for t in TAGS:
    d = res[t]
    print("%-8s %10.4f %10.3f %8.3f %8.3f %8.3f" % (t, d["scene_mean"], d["found"], d["deep"], d["rank2"], d["recall"]))

d = res
gates = [
    ("S2_c scene-mean >= S2_raw + 0.02", d["S2_c"]["scene_mean"] >= d["S2_raw"]["scene_mean"] + 0.02),
    ("S2_c found retention >= 0.95", d["S2_c"]["found"] >= 0.95),
    ("S2_c deep rescue >= 0.60", d["S2_c"]["deep"] >= 0.60),
    ("S2_c rank2 rescue >= 0.35", d["S2_c"]["rank2"] >= 0.35),
]
print()
for label, ok in gates:
    print("  [%s] %s" % ("PASS" if ok else "FAIL", label))
print("VERDICT:", "PASS" if all(ok for _, ok in gates) else "FAIL")