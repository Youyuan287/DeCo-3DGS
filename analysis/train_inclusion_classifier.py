"""Train the candidate-membership classifier for inclusion v1 (PREREG_inclusion_v1.md).

Labels: GT-greedy subset membership on training scenes (LOSO). Features: GT-free
candidate features from the identity/evidence dumps. Output: per-candidate P(member)
for all scenes (out-of-fold predictions).
"""
import os, csv
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
INC = os.path.join(R, "lerf_inclusion_20261030")
ID = os.path.join(R, "lerf_identity_20261030")
EV = os.path.join(R, "lerf_evidence_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
FEATS = ["score", "deficit", "rank", "is_leaf", "n_points", "area", "mean_vis",
         "cover_views", "id_mean", "id_top3", "agree", "agree_top",
         "id_mean_c", "agree_c", "mwp"]


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


ev = {}
for s in SC:
    with open(os.path.join(EV, "%s_evidence.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ev[(s, r["image"], r["prompt"], int(r["rank"]))] = (
                num(r["area_ratio"]), num(r["mean_vis"]))

chosen = {}
with open(os.path.join(INC, "members_all.csv"), "w", newline="", encoding="utf-8") as out:
    wr = None
    for s in SC:
        with open(os.path.join(INC, "%s_greedymembers.csv" % s), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = (s, r["image"], r["prompt"])
                chosen[k] = set(int(x) for x in r["chosen"].split())
                if wr is None:
                    wr = csv.DictWriter(out, fieldnames=list(r))
                    wr.writeheader()
                wr.writerow(r)

rows = []
for s in SC:
    with open(os.path.join(ID, "%s_identity.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = (s, r["image"], r["prompt"])
            rank = int(r["rank"])
            e = ev.get((s, r["image"], r["prompt"], rank))
            if e is None or k not in chosen:
                continue
            rows.append(dict(scene=s, image=r["image"], prompt=r["prompt"], rank=rank,
                             score=num(r["score"]), deficit=num(r["deficit"]),
                             is_leaf=num(r["is_leaf"]), n_points=num(r["n_points"]),
                             area=e[0], mean_vis=e[1],
                             cover_views=num(r["cover_views"]),
                             id_mean=num(r["id_mean"]), id_top3=num(r["id_top3"]),
                             agree=num(r["agree"]), agree_top=num(r["agree_top"]),
                             id_mean_c=num(r["id_mean_c"]), agree_c=num(r["agree_c"]),
                             mwp=num(r["mwp_conf"]),
                             y=int(rank in chosen[k])))

print("rows:", len(rows), "| positive rate %.3f" % np.mean([r["y"] for r in rows]))
X = np.array([[r[f] for f in FEATS] for r in rows], float)
y = np.array([r["y"] for r in rows], int)
pred = np.full(len(rows), np.nan)
for s in SC:
    tr = np.array([r["scene"] != s for r in rows])
    te = ~tr
    ok = tr & np.isfinite(X).all(1)
    try:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                           min_samples_leaf=30, class_weight="balanced")
    except TypeError:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05,
                                           min_samples_leaf=30)
    m.fit(X[ok], y[ok])
    pred[te] = m.predict_proba(X[te])[:, 1]
    print("fold", s, "train pos %.3f" % y[ok].mean())

with open(os.path.join(INC, "preds_incl.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["scene", "image", "prompt", "rank", "p", "y"])
    for r, p in zip(rows, pred):
        wr.writerow([r["scene"], r["image"], r["prompt"], r["rank"], p, r["y"]])
print("wrote", os.path.join(INC, "preds_incl.csv"))

# quick diagnostics: top-1 probability, greedy-subset recall at p>=0.5
from collections import defaultdict
byq = defaultdict(list)
for r, p in zip(rows, pred):
    byq[(r["scene"], r["image"], r["prompt"])].append((r["rank"], p, r["y"]))
rec, tot = 0, 0
sel_sizes = []
for k, lst in byq.items():
    S = [rank for rank, p, yy in lst if p >= 0.5]
    if 1 not in S:
        S.append(1)
    sel_sizes.append(len(S))
    lab = chosen[k]
    if lab:
        tot += 1
        rec += len(set(S) & lab) / len(lab)
print("mean selected size at p>=0.5: %.2f" % np.mean(sel_sizes))
print("greedy-membership recall (queries with non-empty label): %.3f over %d" % (rec / max(tot, 1), tot))