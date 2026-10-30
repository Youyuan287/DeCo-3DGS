"""Stage-1 audit for identity pilot v2 (see PREREG_identity_v2.md).

Joins the existing identity dump (centered identity stream) with the evidence dump
(scale stream), builds the pre-registered fused evidence F, and evaluates the
frozen decision rule R_F on ranking-level metrics with scene-stratified bootstrap.
No rendering, no training, GT used only for the IoU label of 'best candidate'.
"""
import os, csv, math
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
ID = os.path.join(R, "lerf_identity_20261030")
EV = os.path.join(R, "lerf_evidence_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else np.zeros_like(x)


def pair_auc(v, best):
    v = np.asarray(v, float)
    o = np.delete(v, best)
    ok = np.isfinite(o)
    if not np.isfinite(v[best]) or not ok.any():
        return float("nan")
    return float(np.mean((v[best] > o[ok]) + 0.5 * (v[best] == o[ok])))


def boot_ci(vals, scenes, B=B_BOOT):
    rng = np.random.default_rng(SEED)
    by = {}
    for v, s in zip(vals, scenes):
        if v == v:
            by.setdefault(s, []).append(v)
    if not by:
        return (float("nan"), float("nan"))
    arrs = {s: np.array(v) for s, v in by.items()}
    out = []
    for _ in range(B):
        acc = []
        for s, a in arrs.items():
            acc.extend(a[rng.integers(0, len(a), len(a))].tolist())
        out.append(np.mean(acc))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


ev = {}
for s in SC:
    with open(os.path.join(EV, "%s_evidence.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ev[(s, r["image"], r["prompt"], int(r["rank"]))] = (
                num(r["area_ratio"]), num(r["mean_vis"]), num(r["iou"]))

queries = {}
for s in SC:
    with open(os.path.join(ID, "%s_identity.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (s, r["image"], r["prompt"])
            rank = int(r["rank"])
            e = ev.get((s, r["image"], r["prompt"], rank))
            row = dict(rank=rank, root_id=int(float(r["root_id"])),
                       is_leaf=num(r["is_leaf"]),
                       iou=num(r["iou"]),
                       id_mean_c=num(r["id_mean_c"]), agree_c=num(r["agree_c"]),
                       n_points=num(r["n_points"]),
                       area=num(r["area_ratio"]),
                       mean_vis=e[1] if e else float("nan"))
            queries.setdefault(key, []).append(row)

Q = []
for key, rows in queries.items():
    rows.sort(key=lambda x: x["rank"])
    iou = np.array([x["iou"] for x in rows], float)
    best = int(np.nanargmax(iou))
    biou = float(iou[best])
    fam = "recall_fail" if biou < 0.3 else (
        "found" if best == 0 else ("rank2" if best == 1 else "deep"))
    area = np.array([x["area"] for x in rows], float)
    mv = np.array([x["mean_vis"] for x in rows], float)
    idc = np.array([x["id_mean_c"] for x in rows], float)
    agc = np.array([x["agree_c"] for x in rows], float)
    A = zs(area) + zs(mv)
    Vc = zs(idc) + zs(agc)
    F = zs(A) + zs(Vc)
    Q.append(dict(key=key, scene=key[0], rows=rows, best=best, fam=fam,
                  same_root=bool(rows[best]["root_id"] == rows[0]["root_id"]),
                  A=A, Vc=Vc, F=F,
                  n_points=np.array([x["n_points"] for x in rows], float),
                  area=area))

names = Q
hard = [q for q in names if q["fam"] in ("rank2", "deep")]
xhard = [q for q in hard if not q["same_root"]]
deep = [q for q in names if q["fam"] == "deep"]
found = [q for q in names if q["fam"] == "found"]
valid = [q for q in names if q["fam"] != "recall_fail"]


def pick(v):
    return int(np.argmax(np.where(np.isfinite(v), v, -np.inf)))


def gate_rule(q):
    a = q["area"]
    if len(a) and a[0] < 0.001 and a.max() > 2 * a[0] and pick(q["F"]) != 0:
        return pick(q["F"])
    return 0


def report(sub, label, vec_key, rule=None, kind="rescue"):
    vals, scenes = [], []
    for q in sub:
        v = q[vec_key]
        p = gate_rule(q) if rule == "gate" else pick(v)
        if kind == "rescue":
            vals.append(float(p == q["best"]))
        elif kind == "retention":
            vals.append(float(p == 0))
        elif kind == "auc":
            vals.append(pair_auc(v, q["best"]))
        scenes.append(q["scene"])
    ok = [v for v in vals if v == v]
    m = float(np.mean(ok)) if ok else float("nan")
    ci = boot_ci(vals, scenes)
    return m, ci, len(ok)


def residual_auc(sub, key):
    vals, scenes = [], []
    for q in sub:
        y = q[key]
        c = np.log1p(np.where(np.isfinite(q["n_points"]), q["n_points"], np.nan))
        ok = np.isfinite(y) & np.isfinite(c) & (c > 0)
        if ok.sum() < 3 or q["best"] not in np.flatnonzero(ok):
            vals.append(float("nan"))
        else:
            x = c[ok]
            yy = y[ok]
            Amat = np.vstack([np.ones_like(x), x]).T
            beta, *_ = np.linalg.lstsq(Amat, yy, rcond=None)
            r = yy - Amat @ beta
            if np.std(r) < 1e-12:
                vals.append(float("nan"))
            else:
                idx = np.flatnonzero(ok)
                bpos = int(np.where(idx == q["best"])[0][0])
                vals.append(pair_auc(r, bpos))
        scenes.append(q["scene"])
    okv = [v for v in vals if v == v]
    return (float(np.mean(okv)) if okv else float("nan"), boot_ci(vals, scenes), len(okv))


lines = ["Identity pilot v2 — stage 1 (PREREG_identity_v2.md), queries=%d" % len(names)]
lines.append("families: hard=%d (deep %d, rank2 %d), cross_root_hard=%d, found=%d, recall_fail=%d"
             % (len(hard), len(deep), len(hard) - len(deep), len(xhard), len(found),
                len(names) - len(valid)))

for key, label in (("F", "F=z(A)+z(Vc)"), ("Vc", "Vc"), ("A", "A=z(area)+z(mean_vis)")):
    for sub, sname, kind in ((hard, "hard", "auc"), (hard, "hard", "rescue"),
                             (xhard, "cross_root_hard", "rescue"),
                             (deep, "deep", "rescue"), (found, "found", "retention")):
        m, ci, n = report(sub, sname, key, kind=kind)
        lines.append("  %-10s %-16s %-10s %.3f [%.3f,%.3f] n=%d"
                     % (label, sname, kind, m, ci[0], ci[1], n))
    lines.append("")

m, ci, n = report(hard, "hard", "F", rule="gate", kind="rescue")
lines.append("R_F_gate hard rescue %.3f [%.3f,%.3f] n=%d" % (m, ci[0], ci[1], n))
m, ci, n = report(found, "found", "F", rule="gate", kind="retention")
lines.append("R_F_gate found retention %.3f [%.3f,%.3f] n=%d" % (m, ci[0], ci[1], n))

m, ci, n = residual_auc(hard, "F")
lines.append("F residualized on log n_points (hard): AUC %.3f [%.3f,%.3f] n=%d" % (m, ci[0], ci[1], n))

# gate
g = []
mr, cr, _ = report(hard, "hard", "F", kind="rescue")
g.append(("hard rescue(R_F)>=0.50 & CI_low>0.30", mr >= 0.50 and cr[0] > 0.30))
mret, cret, _ = report(found, "found", "F", kind="retention")
g.append(("found retention>=0.85 & CI_low>0.75", mret >= 0.85 and cret[0] > 0.75))
mx, cx, _ = report(xhard, "cross_root_hard", "F", kind="rescue")
g.append(("cross_root_hard rescue>=0.45 & CI_low>0.25", mx >= 0.45 and cx[0] > 0.25))
mr2, cr2, _ = residual_auc(hard, "F")
g.append(("hard residual AUC(F) CI_low>0.60", cr2[0] > 0.60))
lines.append("")
for label, ok in g:
    lines.append("  [%s] %s" % ("PASS" if ok else "FAIL", label))
verdict = "PASS" if all(ok for _, ok in g) else "FAIL"
lines.append("VERDICT: %s" % verdict)

txt = "\n".join(lines)
print(txt)
with open(os.path.join(ID, "audit_v2_stage1.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")