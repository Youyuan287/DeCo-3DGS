"""Phase B audit (PREREG_identity_phaseB.md): segment-level identity evidence.

Same statistics as the v2 stage-1 audit (scene-stratified bootstrap, seed frozen);
identity stream comes from <scene>_segidentity.csv, scale stream from the evidence
dump. Gate 4 uses Vc_seg (not F) per the Phase B prereg.
"""
import os, csv
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
ID = os.path.join(R, "lerf_segidentity_20261030")
EV = os.path.join(R, "lerf_evidence_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
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
                num(r["area_ratio"]), num(r["mean_vis"]))

queries = {}
fb = 0
for s in SC:
    with open(os.path.join(ID, "%s_segidentity.csv" % s), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (s, r["image"], r["prompt"])
            rank = int(r["rank"])
            e = ev.get((s, r["image"], r["prompt"], rank))
            row = dict(rank=rank, root_id=int(float(r["root_id"])),
                       iou=num(r["iou"]), n_points=num(r["n_points"]),
                       area=num(r["area_ratio"]),
                       id_mean_c=num(r["id_mean_c"]), agree_c=num(r["agree_c"]),
                       mean_vis=e[1] if e else float("nan"),
                       fb=num(r["fallback_seg"]) if "fallback_seg" in r else 0)
            fb += int(row["fb"] == 1)
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
    Q.append(dict(key=key, scene=key[0], rows=rows, best=best, fam=fam,
                  same_root=bool(rows[best]["root_id"] == rows[0]["root_id"]),
                  A=A, Vc=Vc, n_points=np.array([x["n_points"] for x in rows], float),
                  area=area))

hard = [q for q in Q if q["fam"] in ("rank2", "deep")]
xhard = [q for q in hard if not q["same_root"]]
deep = [q for q in Q if q["fam"] == "deep"]
found = [q for q in Q if q["fam"] == "found"]
valid = [q for q in Q if q["fam"] != "recall_fail"]


def pick(q, key, cons=False):
    if cons:
        pa = int(np.argmax(q["A"]))
        pv = int(np.argmax(q["Vc"]))
        return int(pa) if (pa == pv and pa != 0) else 0
    return int(np.argmax(q[key]))


def report(sub, key, kind, cons=False):
    vals, sc = [], []
    for q in sub:
        p = pick(q, key, cons)
        vals.append(float(p == q["best"]) if kind == "rescue" else float(p == 0))
        sc.append(q["scene"])
    ok = [v for v in vals if v == v]
    return (float(np.mean(ok)) if ok else float("nan"), boot_ci(vals, sc), len(ok))


def auc_report(sub, key):
    vals, sc = [], []
    for q in sub:
        vals.append(pair_auc(q[key], q["best"]))
        sc.append(q["scene"])
    ok = [v for v in vals if v == v]
    return (float(np.mean(ok)) if ok else float("nan"), boot_ci(vals, sc), len(ok))


def residual_auc(sub, key):
    vals, sc = [], []
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
        sc.append(q["scene"])
    okv = [v for v in vals if v == v]
    return (float(np.mean(okv)) if okv else float("nan"), boot_ci(vals, sc), len(okv))


lines = ["Phase B segment identity audit (PREREG_identity_phaseB.md), queries=%d" % len(Q)]
lines.append("families: hard=%d (deep %d), cross_root_hard=%d, found=%d, recall_fail=%d; fallback rows=%d"
             % (len(hard), len(deep), len(xhard), len(found),
                len(Q) - len(valid), fb))

for key, label in (("Vc", "Vc_seg"), ("A", "A(scale)")):
    for sub, name, kind in ((hard, "hard", "auc"), (hard, "hard", "rescue"),
                            (xhard, "cross_root_hard", "rescue"),
                            (deep, "deep", "rescue"), (found, "found", "retention")):
        if kind == "auc":
            m, ci, n = auc_report(sub, key)
        else:
            m, ci, n = report(sub, key, kind)
        lines.append("  %-8s %-16s %-10s %.3f [%.3f,%.3f] n=%d"
                     % (label, name, kind, m, ci[0], ci[1], n))
    lines.append("")

m1, c1, _ = report(hard, "Vc", "rescue", cons=True)
lines.append("R_cons_seg hard rescue %.3f [%.3f,%.3f]" % (m1, c1[0], c1[1]))
m2, c2, _ = report(found, "Vc", "retention", cons=True)
lines.append("R_cons_seg found retention %.3f [%.3f,%.3f]" % (m2, c2[0], c2[1]))
mr, cr, _ = residual_auc(hard, "Vc")
lines.append("Vc_seg residualized on log n_points (hard): AUC %.3f [%.3f,%.3f] n=%d"
             % (mr, cr[0], cr[1], _))

g = []
mr2, cr2, _ = report(hard, "Vc", "rescue")
g.append(("hard rescue(Vc_seg)>=0.50 & CI_low>0.30", mr2 >= 0.50 and cr2[0] > 0.30))
mret, cret, _ = report(found, "Vc", "retention")
g.append(("found retention>=0.85 & CI_low>0.75", mret >= 0.85 and cret[0] > 0.75))
mx, cx, _ = report(xhard, "Vc", "rescue")
g.append(("cross_root_hard rescue>=0.45 & CI_low>0.25", mx >= 0.45 and cx[0] > 0.25))
mres, cres, _ = residual_auc(hard, "Vc")
g.append(("hard residual AUC(Vc_seg) CI_low>0.60", cres[0] > 0.60))
lines.append("")
for label, ok in g:
    lines.append("  [%s] %s" % ("PASS" if ok else "FAIL", label))
verdict = "PASS" if all(ok for _, ok in g) else "FAIL"
lines.append("VERDICT: %s" % verdict)

txt = "\n".join(lines)
print(txt)
with open(os.path.join(ID, "audit_phaseB.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")