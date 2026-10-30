"""Audit of the identity-evidence pilot (PREREG_identity.md).

Pure numpy + csv (runs anywhere). Loads <scene>_identity.csv for the 4 scenes,
assigns failure families exactly as predefined, and reports within-query pairwise
AUC / rescue rate with scene-stratified bootstrap CIs, plus size/context
residualization and the pre-registered gate decision.
"""
import os, sys, csv, math
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
ID = os.path.join(R, "lerf_identity_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
B_BOOT = 2000
SEED = 20261030

FEATS = ["id_mean", "id_top3", "agree", "agree_top", "cover_frac",
         "cover_views", "mwp_conf", "id_mean_c", "agree_c", "V", "Vc"]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def load():
    qs = {}
    for s in SC:
        path = os.path.join(ID, "%s_identity.csv" % s)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {k: r[k] for k in ("scene", "image", "prompt")}
                row["rank"] = int(r["rank"])
                for k in ("score", "deficit", "n_points", "area_ratio",
                          "cover_frac", "cover_views", "fallback_root", "is_leaf",
                          "root_id", "id_mean", "id_top3", "agree", "agree_top",
                          "mwp_conf", "id_mean_c", "agree_c", "iou"):
                    row[k] = num(r[k])
                key = (r["scene"], r["image"], r["prompt"])
                qs.setdefault(key, []).append(row)
    queries = []
    for key, rows in qs.items():
        rows.sort(key=lambda x: x["rank"])
        iou = np.array([x["iou"] for x in rows], float)
        best = int(np.nanargmax(iou))
        biou = float(iou[best])
        rank1_root = rows[0]["root_id"]
        fam = "recall_fail" if biou < 0.3 else (
            "found" if best == 0 else ("rank2" if best == 1 else "deep"))
        queries.append(dict(scene=key[0], key=key, rows=rows, best=best, fam=fam,
                            best_iou=biou,
                            same_root=bool(rows[best]["root_id"] == rank1_root)))
    return queries


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-12 else np.zeros_like(x)


def fvals(q, name):
    rows = q["rows"]
    get = lambda k: np.array([r[k] for r in rows], float)
    if name == "V":
        return zs(get("id_mean")) + zs(get("agree"))
    if name == "Vc":
        return zs(get("id_mean_c")) + zs(get("agree_c"))
    return get(name)


def pair_auc(v, best):
    v = np.asarray(v, float)
    if len(v) < 2 or not np.isfinite(v).any():
        return float("nan")
    b = v[best]
    o = np.delete(v, best)
    ok = np.isfinite(o)
    if not np.isfinite(b) or not ok.any():
        return float("nan")
    return float(np.mean((b > o[ok]) + 0.5 * (b == o[ok])))


def rescue(v, best):
    v = np.asarray(v, float)
    if len(v) < 2 or not np.isfinite(v).any():
        return float("nan")
    vv = np.where(np.isfinite(v), v, -np.inf)
    return float(np.argmax(vv) == best)


def boot_ci(per_query_vals, queries, metric="mean"):
    """Scene-stratified bootstrap over queries. per_query_vals: list aligned with queries."""
    rng = np.random.default_rng(SEED)
    by_scene = {}
    for q, v in zip(queries, per_query_vals):
        if v == v:  # not NaN
            by_scene.setdefault(q["scene"], []).append(v)
    arrs = {s: np.array(vs) for s, vs in by_scene.items()}
    if not arrs:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(B_BOOT):
        vals = []
        for s, a in arrs.items():
            idx = rng.integers(0, len(a), len(a))
            vals.extend(a[idx].tolist())
        stats.append(np.mean(vals))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def subset(queries, name):
    if name == "valid":
        return [q for q in queries if q["fam"] != "recall_fail"]
    if name == "hard":
        return [q for q in queries if q["fam"] in ("rank2", "deep")]
    if name == "cross_root_hard":
        return [q for q in queries if q["fam"] in ("rank2", "deep") and not q["same_root"]]
    return [q for q in queries if q["fam"] == name]


def report(queries, name, subset_name, out):
    sub = subset(queries, subset_name)
    rows = [("%-12s n=%3d" % (subset_name, len(sub)))]
    for f in FEATS:
        per_auc, per_res = [], []
        for q in sub:
            v = fvals(q, f)
            per_auc.append(pair_auc(v, q["best"]))
            per_res.append(rescue(v, q["best"]))
        ok_auc = [a for a in per_auc if a == a]
        ok_res = [r for r in per_res if r == r]
        if not ok_auc:
            rows.append("  %-10s (no valid queries)" % f)
            continue
        auc = float(np.mean(ok_auc))
        aci = boot_ci(per_auc, sub)
        res = float(np.mean(ok_res))
        rci = boot_ci(per_res, sub)
        rows.append("  %-10s AUC %.3f [%.3f,%.3f]  rescue %.3f [%.3f,%.3f]  n_auc=%d"
                    % (f, auc, aci[0], aci[1], res, rci[0], rci[1], len(ok_auc)))
    return rows


def residual_auc(queries, name, control, feat="V"):
    per = []
    for q in queries:
        rows = q["rows"]
        y = fvals(q, feat)
        c = np.array([r[control] for r in rows], float)
        ok = np.isfinite(y) & np.isfinite(c)
        if ok.sum() < 3:
            per.append(float("nan"))
            continue
        x = np.log1p(c[ok])
        yy = y[ok]
        A = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
        r = yy - A @ beta
        if np.std(r) < 1e-12:
            per.append(float("nan"))
            continue
        # AUC on residualized subset: compare residual at best vs others (within ok rows)
        idx = np.flatnonzero(ok)
        if q["best"] not in idx:
            per.append(float("nan"))
            continue
        bpos = int(np.where(idx == q["best"])[0][0])
        per.append(pair_auc(r, bpos))
    okv = [v for v in per if v == v]
    if not okv:
        return float("nan"), (float("nan"), float("nan")), 0
    return float(np.mean(okv)), boot_ci(per, queries), len(okv)


def main():
    queries = load()
    out = []
    out.append("Identity-evidence audit (PREREG_identity.md), queries=%d" % len(queries))
    for s in SC:
        sub = [q for q in queries if q["scene"] == s]
        fam = {k: sum(1 for q in sub if q["fam"] == k)
               for k in ("found", "rank2", "deep", "recall_fail")}
        out.append("  %-14s n=%3d %s" % (s, len(sub), fam))
    out.append("")

    for sub_name in ("valid", "hard", "cross_root_hard", "deep", "rank2", "found"):
        out.extend(report(queries, "audit", sub_name, None))
        out.append("")

    valid = subset(queries, "valid")
    hard = subset(queries, "hard")
    xhard = subset(queries, "cross_root_hard")
    deep = subset(queries, "deep")
    found = subset(queries, "found")

    for ctrl in ("n_points", "cover_views"):
        a1, c1, n1 = residual_auc(valid, "audit", ctrl)
        a2, c2, n2 = residual_auc(hard, "audit", ctrl)
        out.append("Residualized on log1p(%s): valid AUC %.3f [%.3f,%.3f] n=%d | hard AUC %.3f [%.3f,%.3f] n=%d"
                   % (ctrl, a1, c1[0], c1[1], n1, a2, c2[0], c2[1], n2))
    out.append("")

    # fallback / NaN / tie diagnostics
    fb = sum(1 for q in queries if any(r["fallback_root"] == 1 for r in q["rows"]))
    nanV = sum(1 for q in queries if not np.isfinite(fvals(q, "V")).any())
    out.append("fallback_root queries: %d ; all-NaN V queries: %d" % (fb, nanV))
    out.append("")

    def gate_metric(qs, name, kind):
        vals = []
        for q in qs:
            v = fvals(q, name)
            vals.append(pair_auc(v, q["best"]) if kind == "auc" else rescue(v, q["best"]))
        vv = [x for x in vals if x == x]
        return float(np.mean(vv)), boot_ci(vals, qs)

    g = []
    a_hard, ci_hard = gate_metric(hard, "V", "auc")
    r_hard, rci_hard = gate_metric(hard, "V", "res")
    a_x, ci_x = gate_metric(xhard, "V", "auc")
    a_found, ci_found = gate_metric(found, "V", "auc")
    a_deep, ci_deep = gate_metric(deep, "V", "auc")
    a_res, ci_res, _ = residual_auc(hard, "audit", "n_points")
    checks = [
        ("hard AUC>=0.75 & CI_low>0.60", a_hard >= 0.75 and ci_hard[0] > 0.60),
        ("hard rescue>=0.45 & CI_low>0.30", r_hard >= 0.45 and rci_hard[0] > 0.30),
        ("cross_root_hard AUC CI_low>0.60", ci_x[0] > 0.60),
        ("hard residual-(n_points) AUC CI_low>0.55", ci_res[0] > 0.55),
        ("found AUC>=0.60 (harm)", a_found >= 0.60),
        ("deep AUC>=0.70 (secondary)", a_deep >= 0.70),
    ]
    out.append("GATE: hard AUC %.3f [%.3f,%.3f] rescue %.3f [%.3f,%.3f]; cross_root AUC CI [%.3f,%.3f]; found AUC %.3f; deep AUC %.3f; resid CI [%.3f,%.3f]"
               % (a_hard, ci_hard[0], ci_hard[1], r_hard, rci_hard[0], rci_hard[1],
                  ci_x[0], ci_x[1], a_found, a_deep, ci_res[0], ci_res[1]))
    for label, ok in checks:
        out.append("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    verdict = "PASS" if all(ok for _, ok in checks) else "FAIL"
    out.append("VERDICT: %s" % verdict)

    txt = "\n".join(out)
    print(txt)
    with open(os.path.join(ID, "audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()