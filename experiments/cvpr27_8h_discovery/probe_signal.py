"""Can any NON-text signal identify the better candidate inside the released pool?

Diagnostic only (LOSO over the 4 LERF scenes). It answers one question:
is the 0.5565 -> 0.6948 single-candidate gap recoverable from geometry/visibility,
or is the information simply absent from everything we currently measure?

Feature groups:
  TEXT  similarity-derived  (score, deficit, id_*, agree*, obs_sim_*)
  GEO   geometry/visibility (area, n_points, cover, vis, flatness, height, ...)
  BOTH

Every feature is converted to a within-query percentile so that the model can only
learn "which candidate stands out among its rivals", not query-level scale.
"""
import csv, glob, os, sys
import numpy as np
from sklearn.linear_model import Ridge

G = sys.argv[1] if len(sys.argv) > 1 else "analysis/results/20261101/cnone_gate_tf"
EV = sys.argv[2] if len(sys.argv) > 2 else "analysis/results/20261101/cnone_ev"

TEXT = ["score", "deficit", "id_mean", "id_top3", "agree", "agree_top",
        "id_mean_c", "agree_c", "obs_sim_mean", "obs_sim_std", "obs_sim_max"]
GEO = ["n_points", "area_ratio", "cover_frac", "cover_views", "mwp_conf",
       "vis_frac", "mean_vis", "mem_frac", "mean_mem", "bbox_diag",
       "flatness", "height", "obs_views", "level"]
ALL = TEXT + GEO


def load():
    ev = {}
    for f in glob.glob(os.path.join(EV, "*_evidence.csv")):
        for r in csv.DictReader(open(f)):
            ev[(r["scene"], r["image"], r["prompt"], int(r["rank"]))] = r
    rows = []
    for f in sorted(glob.glob(os.path.join(G, "*_segidentity.csv"))):
        for r in csv.DictReader(open(f)):
            k = (r["scene"], r["image"], r["prompt"], int(r["rank"]))
            e = ev.get(k, {})
            d = dict(scene=r["scene"], image=r["image"], prompt=r["prompt"],
                     rank=int(r["rank"]), iou=float(r["iou"]))
            for c in ALL:
                v = r.get(c) or e.get(c)
                try:
                    d[c] = float(v)
                except (TypeError, ValueError):
                    d[c] = np.nan
            rows.append(d)
    return rows


def pct(vals):
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    out = np.zeros_like(v)
    if ok.sum() <= 1:
        return out
    order = v[ok].argsort().argsort().astype(float)
    out[ok] = order / max(len(order) - 1, 1)
    if (~ok).any():
        out[~ok] = 0.5
    return out


def build(rows):
    q = {}
    for r in rows:
        q.setdefault((r["scene"], r["image"], r["prompt"]), []).append(r)
    X, y, meta = [], [], []
    for k, rs in q.items():
        rs = sorted(rs, key=lambda z: z["rank"])
        for c in ALL:
            col = pct([z[c] for z in rs])
            for i, z in enumerate(rs):
                z["_" + c] = col[i]
        for z in rs:
            X.append([z["_" + c] for c in ALL])
            y.append(z["iou"])
            meta.append((k[0], k[1], k[2], z["rank"]))
    return np.array(X), np.array(y), meta


def macro(meta, pred_rank, rows):
    """pred_rank: dict (scene,image,prompt) -> chosen rank"""
    iou = {(r["scene"], r["image"], r["prompt"], r["rank"]): r["iou"] for r in rows}
    per_scene = {}
    sel = {}
    for (s, im, pr, rk) in meta:
        sel.setdefault((s, im, pr), rk)
    for (s, im, pr), rk in sel.items():
        per_scene.setdefault(s, {}).setdefault(im, []).append(iou[(s, im, pr, rk)])
    return float(np.mean([np.mean([np.mean(v) for v in imgs.values()])
                          for imgs in per_scene.values()])), sel, iou


def main():
    rows = load()
    X, y, meta = build(rows)
    scenes = sorted({m[0] for m in meta})
    queries = sorted({(m[0], m[1], m[2]) for m in meta})
    print("rows %d  queries %d  scenes %d" % (len(rows), len(queries), len(scenes)))

    # baselines
    sel_r1 = {q: min([m[3] for m in meta if (m[0], m[1], m[2]) == q]) for q in queries}
    _, _, iou = macro(meta, None, rows)
    def score_sel(sel):
        per = {}
        for (s, im, pr), rk in sel.items():
            per.setdefault(s, {}).setdefault(im, []).append(iou[(s, im, pr, rk)])
        return float(np.mean([np.mean([np.mean(v) for v in d.values()]) for d in per.values()]))
    print("%-26s %.4f" % ("rank1 (released top-1)", score_sel(sel_r1)))
    sel_or = {q: max([m[3] for m in meta if (m[0], m[1], m[2]) == q],
                     key=lambda rk: iou[(q[0], q[1], q[2], rk)]) for q in queries}
    print("%-26s %.4f" % ("pool oracle", score_sel(sel_or)))

    # univariate training-free rules
    qmap = {}
    for r in rows:
        qmap.setdefault((r["scene"], r["image"], r["prompt"]), []).append(r)
    for c in ["area_ratio", "n_points", "cover_views", "obs_views", "mwp_conf",
              "bbox_diag", "mean_vis", "flatness"]:
        sel = {}
        for q, rs in qmap.items():
            rs2 = [z for z in rs if np.isfinite(z[c])]
            if not rs2:
                sel[q] = 1
                continue
            best = max(rs2, key=lambda z: z[c])
            if c == "flatness":
                best = min(rs2, key=lambda z: z[c])
            sel[q] = best["rank"]
        print("%-26s %.4f" % ("max " + c if c != "flatness" else "min flatness",
                              score_sel(sel)))

    # LOSO ridge by feature group
    idx = {c: i for i, c in enumerate(ALL)}
    for name, cols in [("TEXT", TEXT), ("GEO", GEO), ("BOTH", ALL)]:
        cidx = [idx[c] for c in cols]
        sel = {}
        for s in scenes:
            tr = np.array([m[0] != s for m in meta])
            te = ~tr
            m = Ridge(alpha=1.0).fit(X[tr][:, cidx], y[tr])
            p = m.predict(X[te][:, cidx])
            qsel = {}
            j = 0
            for i, mm in enumerate(meta):
                if not te[i]:
                    continue
                k = (mm[0], mm[1], mm[2])
                if k not in qsel or p[j] > qsel[k][0]:
                    qsel[k] = (p[j], mm[3])
                j += 1
            for k, (_, rk) in qsel.items():
                sel[k] = rk
        print("%-26s %.4f" % ("LOSO ridge " + name, score_sel(sel)))


if __name__ == "__main__":
    main()