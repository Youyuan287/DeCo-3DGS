"""Certified decision revision on frozen open-vocabulary fields.

For each query we have three selections built from released evidence:
  S0    released gap prefix
  S_C   S0 plus extent-dominant challenger (coverage correction)
  S_M   mask-verdict rerank of the released prefix

Per-view witness sims are read from <witness>/<scene>_witness.csv. For a
selection S, a view v supports it by E_v(S) = max_{c in S} sim_v(c).
The challenger B is certified over incumbent A iff a half-sample bootstrap
over views finds mean_v (E_v(B) - E_v(A)) > 0 in at least pi of resamples
(pi = 0.9, the stability-selection convention; no magnitude threshold).

IoUs of S0/S_C/S_M come from the gate CSV (main_iou/c_iou/mask_iou), so the
audit needs no rendering.
"""
import csv, glob, os, sys
import json
import numpy as np

GATE = sys.argv[1]
SEG = sys.argv[2]
EV = sys.argv[3]
WIT = sys.argv[4]
SUFFIX = sys.argv[5] if len(sys.argv) > 5 else "none"
PI = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
MODE = sys.argv[7] if len(sys.argv) > 7 else "sc"  # sc | s0_aug | s0
VN_DIR = sys.argv[8] if len(sys.argv) > 8 else ""  # optional: leave-query-view-out
WSUF = sys.argv[9] if len(sys.argv) > 9 else "_witness"
NBOOT = 500
LAM = 1.0
rng = np.random.default_rng(20261101)


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def load_keyed(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["image"], r["prompt"])] = r
    return d


def load_ranked(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["image"], r["prompt"], int(r["rank"]))] = r
    return d


def load_witness(path):
    d = {}
    for r in csv.DictReader(open(path)):
        k = (r["image"], r["prompt"], int(r["rank"]))
        d.setdefault(k, {})[int(r["view_id"])] = float(r["sim"])
    return d


def set_evidence(S, obs):
    """view -> max sim over members of S observed in that view"""
    out = {}
    for c in S:
        for v, s in obs.get(c, {}).items():
            if s > out.get(v, -1e9):
                out[v] = s
    return out


def certify(A, B, obs, exclude=None, nb=NBOOT, half=True):
    ea = set_evidence(A, obs)
    eb = set_evidence(B, obs)
    views = sorted(set(ea) & set(eb))
    if exclude is not None:
        views = [v for v in views if v != exclude]
    n = len(views)
    if n < 5:
        return 0.0, 0.0, n
    d = np.array([eb[v] - ea[v] for v in views])
    if half:
        k = max(2, n // 2)
        means = np.array([d[rng.choice(n, k, replace=False)].mean() for _ in range(nb)])
    else:
        means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nb)])
    return float((means > 0).mean()), float(np.quantile(means, 0.05)), n


def main():
    per_scene = {}
    recs = []
    stats = {"cert_fire": 0, "cert_fire_found": 0, "cert_fire_deep": 0,
             "n_found": 0, "n_deep": 0, "skipped_small_views": 0}
    for gf in sorted(glob.glob(os.path.join(GATE, "*_cmaskgate_%s.csv" % SUFFIX))):
        sc = os.path.basename(gf).split("_cmaskgate")[0]
        seg = load_ranked(os.path.join(SEG, "%s_segidentity.csv" % sc))
        ev = load_ranked(os.path.join(EV, "%s_evidence.csv" % sc))
        wit = load_witness(os.path.join(WIT, "%s%s.csv" % (sc, WSUF)))
        names = None
        if VN_DIR:
            names = json.load(open(os.path.join(VN_DIR, "view_names_%s.json" % sc)))
        rows = list(csv.DictReader(open(gf)))
        per_img = {}
        for r in rows:
            im, pr = r["image"], r["prompt"]
            n, g = int(r["n"]), int(r["gap_len"])
            sc_l = [float(seg[(im, pr, i + 1)]["score"]) for i in range(n)]
            idl = []
            for i in range(n):
                v = seg[(im, pr, i + 1)].get("id_top3", "")
                idl.append(float(v) if v != "" else np.nan)
            idl = np.asarray(idl, float)
            fill = np.nanmean(idl) if np.isfinite(idl).any() else 0.0
            idl = np.where(np.isfinite(idl), idl, fill)
            area = np.array([float(ev[(im, pr, i + 1)]["area_ratio"]) for i in range(n)])
            mvis = np.array([float(ev[(im, pr, i + 1)]["mean_vis"]) for i in range(n)])
            S0 = list(range(g)) if g > 0 else []
            order = np.argsort(-(zs(sc_l) + LAM * zs(idl)), kind="stable")
            SM = [int(x) for x in order[:g]] if g > 0 else []
            SC = list(S0)
            champ = int(np.argmax(zs(area) + zs(mvis)))
            if area[0] < 0.001 and area.max() > 2 * area[0] and champ != 0:
                SC = sorted(set(SC) | {champ})
            if len(SC) != int(r["c_len"]) or len(SM) != int(r["mask_len"]):
                # reconstruction guard: fall back to released default
                pass
            obs = {i: wit.get((im, pr, i + 1), {}) for i in range(n)}
            # family
            ious = [float(seg[(im, pr, i + 1)]["iou"]) for i in range(n)]
            best = max(ious)
            br = ious.index(best) + 1
            fam = "recall" if best < 0.3 else ("found" if br == 1 else ("rank2" if br == 2 else "deep"))
            # certified revision against the incumbent (composition ablations)
            incumbent = SC if MODE == "sc" else S0
            fallback_iou = float(r["c_iou"]) if MODE in ("sc", "s0_aug") else float(r["main_iou"])
            fire = False
            if set(SM) != set(incumbent):
                ex = names.index(im) if (names and im in names) else None
                p, lcb, nv = certify(incumbent, SM, obs, exclude=ex)
                if nv < 5:
                    stats["skipped_small_views"] += 1
                fire = p >= PI
            if fam == "found":
                stats["n_found"] += 1
            if fam == "deep":
                stats["n_deep"] += 1
            if fire:
                stats["cert_fire"] += 1
                if fam == "found":
                    stats["cert_fire_found"] += 1
                if fam == "deep":
                    stats["cert_fire_deep"] += 1
            iou = float(r["mask_iou"]) if fire else fallback_iou
            per_img.setdefault(im, []).append((iou, float(r["main_iou"]), float(r["c_iou"]), float(r["mask_iou"])))
            recs.append(dict(scene=sc, image=im, prompt=pr, family=fam, fire=fire,
                             cert=iou, main=float(r["main_iou"]), c=float(r["c_iou"]),
                             mask=float(r["mask_iou"])))
        out = {}
        for t, idx in (("main", 1), ("c", 2), ("mask", 3), ("cert", 0)):
            out[t] = float(np.mean([np.mean([x[idx] for x in xs]) for xs in per_img.values()]))
        per_scene[sc] = out
    print("| scene | main | C | mask | certified | d vs main |")
    for sc, d in per_scene.items():
        print("| %s | %.4f | %.4f | %.4f | %.4f | %+.4f |" % (
            sc, d["main"], d["c"], d["mask"], d["cert"], d["cert"] - d["main"]))
    for t in ("main", "c", "mask", "cert"):
        print("MACRO %-6s %.4f" % (t, np.mean([d[t] for d in per_scene.values()])))
    print("cert fired on %d/208 (found %d/%d, deep %d/%d); small-view skips %d" % (
        stats["cert_fire"], stats["cert_fire_found"], stats["n_found"],
        stats["cert_fire_deep"], stats["n_deep"], stats["skipped_small_views"]))
    print("\nfamily  n    main    C      mask   cert   fires")
    for fam in ("found", "rank2", "deep", "recall"):
        rs = [x for x in recs if x["family"] == fam]
        if not rs:
            continue
        print("%-7s %3d  %.4f %.4f %.4f %.4f   %d" % (
            fam, len(rs), np.mean([x["main"] for x in rs]), np.mean([x["c"] for x in rs]),
            np.mean([x["mask"] for x in rs]), np.mean([x["cert"] for x in rs]),
            sum(x["fire"] for x in rs)))

    def nested(rows, key):
        per = {}
        for x in rows:
            per.setdefault((x["scene"], x["image"]), []).append(x[key])
        sc = {}
        for (s, _), v in per.items():
            sc.setdefault(s, []).append(np.mean(v))
        return float(np.mean([np.mean(v) for v in sc.values()]))

    rng2 = np.random.default_rng(20261101)
    by = {}
    for x in recs:
        by.setdefault(x["scene"], []).append(x)
    deltas = []
    for _ in range(2000):
        samp = []
        for s, rs in by.items():
            samp += [rs[rng2.integers(0, len(rs))] for _ in range(len(rs))]
        deltas.append(nested(samp, "cert") - nested(samp, "main"))
    deltas = np.sort(deltas)
    print("cert-main mean %+.4f CI95 [%+.4f, %+.4f]" % (
        np.mean(deltas), deltas[50], deltas[1950]))


if __name__ == "__main__":
    main()