"""Aggregate the LERF-OVS candidate-oracle run and join it to the Grounded-SAM results.

Reads <out>/<scene>_perquery.csv (from lerf_ovs_cand_oracle.py), the released baseline
per-query CSVs, and the 3D-side gate table that contains the 35 queries where
Grounded-SAM beat CLIP by >0.05.

Aggregation follows the evaluator: mean over queries within an image, mean over images
within a scene, mean over the four scenes.
"""
import argparse
import collections
import csv
import json
import os
import statistics as st

ORACLES = ("c0_iou", "root_oracle", "hier_oracle", "combo_oracle",
           "deep_oracle", "grand_oracle")


def load_candidates(paths):
    """extra decompositions that need no new renders, read off the candidate table."""
    by_q = collections.defaultdict(lambda: {"root10": [], "l12": []})
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            q = (r["scene"], r["image"], r["prompt"])
            lv, v = int(r["level"]), float(r["iou"])
            if lv == 3:
                by_q[q]["root10"].append(v)
            elif lv in (1, 2):
                by_q[q]["l12"].append(v)
    return by_q


def load_rows(paths):
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows.extend(csv.DictReader(open(p)))
    for r in rows:
        for k in ORACLES:
            r[k] = float(r[k])
    return rows


def scene_img_means(rows, keys=ORACLES):
    """scene -> image -> mean over that image's queries, for every oracle."""
    per = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        per[r["scene"]][r["image"]][(r["prompt"])] = r
    out = {}
    for sc, imgs in per.items():
        out[sc] = {}
        for k in keys:
            out[sc][k] = st.mean([st.mean([q[k] for q in qs.values()]) for qs in imgs.values()])
    return out


def macro(sm, k):
    return st.mean([sm[sc][k] for sc in sm])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_dir", required=True)
    ap.add_argument("--base_dir", default="/root/rivermind-data/repro/deco-b123/base4_20261104")
    ap.add_argument("--gate3d", default="/root/rivermind-data/repro/deco-b123/"
                                        "scannet_sam_density_20261103/gate3d/gate3d.csv")
    ap.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scenes = args.scenes.split(",")

    rows = load_rows([os.path.join(args.oracle_dir, "%s_perquery.csv" % s) for s in scenes])
    cands = load_candidates([os.path.join(args.oracle_dir, "%s_candidates.csv" % s)
                             for s in scenes])
    keys = ORACLES + ("root10_oracle", "l12_oracle")
    for r in rows:
        q = (r["scene"], r["image"], r["prompt"])
        c = cands.get(q, {"root10": [], "l12": []})
        r["root10_oracle"] = max([r["c0_iou"]] + c["root10"])
        r["l12_oracle"] = max([r["c0_iou"]] + c["l12"])
    sm = scene_img_means(rows, keys)

    # parity vs the released baseline
    base = {}
    for sc in scenes:
        p = os.path.join(args.base_dir, sc, "per_query.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                base[(sc, r["image"], r["prompt"])] = float(r["iou"])
    diffs = [abs(r["c0_iou"] - base[(r["scene"], r["image"], r["prompt"])])
             for r in rows if (r["scene"], r["image"], r["prompt"]) in base]
    base_rows = {}
    for sc in scenes:
        p = os.path.join(args.base_dir, sc, "per_query.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                base_rows.setdefault((sc, r["image"]), {})[r["prompt"]] = float(r["iou"])
    base_macro = st.mean([
        st.mean([st.mean(list(qs.values())) for (s2, _), qs in base_rows.items() if s2 == sc])
        for sc in scenes])

    report = {"n_queries": len(rows),
              "parity_max_abs_diff": max(diffs) if diffs else None,
              "parity_n_common": len(diffs),
              "baseline_macro_csv": round(base_macro, 4),
              "aggregation": "per-image mean -> per-scene mean -> 4-scene macro"}
    report["scene"] = {sc: {k: round(sm[sc][k], 4) for k in keys} for sc in sm}
    report["macro"] = {k: round(macro(sm, k), 4) for k in keys}
    report["macro_minus_base"] = {
        k: round(macro(sm, k) - macro(sm, "c0_iou"), 4) for k in keys}

    for t in (0.05, 0.10):
        report["n_query_gain_gt_%.2f" % t] = {
            k: sum(1 for r in rows if r[k] > r["c0_iou"] + t) for k in keys[1:]}

    # candidate counts
    report["mean_candidates"] = {
        "n_root": round(st.mean([int(r["n_root"]) for r in rows]), 2),
        "n_pre": round(st.mean([int(r["n_pre"]) for r in rows]), 2),
        "n_combo": round(st.mean([int(r["n_combo"]) for r in rows]), 2),
        "n_deep": round(st.mean([int(r["n_deep"]) for r in rows]), 2),
        "n_unique_renders": round(st.mean([int(r["n_unique_renders"]) for r in rows]), 1)}

    # ---- Grounded-SAM: the 35 winning queries
    g3 = {}
    if os.path.exists(args.gate3d):
        for r in csv.DictReader(open(args.gate3d)):
            g3[(r["scene"], r["frame"], r["name"])] = r
    idx = {(r["scene"], r["image"], r["prompt"]): r for r in rows}
    wins = [k for k, r in g3.items() if float(r["gsam3d"]) > float(r["clip3d"]) + 0.05]

    detail, agg = [], collections.Counter()
    for k in wins:
        r = idx.get(k)
        if r is None:
            agg["missing"] += 1
            continue
        g = float(g3[k]["gsam3d"])
        d = dict(scene=k[0], image=k[1], prompt=k[2], base=round(r["c0_iou"], 4),
                 gsam=round(g, 4), root=round(r["root_oracle"], 4),
                 hier=round(r["hier_oracle"], 4), combo=round(r["combo_oracle"], 4),
                 root10=round(r["root10_oracle"], 4), l12=round(r["l12_oracle"], 4),
                 deep=round(r["deep_oracle"], 4), grand=round(r["grand_oracle"], 4))
        d["combo_minus_base"] = round(r["combo_oracle"] - r["c0_iou"], 4)
        d["grand_minus_base"] = round(r["grand_oracle"] - r["c0_iou"], 4)
        d["combo_ge_gsam"] = int(r["combo_oracle"] >= g - 1e-9)
        d["grand_ge_gsam"] = int(r["grand_oracle"] >= g - 1e-9)
        d["l12_ge_gsam"] = int(r["l12_oracle"] >= g - 1e-9)
        d["l12_gain_gt_0.05"] = int(r["l12_oracle"] - r["c0_iou"] > 0.05)
        d["root10_gain_gt_0.05"] = int(r["root10_oracle"] - r["c0_iou"] > 0.05)
        detail.append(d)
        agg["combo_ge_gsam"] += d["combo_ge_gsam"]
        agg["grand_ge_gsam"] += d["grand_ge_gsam"]
        agg["l12_ge_gsam"] += d["l12_ge_gsam"]
        agg["l12_gain_gt_0.05"] += d["l12_gain_gt_0.05"]
        agg["root10_gain_gt_0.05"] += d["root10_gain_gt_0.05"]
        agg["combo_gain_gt_0.05"] += int(d["combo_minus_base"] > 0.05)
        agg["combo_gain_gt_0.10"] += int(d["combo_minus_base"] > 0.10)
        agg["grand_gain_gt_0.05"] += int(d["grand_minus_base"] > 0.05)
        agg["grand_gain_gt_0.10"] += int(d["grand_minus_base"] > 0.10)
    report["gsam_wins"] = dict(n=len(wins), **dict(agg))
    report["gsam_wins_detail"] = detail

    # same statistics on all queries, to expose the selection bias
    allg = collections.Counter()
    for k, r in idx.items():
        gain = r["combo_oracle"] - r["c0_iou"]
        allg["combo_gain_gt_0.05"] += int(gain > 0.05)
        allg["combo_gain_gt_0.10"] += int(gain > 0.10)
        allg["grand_gain_gt_0.05"] += int(r["grand_oracle"] - r["c0_iou"] > 0.05)
        allg["grand_gain_gt_0.10"] += int(r["grand_oracle"] - r["c0_iou"] > 0.10)
    report["all_queries"] = dict(n=len(idx), **allg)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    if detail:
        with open(os.path.join(args.out, "gsam_wins_detail.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(detail[0]))
            wr.writeheader()
            wr.writerows(detail)
    print(json.dumps({k: v for k, v in report.items() if k != "gsam_wins_detail"},
                     indent=2))


if __name__ == "__main__":
    main()