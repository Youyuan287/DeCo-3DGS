"""Offline reliability aggregate for the LERF-OVS decision certificate.

Uses only existing witness / evidence / gate CSVs. No model or renderer is run.
Statistic: d(q) = fraction of views whose per-view top-1 candidate differs from
the merged-feature top-1 candidate (NAG score). High d means per-view evidence
carries independent information; low d means feature-level fusion is sufficient.
"""

import argparse
import collections
import csv
import json
import random
import statistics as st

SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]


def load_witness(root, scene):
    path = f"{root}/witness/{scene}_witness.csv"
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for row in csv.DictReader(open(path, encoding="utf-8")):
        out[(row["image"], row["prompt"])][row["rank"]][row["view_id"]] = float(row["sim"])
    return out


def load_evidence(root, scene, folder):
    path = f"{root}/{folder}/{scene}_evidence.csv"
    out = collections.defaultdict(dict)
    for row in csv.DictReader(open(path, encoding="utf-8")):
        out[(row["image"], row["prompt"])][row["rank"]] = row
    return out


def load_gate(root, scene):
    path = f"{root}/cnone_gate_tf/{scene}_cmaskgate_none.csv"
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        out[(row["image"], row["prompt"])] = (float(row["main_iou"]), float(row["gatee_iou"]))
    return out


def collect(root, evidence_folder):
    rows = []
    for scene in SCENES:
        witness = load_witness(root, scene)
        evidence = load_evidence(root, scene, evidence_folder)
        gate = load_gate(root, scene)
        for query in sorted(set(witness) & set(evidence) & set(gate)):
            ranks = sorted(set(witness[query]) & set(evidence[query]), key=int)
            if not ranks:
                continue
            merged = min(
                ranks,
                key=lambda r: (-float(evidence[query][r]["score"]), int(r)),
            )
            views = set().union(*[set(witness[query][r]) for r in ranks])
            per_view = {
                v: min(
                    ranks,
                    key=lambda r: (-witness[query][r].get(v, float("-inf")), int(r)),
                )
                for v in sorted(views)
            }
            disagreement = sum(1 for v in views if per_view[v] != merged) / len(views)
            main_iou, gate_iou = gate[query]
            rows.append(
                dict(
                    scene=scene,
                    image=query[0],
                    prompt=query[1],
                    disagreement=disagreement,
                    main=main_iou,
                    gate=gate_iou,
                )
            )
    return rows


def macro(selector, rows):
    scene_values = []
    for scene in SCENES:
        images = collections.defaultdict(list)
        for row in rows:
            if row["scene"] == scene:
                images[row["image"]].append(selector(row))
        if images:
            scene_values.append(st.mean([st.mean(v) for v in images.values()]))
    return st.mean(scene_values)


def auc(xs, ys):
    pairs = sorted(zip(xs, ys))
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = rank
        i = j + 1
    pos = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 1]
    neg = [ranks[k] for k, (_, y) in enumerate(pairs) if y == 0]
    if not pos or not neg:
        return None
    return (sum(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="analysis/results/20261101")
    parser.add_argument("--boot", type=int, default=5000)
    args = parser.parse_args()

    rows = collect(args.root, "ev_image")
    print(f"queries: {len(rows)}")
    print(f"{'scene':16s} {'n':>4s} {'mean_d':>8s} {'never':>8s} {'always':>8s} {'t0.5':>8s} {'applied':>8s}")
    for scene in SCENES:
        sub = [r for r in rows if r["scene"] == scene]
        never = macro(lambda r: r["main"], sub)
        always = macro(lambda r: r["gate"], sub)
        t05 = macro(lambda r: r["gate"] if r["disagreement"] >= 0.5 else r["main"], sub)
        applied = sum(1 for r in sub if r["disagreement"] >= 0.5)
        print(
            f"{scene:16s} {len(sub):4d} {st.mean([r['disagreement'] for r in sub]):8.4f} "
            f"{never:8.4f} {always:8.4f} {t05:8.4f} {applied:8d}"
        )

    policies = {
        "never": lambda r: r["main"],
        "always": lambda r: r["gate"],
        "t0.5": lambda r: r["gate"] if r["disagreement"] >= 0.5 else r["main"],
        "oracle": lambda r: r["gate"] if r["gate"] > r["main"] + 1e-12 else r["main"],
    }
    point = {name: macro(fn, rows) for name, fn in policies.items()}
    print("point", json.dumps({k: round(v, 6) for k, v in point.items()}))

    xs = [r["disagreement"] for r in rows]
    ys = [r["gate"] - r["main"] for r in rows]
    print("AUC(disagreement -> helps)", round(auc(xs, [1 if y > 1e-12 else 0 for y in ys]), 4))
    helped = [r["disagreement"] for r in rows if r["gate"] - r["main"] > 1e-12]
    not_helped = [r["disagreement"] for r in rows if r["gate"] - r["main"] <= 1e-12]
    print(
        "mean_d helped/not",
        round(st.mean(helped), 4),
        len(helped),
        round(st.mean(not_helped), 4),
        len(not_helped),
    )

    order = sorted(range(len(rows)), key=lambda i: xs[i])
    q = len(rows) // 4
    for name, idx in [
        ("Q1", order[:q]),
        ("Q2", order[q : 2 * q]),
        ("Q3", order[2 * q : 3 * q]),
        ("Q4", order[3 * q :]),
    ]:
        print(
            f"{name} d={st.mean([xs[i] for i in idx]):.3f} "
            f"mean_gain={st.mean([ys[i] for i in idx]):+.4f} n={len(idx)}"
        )

    images_by_scene = {
        scene: sorted(set(r["image"] for r in rows if r["scene"] == scene)) for scene in SCENES
    }
    random.seed(0)
    diffs = collections.defaultdict(list)
    for _ in range(args.boot):
        sample = []
        for scene in SCENES:
            images = images_by_scene[scene]
            for _ in range(len(images)):
                image = random.choice(images)
                sample += [r for r in rows if r["scene"] == scene and r["image"] == image]
        values = {name: macro(fn, sample) for name, fn in policies.items()}
        diffs["always-never"].append(values["always"] - values["never"])
        diffs["t0.5-always"].append(values["t0.5"] - values["always"])
        diffs["oracle-t0.5"].append(values["oracle"] - values["t0.5"])
    for name, values in diffs.items():
        values.sort()
        lo = values[int(0.025 * args.boot)]
        hi = values[int(0.975 * args.boot)]
        positive = sum(v > 0 for v in values) / args.boot
        print(
            f"{name} mean={st.mean(values):.6f} "
            f"95%CI=[{lo:.6f},{hi:.6f}] positive={positive:.3f}"
        )


if __name__ == "__main__":
    main()