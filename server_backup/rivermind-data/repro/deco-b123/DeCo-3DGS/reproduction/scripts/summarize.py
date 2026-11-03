import json, csv, os
R = "/root/rivermind-data/ReLaGS_Reproduction"
paper = {"figurines": 64.7, "ramen": 51.2, "teatime": 81.0, "waldo_kitchen": 60.6}
rows = []
for sc in paper:
    f = os.path.join(R, "results", f"{sc}_summary.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); m = d["miou"] * 100
    rows.append(dict(scene=sc, repro_miou=round(m, 2), paper_miou=paper[sc],
                     diff=round(m - paper[sc], 2), n_queries=d["n_queries"],
                     protocol='topk=%d,level_until=%d,remove_small=%s' % (
                         d["protocol"]["topk"], d["protocol"]["level_until"],
                         d["protocol"]["remove_small"])))
mean = sum(r["repro_miou"] for r in rows) / len(rows)
rows.append(dict(scene="MEAN", repro_miou=round(mean, 2), paper_miou=64.4,
                 diff=round(mean - 64.4, 2), n_queries=sum(r["n_queries"] for r in rows),
                 protocol=rows[0]["protocol"]))
with open(os.path.join(R, "results", "lerf_scene_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scene", "repro_miou", "paper_miou", "diff", "n_queries", "protocol"])
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
