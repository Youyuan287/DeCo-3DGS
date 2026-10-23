#!/usr/bin/env python
# ReLaGS LERF-OVS reproduction — retrieval hyper-parameter sweep
# 用途: 扫描 topk / level_until / filter_small / remove_small, 输出每配置 mIoU、
#       最优全局配置、以及"逐 query 取最优"的 ORACLE 上限(GT 驱动, 仅用于估计天花板)。
# 注意: ORACLE 行使用了 GT 选择配置, 不可作为复现结果引用。
"""Sweep retrieval hyper-parameters of the cluster-based LERF protocol.

Reports (a) mIoU for every config, (b) the best single global config,
(c) a per-query oracle upper bound (best config chosen per query with GT).
GT-driven: this is an upper-bound study, not a reproduction claim.
"""
import os, sys, json, csv, argparse, itertools
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch, cv2

RE = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RE)
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG

def poly(shape, pts):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1); return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", default="1,3,5,10")
    ap.add_argument("--level_until", default="-1,0,1,2,3")
    ap.add_argument("--filter_small", default="0,50,500")
    ap.add_argument("--remove_small", default="1")
    ap.add_argument("--per_query_csv", default="")
    args = ap.parse_args()

    p = argparse.ArgumentParser(); lp = ModelParams(p); pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(["-s", args.source, "-m", args.model])
    dataset = lp.extract(pargs); pipe = pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device="cuda")
    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure(); vlm.load_model()

    gt_dir = os.path.join(args.gt_root, args.scene)
    imgs = sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg"))
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    cfgs = [dict(topk=t, level_until=l, filter_small=f, remove_small=bool(r))
            for t, l, f, r in itertools.product(
                [int(x) for x in args.topk.split(",")],
                [int(x) for x in args.level_until.split(",")],
                [int(x) for x in args.filter_small.split(",")],
                [int(x) for x in args.remove_small.split(",")])]
    per = {i: {} for i in range(len(cfgs))}   # cfg idx -> (img,prompt) -> iou
    gt_cache = {}
    with torch.no_grad():
        for im in imgs:
            name = im[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            prompts = []
            for o in anno["objects"]:
                if o["category"] not in prompts: prompts.append(o["category"])
            cam = cams.get(name)
            if cam is None: continue
            h, w = cam.image_height, cam.image_width
            for prompt in prompts:
                vlm.encode_text(prompt)
                sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
                gt = np.zeros((h,w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, poly((h,w), o["segmentation"]))
                gt_cache[(name,prompt)] = gt
                for ci, c in enumerate(cfgs):
                    rel, _ = snag.search_matched_superpoint_in_mhtree(
                        sims, topk=c["topk"], level_until=c["level_until"],
                        filter_small=c["filter_small"], remove_small=c["remove_small"])
                    gaussians._semantics = rel.expand(-1, 20).cuda()
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    pred = (sem.reshape(sem.shape[0],-1)[0] > 0.5).reshape(h,w).cpu().numpy()
                    inter = float(np.logical_and(pred, gt).sum()); union = float(np.logical_or(pred, gt).sum())
                    per[ci][(name,prompt)] = inter/union if union>0 else float("nan")
            print(f"  {name}: {len(prompts)} queries done", flush=True)

    rows = []
    for ci, c in enumerate(cfgs):
        v = [x for x in per[ci].values() if not np.isnan(x)]
        rows.append(dict(scene=args.scene, topk=c["topk"], level_until=c["level_until"],
                         filter_small=c["filter_small"], remove_small=int(c["remove_small"]),
                         miou_grand=float(np.mean(v)) if v else float("nan"),
                         n=len(v)))
    rows.sort(key=lambda r: -r["miou_grand"])
    out = os.path.join(args.out, f"sweep_{args.scene}.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("\n=== top 8 configs (pooled over queries) ===")
    for r in rows[:8]:
        print(f"  topk={r['topk']:>2} level_until={r['level_until']:>2} filter_small={r['filter_small']:>4} "
              f"remove_small={r['remove_small']} mIoU={r['miou_grand']:.4f}")
    # per-query oracle
    keys = list(per[0].keys())
    oracle = [max(per[ci].get(k, float("nan")) for ci in range(len(cfgs))) for k in keys]
    print(f"\n=== per-query ORACLE upper bound = {np.mean(oracle):.4f} ({len(oracle)} queries) ===")
    print(f"=== best global config           = {rows[0]['miou_grand']:.4f} ===")
    if args.per_query_csv:
        import csv as _csv
        with open(args.per_query_csv, "a", newline="") as f:
            wr = _csv.writer(f)
            wr.writerow(["scene", "topk", "level_until", "filter_small", "remove_small", "image", "prompt", "iou"])
            for ci, c in enumerate(cfgs):
                for (im, pr), v in per[ci].items():
                    wr.writerow([args.scene, c["topk"], c["level_until"], c["filter_small"],
                                 int(c["remove_small"]), im, pr, round(v, 6)])
    json.dump(dict(scene=args.scene, best_global=rows[0], oracle=float(np.mean(oracle)),
                   n_queries=len(keys), n_configs=len(cfgs)),
              open(os.path.join(args.out, f"sweep_{args.scene}_summary.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
