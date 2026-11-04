"""Audit for the candidate-oracle run: leakage control + random-pool control.

Three numbers per query, all from the same 3D hierarchy:
  true   max IoU of C0 and the Deep pool (root top-10, level-1 top-10, level-2 top-10)
  random same-size pool sampled uniformly (size>50), fixed seed
  perm   the *true* pool scored against another query's GT (next in scene order):
         if candidate construction used GT, this would stay high

Also reports argmax-by-CLIP of the pool (a fixed label-free rule, not the oracle),
so one can see whether the semantic ranking itself carries the signal.
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import torch

REPO = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS"
RELAGS = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
for _p in (REPO, os.path.join(REPO, "deco", "lerf_ovs"), RELAGS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arguments import ModelParams, PipelineParams          # noqa: E402
from gaussian_renderer import render                      # noqa: E402
from nag_data import SemanticNAG                          # noqa: E402
from scene import GaussianModel, Scene                    # noqa: E402
from utils.vlm_utils import ClipSimMeasure                # noqa: E402
from eval_lerf_ovs import polygon_to_mask                 # noqa: E402

SRCROOT = "/root/rivermind-data/ReLaGS_Reproduction/ref/hf_only"
MODELROOT = "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf"
GTROOT = "/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label"


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()

    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(["-s", os.path.join(SRCROOT, args.scene),
                                   "-m", os.path.join(MODELROOT, args.scene)])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(MODELROOT, args.scene, "sai_nag.pt"), map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    root_level = len(snag.labels) - 1
    leaf_level = root_level - 1
    counts = {lv: torch.bincount(snag.labels[lv], minlength=int(snag.labels[lv].max()) + 1)
              for lv in range(len(snag.labels))}
    eligible = {lv: torch.nonzero(counts[lv] > 50).flatten().cpu()
                for lv in (1, 2, root_level)}

    gt_dir = os.path.join(GTROOT, args.scene)
    cams = {c.image_name: c for c in scene.getTrainCameras()}

    # pass 1: every query's GT, so the permutation can point elsewhere in the scene
    queries, gts = [], []
    for fn in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = fn[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams.get(name)
        if cam is None:
            continue
        h, w = cam.image_height, cam.image_width
        for prompt in dict.fromkeys(o["category"] for o in anno["objects"]):
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            queries.append((name, prompt, cam, h, w))
            gts.append(gt.astype(bool))

    rng = np.random.RandomState(args.seed)
    rows = []
    with torch.no_grad():
        for qi, (name, prompt, cam, h, w) in enumerate(queries):
            gt = gts[qi]
            perm_gt = None
            for off in range(1, len(gts)):
                cand = gts[(qi + off) % len(gts)]
                if cand.shape == gt.shape and not np.array_equal(cand, gt):
                    perm_gt = cand
                    break
            if perm_gt is None:
                continue

            vlm.encode_text(prompt)
            sims = [vlm.compute_similarity(f) for f in snag.feat]
            root_sim, root_ids = snag._get_root_candidates(sims, 50)
            rc = snag._build_root_candidates_with_leaf_info(
                root_ids, root_sim, sims, leaf_level)[:5]
            pre = sorted(snag._analyze_leaf_candidates(rc, leaf_level),
                         key=lambda x: x[3], reverse=True)
            c0 = snag._filter_by_similarity_gap([list(t) for t in pre], remove_small=True)
            c0_nodes = [(int(t[0]), int(t[2])) for t in c0]

            deep, deep_sims = [], []
            for lv, si in ((1, 0), (2, 1), (root_level, 2)):
                s = sims[si]
                vals, ids = torch.topk(s, min(20, s.shape[0]))
                kept = [(int(i), float(v)) for v, i in zip(vals, ids)
                        if int(counts[lv][int(i)]) > 50][:10]
                for i, v in kept:
                    if (lv, i) not in deep:
                        deep.append((lv, i))
                        deep_sims.append(v)

            rnd = []
            for lv in (1, 2, root_level):
                pool = eligible[lv].numpy()
                for i in rng.choice(pool, size=min(10, len(pool)), replace=False):
                    rnd.append((lv, int(i)))
            rnd = [nd for nd in rnd if nd not in deep]

            cache = {}

            def mask(nodes):
                key = frozenset(nodes)
                if key not in cache:
                    rel = torch.zeros(snag.gaussian_num, 1, dtype=torch.float32,
                                      device="cuda")
                    for lv, idx in nodes:
                        rel[snag.labels[lv] == idx, 0] = 1.0
                    gaussians._semantics = rel.expand(-1, 20)
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    cache[key] = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w)
                    cache[key] = cache[key].cpu().numpy().astype(bool)
                return cache[key]

            c0_m = mask(c0_nodes)
            c0_iou = iou(c0_m, gt)
            c0_perm = iou(c0_m, perm_gt)
            true_ious = [c0_iou] + [iou(mask([nd]), gt) for nd in deep]
            perm_ious = [c0_perm] + [iou(mask([nd]), perm_gt) for nd in deep]
            rand_ious = [c0_iou] + [iou(mask([nd]), gt) for nd in rnd]

            top1 = int(np.argmax(deep_sims)) if deep_sims else -1
            top1_iou = iou(mask([deep[top1]]), gt) if top1 >= 0 else 0.0

            rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                             c0=c0_iou, true_max=max(true_ious),
                             perm_max=max(perm_ious), random_max=max(rand_ious),
                             top1_iou=top1_iou,
                             n_deep=len(deep), n_random=len(rnd)))
            if (qi + 1) % 10 == 0:
                print("  %d/%d" % (qi + 1, len(queries)), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_audit.csv" % args.scene), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = dict(scene=args.scene, n=len(rows))
    for k in ("c0", "true_max", "perm_max", "random_max", "top1_iou"):
        v = np.array([r[k] for r in rows], float)
        out[k] = dict(mean=round(float(v.mean()), 4), median=round(float(np.median(v)), 4))
    out["true_gt_perm_by_0.05"] = int(sum(r["perm_max"] > r["c0"] + 0.05 for r in rows))
    out["true_gt_true_by_0.05"] = int(sum(r["true_max"] > r["c0"] + 0.05 for r in rows))
    print("AUDIT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()