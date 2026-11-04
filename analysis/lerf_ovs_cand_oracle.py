"""LERF-OVS: how much of the released 3D hierarchy is recoverable by re-selection alone?

For every query (scene, frame, category) this reproduces the released readout exactly
(center=none, text=canon, topk=5, level_until=3, filter_small=50, sel=gap) and then
renders *additional* candidates that the released readout discards, to measure four
oracles under the official aggregation (per-image mean -> per-scene mean -> 4-scene macro):

  C0          the released gap selection (baseline, must reproduce 0.6401)
  root        C0 + each of the top-5 root nodes, taken individually
  hierarchy   C0 + every pre-gap node from _analyze_leaf_candidates (roots kept whole
              and promoted level-2 leaves), before _filter_by_similarity_gap
  combo       hierarchy + constrained unions built only from the pre-gap pool:
                - swap one C0 node for one non-C0 pool node
                - add one non-C0 pool node to C0
                - top-8 pool node pairs
              unions containing an ancestor/descendant pair are forbidden
  deep        (beyond the released protocol, reported separately) top-10 level-1 and
              top-10 level-2 nodes by their own CLIP sim, plus the top-10 roots

Nothing is fitted.  GT is only used to score already-built candidates; no GT ever
enters candidate construction.  Candidate budgets are fixed and label-free.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time

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

# fixed budgets (label-free)
DEEP_PER_LEVEL = 10
SWAP_BUDGET = 24
ADD_BUDGET = 8
PAIR_BUDGET = 12
PAIR_POOL = 8


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    scene_name = args.scene
    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(["-s", os.path.join(SRCROOT, scene_name),
                                   "-m", os.path.join(MODELROOT, scene_name)])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(MODELROOT, scene_name, "sai_nag.pt"), map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    root_level = len(snag.labels) - 1
    leaf_level = root_level - 1
    assert root_level == 3, "expected 4 hierarchy levels"

    # per-level id counts and one member index per id (for ancestor tests)
    counts, first_idx = {}, {}
    for lv in range(len(snag.labels)):
        lab = snag.labels[lv]
        n_ids = int(lab.max()) + 1
        counts[lv] = torch.bincount(lab, minlength=n_ids)
        f = torch.full((n_ids,), -1, dtype=torch.long, device=lab.device)
        f[lab] = torch.arange(lab.numel(), device=lab.device)
        first_idx[lv] = f

    def contains(a, b):
        """node a (coarser) contains node b."""
        (la, ia), (lb, ib) = a, b
        if la == lb:
            return ia == ib
        if la < lb:
            return contains(b, a)
        return int(snag.labels[la][first_idx[lb][ib]]) == ia

    def conflicts(nodes):
        ns = list(nodes)
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                if contains(ns[i], ns[j]):
                    return True
        return False

    def mask_of(nodes):
        rel = torch.zeros(snag.gaussian_num, 1, dtype=torch.float32, device="cuda")
        for lv, idx in nodes:
            rel[snag.labels[lv] == idx, 0] = 1.0
        return rel

    def sims_for(prompt):
        vlm.encode_text(prompt)
        return [vlm.compute_similarity(f) for f in snag.feat]

    gt_dir = os.path.join(GTROOT, scene_name)
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    out_rows, cand_rows = [], []
    n_rendered = 0
    t0 = time.time()

    with torch.no_grad():
        for fn in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
            name = fn[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            cam = cams.get(name)
            if cam is None:
                print("WARN missing camera", name, flush=True)
                continue
            h, w = cam.image_height, cam.image_width
            prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
            for prompt in prompts:
                if args.limit_queries and len(out_rows) >= args.limit_queries:
                    break
                gt = np.zeros((h, w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                gt = gt.astype(bool)

                sims = sims_for(prompt)
                root_sim, root_ids = snag._get_root_candidates(sims, 50)
                rc = snag._build_root_candidates_with_leaf_info(
                    root_ids, root_sim, sims, leaf_level)[:5]
                pre = snag._analyze_leaf_candidates(rc, leaf_level)
                pre = sorted(pre, key=lambda x: x[3], reverse=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    c0 = snag._filter_by_similarity_gap(
                        [list(t) for t in pre], remove_small=True)

                pre_nodes = []
                for t in pre:
                    nd = (int(t[0]), int(t[2]))
                    if nd not in pre_nodes:
                        pre_nodes.append(nd)
                root_nodes = []
                for t in rc:
                    nd = (root_level, int(t[1]))
                    if nd not in root_nodes:
                        root_nodes.append(nd)
                c0_nodes = [(int(t[0]), int(t[2])) for t in c0]

                deep_nodes = []
                for lv, sim_idx, budget in ((1, 0, DEEP_PER_LEVEL),
                                            (2, 1, DEEP_PER_LEVEL),
                                            (root_level, 2, DEEP_PER_LEVEL)):
                    s = sims[sim_idx]
                    vals, ids = torch.topk(s, min(20, s.shape[0]))
                    kept = [int(i) for v, i in zip(vals, ids)
                            if int(counts[lv][int(i)]) > 50]
                    for i in kept[:budget]:
                        nd = (lv, i)
                        if nd not in deep_nodes:
                            deep_nodes.append(nd)

                # constrained combinations, all built from the label-free pool
                pool = pre_nodes
                pool_extra = [nd for nd in pool if nd not in c0_nodes]
                combos, seen = [], set()

                def add_combo(nodes):
                    key = frozenset(nodes)
                    if not nodes or key in seen or conflicts(nodes):
                        return False
                    seen.add(key)
                    combos.append(tuple(sorted(nodes)))
                    return True

                n_swap = 0
                for rem in c0_nodes:
                    for alt in pool_extra:
                        if n_swap >= SWAP_BUDGET:
                            break
                        if add_combo([nd for nd in c0_nodes if nd != rem] + [alt]):
                            n_swap += 1
                    if n_swap >= SWAP_BUDGET:
                        break
                n_add = 0
                for alt in pool_extra:
                    if n_add >= ADD_BUDGET:
                        break
                    if add_combo(c0_nodes + [alt]):
                        n_add += 1
                n_pair = 0
                for a in range(min(PAIR_POOL, len(pool))):
                    for b in range(a + 1, min(PAIR_POOL, len(pool))):
                        if n_pair >= PAIR_BUDGET:
                            break
                        if add_combo([pool[a], pool[b]]):
                            n_pair += 1
                    if n_pair >= PAIR_BUDGET:
                        break

                cache = {}

                def score(nodes, kind):
                    nonlocal n_rendered
                    key = frozenset(nodes)
                    if key not in cache:
                        gaussians._semantics = mask_of(nodes).expand(-1, 20)
                        sem = render(cam, gaussians, pipe, bg)["semantics"]
                        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w)
                        cache[key] = iou(pred.cpu().numpy().astype(bool), gt)
                        n_rendered += 1
                    v = cache[key]
                    cand_rows.append(dict(
                        scene=scene_name, image=name, prompt=prompt, kind=kind,
                        level=int(nodes[0][0]) if len(nodes) == 1 else -1,
                        index=int(nodes[0][1]) if len(nodes) == 1 else -1,
                        n_nodes=len(nodes), iou=round(v, 6)))
                    return v

                c0_iou = score(c0_nodes, "c0")
                root_vals = [score([nd], "root") for nd in root_nodes]
                hier_vals = [score([nd], "hier") for nd in pre_nodes]
                deep_vals = [score([nd], "deep") for nd in deep_nodes]
                combo_vals = [score(list(cb), "combo") for cb in combos]

                best = {}
                def best_of(kind, vals, cands):
                    if not vals:
                        best[kind] = (float("nan"), "")
                        return float("nan")
                    k = int(np.argmax(vals))
                    best[kind] = (vals[k], str(cands[k]))
                    return vals[k]

                root_o = max([c0_iou] + root_vals)
                hier_o = max([c0_iou] + root_vals + hier_vals)
                combo_o = max([c0_iou] + root_vals + hier_vals + combo_vals)
                deep_o = max([c0_iou] + root_vals + deep_vals)
                grand_o = max(combo_o, deep_o)
                best_of("root", root_vals, root_nodes)
                best_of("hier", hier_vals, pre_nodes)
                best_of("combo", combo_vals, combos)
                best_of("deep", deep_vals, deep_nodes)

                out_rows.append(dict(
                    scene=scene_name, image=name, prompt=prompt,
                    c0_iou=round(c0_iou, 6),
                    root_oracle=round(root_o, 6),
                    hier_oracle=round(hier_o, 6),
                    combo_oracle=round(combo_o, 6),
                    deep_oracle=round(deep_o, 6),
                    grand_oracle=round(grand_o, 6),
                    n_root=len(root_nodes), n_pre=len(pre_nodes),
                    n_combo=len(combos), n_deep=len(deep_nodes),
                    n_unique_renders=len(cache),
                    c0_levels=";".join("%d:%d" % nd for nd in c0_nodes),
                    best_root=str(best["root"][1]), best_root_iou=best["root"][0],
                    best_hier=str(best["hier"][1]), best_hier_iou=best["hier"][0],
                    best_combo=str(best["combo"][1]), best_combo_iou=best["combo"][0],
                    best_deep=str(best["deep"][1]), best_deep_iou=best["deep"][0]))
            print("IMG %s q=%d renders=%d elapsed=%.0fs" % (
                name, len(prompts), n_rendered, time.time() - t0), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_perquery.csv" % scene_name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wr.writeheader()
        wr.writerows(out_rows)
    with open(os.path.join(args.out, "%s_candidates.csv" % scene_name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(cand_rows[0]))
        wr.writeheader()
        wr.writerows(cand_rows)
    print("SCENE_DONE", scene_name, len(out_rows), "renders", n_rendered,
          "secs %.0f" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()