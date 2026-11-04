#!/usr/bin/env python
"""Does the "+4.2 from not picking empty candidates" survive the official readout?

The per-view dump (cnone_gate_tf) rendered ONE candidate at a time, so 46.5% of rows
were empty. The official harness (eval_lerf_relags.py) renders the UNION of the gap
prefix, and an invisible cluster contributes zero pixels -- so dropping it may be a
no-op. This script measures that directly.

Variants, per (view, prompt):
  base : official gap prefix, union render
  V1   : drop prefix members invisible in this view (expected: identical pixels)
  V2   : re-derive the gap prefix after ordering by (visible, sim)
  V3   : if the prefix renders empty, fall back to the best VISIBLE candidate in the pool

No GT is used by any variant; GT only scores.
"""
import os, sys, json, csv, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import cv2

RE = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RE)

from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG


def polygon_to_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1)
    return m


def gap_index(cands):
    if not cands:
        return 0
    d = [0] + [cands[i][3] - cands[i + 1][3] for i in range(len(cands) - 1)]
    return int(np.argmax(d))


def main():
    ap = argparse.ArgumentParser()
    for k in ["scene", "source", "model", "nag", "gt_root", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level_until", type=int, default=3)
    ap.add_argument("--filter_small", type=int, default=50)
    ap.add_argument("--min_px", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(["-s", a.source, "-m", a.model])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    nag = torch.load(a.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    gt_dir = os.path.join(a.gt_root, a.scene)
    imgs = sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg"))
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    rows = []

    def render_set(cands, cam, h, w):
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, 20).cuda()
        o = render(cam, gaussians, pipe, bg)["semantics"]
        return (o.reshape(o.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()

    def iou(pred, gt):
        u = np.logical_or(pred, gt).sum()
        return float(np.logical_and(pred, gt).sum() / u) if u > 0 else float("nan")

    with torch.no_grad():
        for im in imgs:
            name = im[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            prompts = []
            for o in anno["objects"]:
                if o["category"] not in prompts:
                    prompts.append(o["category"])
            cam = cams.get(name)
            if cam is None:
                continue
            h, w = cam.image_height, cam.image_width
            for prompt in prompts:
                vlm.encode_text(prompt)
                sims = [vlm.compute_similarity(f) for f in snag.feat]
                _, cands = snag.search_matched_superpoint_in_mhtree(
                    sims, topk=a.topk, level_until=a.level_until,
                    filter_small=a.filter_small, remove_small=True)
                if not cands:
                    continue
                # rebuild the pre-gap pool to allow V2/V3
                root_level, leaf_level = len(snag.labels) - 1, len(snag.labels) - 2
                rv, ri, = snag._get_root_candidates(sims, a.filter_small)
                rc = snag._build_root_candidates_with_leaf_info(ri, rv, sims, leaf_level)[:a.topk]
                oc = sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)

                gt = np.zeros((h, w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                gt = gt.astype(bool)

                # per-candidate visibility in this view
                vis = {}
                for c in oc:
                    key = (c[0], c[2])
                    if key not in vis:
                        m = render_set([c], cam, h, w)
                        vis[key] = int(m.sum())
                visc = lambda c: vis[(c[0], c[2])] >= a.min_px

                preds = {}
                preds["base"] = render_set(cands, cam, h, w)
                preds["V1"] = render_set([c for c in cands if visc(c)], cam, h, w)
                o2 = sorted(oc, key=lambda c: (visc(c), c[3]), reverse=True)
                preds["V2"] = render_set(o2[:gap_index(o2)], cam, h, w)
                pfx = cands
                if all(not visc(c) for c in pfx):
                    vb = [c for c in oc if visc(c)]
                    pfx = [vb[0]] if vb else pfx
                preds["V3"] = render_set(pfx, cam, h, w)

                r = dict(scene=a.scene, image=name, prompt=prompt,
                         n_pool=len(oc), n_prefix=len(cands),
                         n_prefix_visible=sum(1 for c in cands if visc(c)),
                         prefix_matches_rebuild=int(
                             set((c[0], c[2]) for c in cands)
                             == set((c[0], c[2]) for c in oc[:gap_index(oc)])),
                         gt_px=int(gt.sum()))
                for k, p_ in preds.items():
                    r[k + "_iou"] = iou(p_, gt)
                    r[k + "_px"] = int(p_.sum())
                rows.append(r)
                if a.limit and len(rows) >= a.limit:
                    break
            print("  %s %s units=%d" % (a.scene, name, len(rows)), flush=True)
            if a.limit and len(rows) >= a.limit:
                break

    fn = list(rows[0].keys()) if rows else []
    path = os.path.join(a.out, "%s_official.csv" % a.scene)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fn)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print("WROTE %s rows=%d" % (path, len(rows)), flush=True)


if __name__ == "__main__":
    main()