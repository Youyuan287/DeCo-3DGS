"""LERF-OVS evaluation with the two DeCo operators, on top of the certified ReLaGS pipeline.

Arms (all share the same geometry, nag hierarchy, topk/level/filter settings):
  --center none|image          image-side common mode removal, per level
  --text canon|template|lexicon
        canon    : the released path (bare prompt + 4-word canon softmax)  -> reproduces 64.01
        template : tau-LSE over 4 prompt templates
        lexicon  : tau-LSE over a hand-written lexicon entry (fallback: bare prompt)
        lex_tpl  : lexicon + templates pooled in one bank
  --tau 0.02

Usage:
  PYTHONPATH=... python eval_lerf_relags_deco.py --scene ramen ... --center image --text template
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import torch

from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402

TEMPLATES = ("{}", "a photo of a {}", "the {}", "a {} in a room")


def polygon_to_mask(shape, pts):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(pts, np.int32)], 1)
    return mask


def l2n(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def lse_bank(feats, bank, tau):
    """tau-log-mean-exp of the cosine similarities against a bank; feats (N,D), bank (B,D)."""
    sc = feats @ bank.T
    m = sc.max(dim=1, keepdim=True).values
    return (tau * torch.log(torch.exp((sc - m) / tau).mean(dim=1, keepdim=True)) + m).squeeze(1)


def main():
    ap = ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level_until", type=int, default=3)
    ap.add_argument("--filter_small", type=int, default=50)
    ap.add_argument("--no_remove_small", action="store_true")
    ap.add_argument("--center", default="none", choices=["none", "image"])
    ap.add_argument("--text", default="canon", choices=["canon", "template", "lexicon", "lex_tpl", "dot"])
    ap.add_argument("--lexicon", default="")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--norm", default="none", choices=["none", "zscore"])
    ap.add_argument("--text-center", default="none", choices=["none", "global"])
    ap.add_argument("--renorm", type=int, default=1)
    ap.add_argument("--arbitrate", type=int, default=0)
    ap.add_argument("--text-agg", default="lse", choices=["lse", "mean", "canon_mean"])
    ap.add_argument("--sel", default="gap", choices=["gap", "ratio"])
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--tag", default="deco")
    args = ap.parse_args()

    p = ArgumentParser()
    lp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
    dataset = lp.extract(pargs)
    pipe = pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    if args.center == "image":
        # NOTE: the certified LERF-Mask run only subtracts the mean and does NOT renormalise
        if args.renorm:
            snag.feat = [l2n(f - f.mean(0, keepdim=True)) for f in snag.feat]
        else:
            snag.feat = [f - f.mean(0, keepdim=True) for f in snag.feat]
        print("[%s] image common mode removed from %d levels" % (args.scene, len(snag.feat)), flush=True)

    # the text-side common mode is estimated over the class set (all query categories of the scene)
    TEXT_MEAN = None
    lex = {}
    if args.lexicon and os.path.exists(args.lexicon):
        lex = json.load(open(args.lexicon))

    def scale_norm(sims):
        """Make levels comparable: the search compares scores across hierarchy levels."""
        if args.norm == "none":
            return sims
        return [(s - s.mean()) / s.std().clamp(min=1e-6) for s in sims]

    def text_sims(prompt):
        if args.text == "canon":
            vlm.encode_text(prompt)
            return [vlm.compute_similarity(f) for f in snag.feat]
        if args.text == "dot":
            # pure linear readout: score(i) = <f_i, t>. The scope theorem predicts that image-side
            # mean removal is a *constant* shift here, hence a bit-identical mask for every query.
            v = l2n(vlm.get_text_encoding([prompt]).float())[0]
            return [f @ v for f in snag.feat]
        words = [prompt] if args.text in ("template", "lexicon") else []
        if args.text in ("lexicon", "lex_tpl"):
            words = list(lex.get(prompt, [prompt]))
        bank_words = list(words)
        if args.text in ("template", "lex_tpl"):
            bank_words += [t.format(prompt) for t in TEMPLATES]
        bank = vlm.get_text_encoding(bank_words).float()
        if args.text_center == "global" and TEXT_MEAN is not None:
            bank = l2n(bank - TEXT_MEAN)          # remove the shared text mode (across the class set)
        else:
            bank = l2n(bank)
        if args.text_agg == "canon_mean":
            # keep the released canon readout (its distribution is what the candidate filter was
            # tuned on) and only marginalise over the words of the bank
            acc = None
            for w in bank_words:
                vlm.encode_text(w)
                sw = [vlm.compute_similarity(f) for f in snag.feat]
                acc = sw if acc is None else [a + b for a, b in zip(acc, sw)]
            return [a / float(len(bank_words)) for a in acc]
        if args.text_agg == "mean":
            # averaging synonyms denoises; LSE (max-like) favours whichever word happens to fire,
            # which on a single-query retrieval task creates false positives
            return [(f @ bank.T).mean(dim=1) for f in snag.feat]
        return [lse_bank(f, bank, args.tau) for f in snag.feat]

    gt_dir = os.path.join(args.gt_root, args.scene)
    if args.text_center == "global":
        _cats = []
        for _f in sorted(x for x in os.listdir(gt_dir) if x.endswith(".json")):
            for _o in json.load(open(os.path.join(gt_dir, _f))).get("objects", []):
                if _o["category"] not in _cats:
                    _cats.append(_o["category"])
        TEXT_MEAN = vlm.get_text_encoding(_cats).float().mean(0, keepdim=True)
        print("[%s] text common mode from %d categories" % (args.scene, len(_cats)), flush=True)
    imgs = sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg"))
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    rows, per_img = [], []
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
                print("WARN missing camera", name)
                continue
            h, w = cam.image_height, cam.image_width
            vals = []
            if args.arbitrate:
                # closed-set readout, i.e. the ScanNet-style argmax over the categories present in
                # this image, instead of scoring each query in isolation
                all_sims = {q: scale_norm(text_sims(q)) for q in prompts}
                assign = torch.stack([all_sims[q][0] for q in prompts], 0).argmax(0)
                for qi, prompt in enumerate(prompts):
                    gmask = (assign == qi).float()[snag.labels[1]].unsqueeze(1).cuda()
                    gaussians._semantics = gmask.expand(-1, 20)
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w)
                    pred = pred.cpu().numpy().astype(np.uint8)
                    gt = np.zeros((h, w), np.uint8)
                    for o in anno["objects"]:
                        if o["category"] == prompt:
                            gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                    inter = float(np.logical_and(pred, gt).sum())
                    union = float(np.logical_or(pred, gt).sum())
                    iou = inter / union if union > 0 else float("nan")
                    vals.append(iou)
                    rows.append(dict(scene=args.scene, image=name, prompt=prompt, iou=round(iou, 6),
                                     pred_px=int(pred.sum()), gt_px=int(gt.sum()), n_candidates=len(prompts)))
                m = float(np.mean(vals)) if vals else float("nan")
                per_img.append(m)
                print("  %s: %d queries (closed-set), mIoU=%.4f" % (name, len(prompts), m), flush=True)
                continue
            for prompt in prompts:
                sims = scale_norm(text_sims(prompt))
                if args.sel == "gap":
                    rel, cands = snag.search_matched_superpoint_in_mhtree(
                        sims, topk=args.topk, level_until=args.level_until,
                        filter_small=args.filter_small, remove_small=not args.no_remove_small)
                else:
                    # scale-invariant replacement for the gap heuristic: keep the candidates whose
                    # score is within `ratio` of the best one (the gap rule can return the empty set)
                    leaf_level = len(snag.labels) - 2
                    root_sim, root_ids = snag._get_root_candidates(sims, args.filter_small)
                    rc = snag._build_root_candidates_with_leaf_info(root_ids, root_sim, sims, leaf_level)
                    rc = rc[:args.topk]
                    cands = snag._analyze_leaf_candidates(rc, leaf_level)
                    cands = sorted(cands, key=lambda x: x[3], reverse=True)
                    if cands:
                        thr = cands[0][3] * args.ratio
                        kept = [c for c in cands if c[3] >= thr]
                    else:
                        kept = []
                    rel = snag._get_related_gaussians(kept) if kept else torch.zeros(snag.gaussian_num, 1,
                                                                                    dtype=torch.float32)
                    rel = rel.cuda()
                    cands = kept
                gaussians._semantics = rel.expand(-1, 20).cuda()
                sem = render(cam, gaussians, pipe, bg)["semantics"]
                pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w)
                pred = pred.cpu().numpy().astype(np.uint8)
                gt = np.zeros((h, w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                inter = float(np.logical_and(pred, gt).sum())
                union = float(np.logical_or(pred, gt).sum())
                iou = inter / union if union > 0 else float("nan")
                vals.append(iou)
                rows.append(dict(scene=args.scene, image=name, prompt=prompt, iou=round(iou, 6),
                                 pred_px=int(pred.sum()), gt_px=int(gt.sum()), n_candidates=len(cands)))
            m = float(np.mean(vals)) if vals else float("nan")
            per_img.append(m)
            print("  %s: %d queries, mIoU=%.4f" % (name, len(prompts), m), flush=True)

    scene_iou = float(np.mean(per_img)) if per_img else float("nan")
    flat = float(np.mean([r["iou"] for r in rows])) if rows else float("nan")
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scene", "image", "prompt", "iou", "pred_px", "gt_px", "n_candidates"])
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    print("SCENE_RESULT %s mIoU=%.4f flat=%.4f queries=%d images=%d" % (
        args.scene, scene_iou, flat, len(rows), len(per_img)))
    json.dump(dict(scene=args.scene, miou=scene_iou, flat_miou=flat, n_queries=len(rows),
                   per_image=per_img,
                   protocol=dict(topk=args.topk, level_until=args.level_until,
                                 filter_small=args.filter_small, center=args.center,
                                 text=args.text, tau=args.tau, norm=args.norm, text_center=args.text_center, sel=args.sel, ratio=args.ratio, renorm=args.renorm, text_agg=args.text_agg)),
              open(os.path.join(args.out, "%s_%s_summary.json" % (args.scene, args.tag)), "w"), indent=2)


if __name__ == "__main__":
    main()