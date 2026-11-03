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
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# honour RELAGS_REPO here, not only in the runners, so `--help` works as docs/README says
_REL = _os.environ.get("RELAGS_REPO")
if _REL:
    for _p in (_os.path.join(_REL, "ext"), _REL):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
from deco_ops import center_torch  # noqa: E402
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
from evidence_readout_v3 import relative_correction
from evidence_readout_v2 import (  # noqa: E402
    auxiliary_correction,
    descriptors_v2,
    load_bank_v2,
    load_mode_support,
)

TEMPLATES = ("{}", "a photo of a {}", "the {}", "a {} in a room")
V2_CORRECTION_COEFFICIENT = 0.25


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
    ap.add_argument("--renorm", type=int, default=0,
                    help="0 = f - mean (the setting behind -0.08); 1 = l2n(f - mean) (the -17 branch)")
    ap.add_argument("--arbitrate", type=int, default=0)
    ap.add_argument("--text-agg", default="lse", choices=["lse", "mean", "canon_mean"])
    ap.add_argument("--sel", default="gap", choices=["gap", "ratio"])
    ap.add_argument("--ratio", type=float, default=0.7)
    ap.add_argument("--tag", default="deco")
    ap.add_argument("--evidence-v2", default="off", choices=["off", "semantic", "reliable"])
    ap.add_argument("--evidence-bank", default="")
    ap.add_argument("--evidence-observations", default="")
    ap.add_argument("--evidence-members", default="")
    ap.add_argument("--evidence-temperature", type=float, default=0.08)
    ap.add_argument("--evidence-coefficient", type=float, default=V2_CORRECTION_COEFFICIENT)
    ap.add_argument("--evidence-aggregation", default="softmax",
                    choices=["softmax", "maxsim", "prior_maxsim"])
    ap.add_argument("--relative-v3", action="store_true")
    args = ap.parse_args()
    if args.evidence_v2 != "off" and not (args.evidence_bank and args.evidence_observations and args.evidence_members):
        raise ValueError("V2 auxiliary modes require --evidence-bank, --evidence-observations and --evidence-members")

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
    raw_root_mean = snag.feat[-1].mean(dim=0).clone()
    raw_root_count = int(snag.feat[-1].shape[0])
    raw_root_dim = int(snag.feat[-1].shape[1])
    vlm = ClipSimMeasure()
    vlm.load_model()

    evidence = None
    evidence_desc = None
    evidence_consistency = None
    evidence_view_count = None
    evidence_eligible = None
    if args.evidence_v2 != "off":
        evidence, evidence_meta = load_bank_v2(
            args.evidence_bank, args.nag, raw_root_count, raw_root_dim,
            observations_path=args.evidence_observations,
            members_path=args.evidence_members,
        )
        evidence_desc = descriptors_v2(
            evidence, snag.feat[-1].device,
            center=raw_root_mean if args.center == "image" else None,
            renorm=bool(args.renorm),
        )
        _, evidence_consistency, evidence_view_count, evidence_eligible = load_mode_support(
            args.evidence_observations, args.evidence_members, evidence
        )
        print("V2_BANK_META", evidence_meta, flush=True)

    if args.center == "image":
        # one definition, in deco_ops.py; --renorm 0 is what the reported -0.08 used
        snag.feat = center_torch(snag.feat, renorm=bool(args.renorm))
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

    def text_sims(prompt, feats=None):
        if feats is None:
            feats = snag.feat
        if args.text == "canon":
            vlm.encode_text(prompt)
            return [vlm.compute_similarity(f) for f in feats]
        if args.text == "dot":
            # pure linear readout: score(i) = <f_i, t>. The scope theorem predicts that image-side
            # mean removal is a *constant* shift here, hence a bit-identical mask for every query.
            v = l2n(vlm.get_text_encoding([prompt]).float())[0]
            return [f @ v for f in feats]
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
                sw = [vlm.compute_similarity(f) for f in feats]
                acc = sw if acc is None else [a + b for a, b in zip(acc, sw)]
            return [a / float(len(bank_words)) for a in acc]
        if args.text_agg == "mean":
            # averaging synonyms denoises; LSE (max-like) favours whichever word happens to fire,
            # which on a single-query retrieval task creates false positives
            return [(f @ bank.T).mean(dim=1) for f in feats]
        return [lse_bank(f, bank, args.tau) for f in feats]

    def select_candidates(sims):
        """Exact copy of the original DeCo hierarchy selection."""
        if args.sel == "gap":
            return snag.search_matched_superpoint_in_mhtree(
                sims, topk=args.topk, level_until=args.level_until,
                filter_small=args.filter_small, remove_small=not args.no_remove_small)
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
        rel = snag._get_related_gaussians(kept) if kept else torch.zeros(
            snag.gaussian_num, 1, dtype=torch.float32)
        return rel.cuda(), kept

    spatial_cache = {}
    def spatial_records(cands):
        out = []
        for c in cands:
            lev, rid, cid = int(c[0]), int(c[1]), int(c[2])
            kk = (lev,rid,cid)
            if kk not in spatial_cache:
                selected = torch.nonzero(snag.labels[lev] == cid, as_tuple=True)[0]
                xyz = gaussians.get_xyz[selected]
                spatial_cache[kk] = dict(level=lev,root_id=rid,candidate_id=cid,
                    gaussian_count=len(selected),xyz_min=xyz.min(0).values.tolist(),
                    xyz_max=xyz.max(0).values.tolist())
            out.append(dict(**spatial_cache[kk],score=float(c[3])))
        return out

    def candidate_records(cands):
        return [dict(level=int(c[0]), root_id=int(c[1]), candidate_id=int(c[2]),
                     score=float(c[3]), gaussian_count=int(c[6])) for c in cands]

    def render_iou(rel, cam, h, w, anno, prompt):
        gaussians._semantics = rel.expand(-1, 20).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w)
        pred = pred.cpu().numpy().astype(np.uint8)
        gt = np.zeros((h, w), np.uint8)
        for o in anno["objects"]:
            if o["category"] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
        inter = int(np.logical_and(pred, gt).sum())
        union = int(np.logical_or(pred, gt).sum())
        iou = inter / union if union > 0 else float("nan")
        return float(iou), int(pred.sum()), int(gt.sum()), inter, pred

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
                # V2 keeps this branch separate from the original DeCo path
                # below. The original branch is intentionally unchanged.
                if args.evidence_v2 != "off":
                    main_raw = text_sims(prompt)
                    main_sims = scale_norm(main_raw)
                    slot_scores = text_sims(prompt, feats=[evidence_desc])[0].reshape(evidence["priors"].shape)
                    reliability = (np.ones_like(evidence_consistency, dtype=np.float32)
                                   if args.evidence_v2 == "semantic" else evidence_consistency)
                    main_rel, main_cands = select_candidates(main_sims)
                    comparison_ids = sorted({int(c[1]) for c in main_cands})
                    corrected_root, detail = auxiliary_correction(
                        main_raw[-1], slot_scores, evidence, evidence_eligible,
                        reliability, temperature=args.evidence_temperature,
                        coefficient=args.evidence_coefficient,
                        method=args.evidence_aggregation,
                    )
                    if args.relative_v3:
                        corrected_root, detail = relative_correction(
                            main_raw[-1], slot_scores, evidence, evidence_eligible,
                            evidence_consistency, comparison_ids,
                            temperature=args.evidence_temperature,
                            coefficient=args.evidence_coefficient)
                    corrected_raw = list(main_raw)
                    corrected_raw[-1] = corrected_root
                    corrected_sims = scale_norm(corrected_raw)
                    new_rel, new_cands = select_candidates(corrected_sims)
                    main_iou, main_px, gt_px, main_inter, main_mask = render_iou(main_rel, cam, h, w, anno, prompt)
                    new_iou, new_px, _, new_inter, new_mask = render_iou(new_rel, cam, h, w, anno, prompt)
                    import hashlib
                    main_ids = torch.nonzero(main_rel.reshape(-1) > 0, as_tuple=True)[0].cpu().numpy()
                    new_ids = torch.nonzero(new_rel.reshape(-1) > 0, as_tuple=True)[0].cpu().numpy()
                    mask_xor = int(np.count_nonzero(main_mask != new_mask))
                    diagnostics = dict(gaussian_set_equal=bool(np.array_equal(main_ids,new_ids)),
                        gaussian_added=int(len(np.setdiff1d(new_ids,main_ids))),
                        gaussian_removed=int(len(np.setdiff1d(main_ids,new_ids))),
                        mask_xor_pixels=mask_xor, mask_equal=mask_xor==0,
                        main_mask_sha256=hashlib.sha256(main_mask.tobytes()).hexdigest(),
                        new_mask_sha256=hashlib.sha256(new_mask.tobytes()).hexdigest(),
                        comparison_root_ids=json.dumps(comparison_ids))
                    diagnostics['main_spatial_bounds'] = json.dumps(spatial_records(main_cands))
                    diagnostics['new_spatial_bounds'] = json.dumps(spatial_records(new_cands))
                    diagnostics['main_pred_gaussians'] = len(main_ids)
                    diagnostics['new_pred_gaussians'] = len(new_ids)
                    if args.relative_v3:
                        assert args.evidence_v2 == 'reliable'
                    delta_np = detail["delta"].detach().cpu().numpy()
                    corr_np = detail["correction"].detach().cpu().numpy()
                    adv_np = detail["advantage"].detach().cpu().numpy()
                    wt_np = detail["query_weights"].detach().cpu().numpy()
                    rel_np = detail["reliability"].detach().cpu().numpy()
                    active = []
                    for ri, rid in enumerate(evidence["region_ids"].tolist()):
                        for k in range(evidence["priors"].shape[1]):
                            if detail["eligible"][ri, k].item() and (adv_np[ri, k] > 0 or corr_np[ri] > 0):
                                active.append(dict(region_id=int(rid), mode_id=int(k),
                                                   main_score=float(main_raw[-1][int(rid)]),
                                                   mode_score=float(slot_scores[ri, k]),
                                                   advantage=float(adv_np[ri, k]),
                                                   reliability=float(rel_np[ri, k]),
                                                   query_weight=float(wt_np[ri, k]),
                                                   delta=float(delta_np[ri]),
                                                   correction=float(corr_np[ri]),
                                                   view_count=int(evidence_view_count[ri, k])))
                    main_cand_records = candidate_records(main_cands)
                    new_cand_records = candidate_records(new_cands)
                    fp_main = main_px - main_inter
                    fp_new = new_px - new_inter
                    rows.append(dict(
                        **diagnostics, scene=args.scene, image=name, prompt=prompt, iou=round(new_iou, 6),
                        main_iou=round(main_iou, 6), pred_px=new_px, main_pred_px=main_px,
                        gt_px=gt_px, n_candidates=len(new_cands), main_n_candidates=len(main_cands),
                        main_candidate_regions=json.dumps(main_cand_records, separators=(",", ":")),
                        new_candidate_regions=json.dumps(new_cand_records, separators=(",", ":")),
                        gt_coverage_gain=int(new_inter - main_inter),
                        background_false_positive_change=int(fp_new - fp_main),
                        auxiliary_triggered=bool(active),
                        auxiliary_details=json.dumps(active, separators=(",", ":")),
                        raw_delta_max=float(delta_np.max()) if len(delta_np) else 0.0,
                        final_correction_max=float(corr_np.max()) if len(corr_np) else 0.0,
                    ))
                    vals.append(new_iou)
                    continue
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
    with open(args.csv, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["scene", "image", "prompt", "iou"]
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    print("SCENE_RESULT %s mIoU=%.4f flat=%.4f queries=%d images=%d" % (
        args.scene, scene_iou, flat, len(rows), len(per_img)))
    summary = dict(scene=args.scene, miou=scene_iou, flat_miou=flat, n_queries=len(rows),
                   per_image=per_img,
                   protocol=dict(topk=args.topk, level_until=args.level_until,
                                 filter_small=args.filter_small, center=args.center,
                                 text=args.text, tau=args.tau, norm=args.norm, text_center=args.text_center,
                                 sel=args.sel, ratio=args.ratio, renorm=args.renorm, text_agg=args.text_agg,
                                 evidence_v2=args.evidence_v2, evidence_bank=args.evidence_bank,
                                 evidence_temperature=args.evidence_temperature,
                                 correction_coefficient=args.evidence_coefficient,
                                 evidence_aggregation=args.evidence_aggregation))
    if args.evidence_v2 != "off" and rows:
        base = [float(r["main_iou"]) for r in rows]
        new = [float(r["iou"]) for r in rows]
        diffs = np.asarray(new) - np.asarray(base)
        summary.update(dict(main_miou=float(np.mean(base)), main_flat_miou=float(np.mean(base)),
                            improved_queries=int((diffs > 1e-9).sum()),
                            degraded_queries=int((diffs < -1e-9).sum()),
                            unchanged_queries=int((np.abs(diffs) <= 1e-9).sum()),
                            total_iou_gain=float(diffs[diffs > 0].sum()),
                            total_iou_loss=float(-diffs[diffs < 0].sum()),
                            triggered_queries=int(sum(bool(r.get("auxiliary_triggered")) for r in rows))))
    json.dump(summary, open(os.path.join(args.out, "%s_%s_summary.json" % (args.scene, args.tag)), "w"), indent=2)


if __name__ == "__main__":
    main()
