"""3D-OVS evaluation with the two DeCo operators, on top of the ReLaGS pipeline.

Why this file exists: DeCo ships harnesses for ScanNet-10, LERF-Mask and LERF-OVS
only.  3D-OVS (Liu et al., NeurIPS 2023) has a different annotation format --
per-class *binary PNG masks* on 5 labelled views per scene, instead of LERF's
per-image JSON polygons -- so the readout is re-implemented here while the
candidate search, the renderer and the two operators are reused verbatim from
``deco/lerf_ovs/eval_lerf_ovs.py`` and ``deco_ops.py``.

Two readouts (``--mode``):

  prompt  (default) the literature readout: one prompt at a time, the matched
          superpoints become a binary per-Gaussian selection, the mask is
          rasterised and thresholded at 0.5.  Every 3D-OVS number published by
          LangSplat / THGS / ReLaGS-style pipelines is produced this way.
          This is a *single-query* readout, so -- per deco/PROTOCOLS.md §4 --
          C1 (mean removal) is a constant shift and cannot move the ranking.

  multi   the ScanNet-style readout: all scene classes are scored at once, each
          Gaussian takes its argmax class (background = channel 19), the class
          map is rasterised and every class mask is read off it.  This is the
          readout the DeCo operators were designed for; it is *not* the
          protocol the published 3D-OVS numbers use, and it is reported as an
          extra column, never as a replacement.

Arms (identical geometry / nag hierarchy / topk / level / filter settings):
  --center none|image      C1, image-side common-mode removal, per level
  --text canon|template    C2, tau-LSE over prompt templates (canon = bare prompt)
  --renorm 0|1             0 = f - mean (DeCo default), 1 = l2n(f - mean)

Protocol: per labelled view, mIoU over that view's classes, then mean over the
5 views (``SCENE_RESULT``); the flat mean over every (view, class) pair is also
reported for reference.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402

TEMPLATES = ("{}", "a photo of a {}", "the {}", "a {} in a room")
OVS_TEMPLATES = ("{}", "a photo of a {}", "the {}", "a close-up photo of a {}")
N_CHANNELS = 20          # must match GaussianModel(..., 20); last channel = background
BG_CHANNEL = N_CHANNELS - 1


def l2n(x, eps=1e-12):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def lse_bank(feats, bank, tau):
    """tau-log-mean-exp of cosine similarities; feats (N,D), bank (B,D)."""
    sc = feats @ bank.T
    m = sc.max(dim=1, keepdim=True).values
    return (tau * torch.log(torch.exp((sc - m) / tau).mean(dim=1, keepdim=True)) + m).squeeze(1)


def read_mask(path):
    """GT mask at its native resolution (3D-OVS ships 4096x3072 PNGs)."""
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    return (gt > 0).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True, help="dir containing <scene>/segmentations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level_until", type=int, default=3)
    ap.add_argument("--filter_small", type=int, default=50)
    ap.add_argument("--no_remove_small", action="store_true")
    ap.add_argument("--cand-mode", default="gap",
                    choices=["gap", "gap2", "thresh", "all", "top1"],
                    help="how superpoint candidates are truncated")
    ap.add_argument("--cand-margin", type=float, default=0.02,
                    help="thresh mode: keep candidates with sim >= best - margin")
    ap.add_argument("--mode", default="prompt", choices=["prompt", "multi", "oracle", "pixsim", "gthresh"],
                    help="prompt = per-class single-query (literature); multi = argmax class map; "
                         "oracle = select every Gaussian that projects inside the GT mask (ceiling test)")
    ap.add_argument("--pred-thresh", type=float, default=0.5,
                    help="rasterised-semantics threshold for the binary prediction (default 0.5)")
    ap.add_argument("--pixsim-level", type=int, default=1,
                    help="pixsim mode: hierarchy level whose cluster CLIP features are attached to "
                         "every Gaussian (1 = finest level that carries features)")
    ap.add_argument("--pixsim-thr", default="otsu",
                    help="pixsim mode: 'otsu' = per-class Otsu on the rendered similarity map, "
                         "or a float absolute threshold")
    ap.add_argument("--expand-levels", default="",
                    help="comma separated hierarchy levels (e.g. '2,1'): after candidate "
                         "selection, also select the ancestors of every selected cluster at "
                         "those levels (coverage fix for big low-contrast classes)")
    ap.add_argument("--norm-alpha", action="store_true",
                    help="divide the rendered selection by the selection's own accumulated alpha, "
                         "i.e. report the alpha-weighted MEAN membership of the visible Gaussians "
                         "instead of the unnormalised sum; a class whose Gaussians are thin or "
                         "low-opacity then still reads ~1.0 where it is what the view sees")
    ap.add_argument("--empty-fallback", default="none", choices=["none", "biggest"],
                    help="when the rendered mask of a class is empty (under-covered selection), "
                         "re-select from the full candidate pool the k largest clusters whose "
                         "similarity is within --fb-margin of the best one, and render again")
    ap.add_argument("--fb-margin", type=float, default=0.15)
    ap.add_argument("--fb-k", type=int, default=2)
    ap.add_argument("--size-lambda", type=float, default=0.0,
                    help="re-rank candidates by sim + lambda*log10(n_points) before truncation; "
                         "biases the search towards large regions without looking at the GT")
    ap.add_argument("--mask-render", default="blend", choices=["blend", "kill"],
                    help="blend = rasterise the binary selection over the whole scene (shipped "
                         "behaviour: unselected Gaussians keep their opacity, so a selected "
                         "Gaussian hidden behind them contributes < 0.5 weight and the mask can "
                         "come out empty); kill = render the selection as its own scene by "
                         "zeroing the opacity of unselected Gaussians, so the mask is exactly the "
                         "screen coverage of the selection")
    ap.add_argument("--debug-cands", action="store_true",
                    help="print every retrieved candidate (level, index, sim, n_points) for the "
                         "first labelled view and exit -- fact-finding, renders nothing")
    ap.add_argument("--gthresh-k", type=float, default=1.0,
                    help="gthresh mode: keep clusters with z-scored similarity >= mean + k*std")
    ap.add_argument("--gthresh-level", type=int, default=1,
                    help="gthresh mode: hierarchy level whose clusters are thresholded")
    ap.add_argument("--center", default="none", choices=["none", "image"])
    ap.add_argument("--text", default="canon", choices=["canon", "template", "dot"])
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--norm", default="none", choices=["none", "zscore"])
    ap.add_argument("--renorm", type=int, default=0)
    ap.add_argument("--text-agg", default="lse", choices=["lse", "mean"])
    ap.add_argument("--text-center", default="none", choices=["none", "global"],
                    help="C2a (deco_ops.text_center): t <- l2n(t - mean_class_set(t)). 'global' "
                         "subtracts the mean embedding of the scene's class set before scoring; "
                         "'none' leaves the bank uncentred (C2b/LSE only, which is NOT C2)")
    ap.add_argument("--tpl", default="lerf",
                    help="prompt-template set: 'lerf' (DeCo's 4 LERF templates), 'ovs' (indoor/outdoor "
                         "neutral, drops 'a {} in a room'), or a '|'-separated custom list")
    ap.add_argument("--tag", default="deco")
    ap.add_argument("--save_masks", default="")
    ap.add_argument("--resolution", type=int, default=4,
                    help="must match the resolution 2DGS was trained at")
    args = ap.parse_args()

    p = argparse.ArgumentParser()
    lp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model, "-r", str(args.resolution)])
    dataset = lp.extract(pargs)
    pipe = pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, N_CHANNELS)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    if args.center == "image":
        snag.feat = center_torch(snag.feat, renorm=bool(args.renorm))
        print("[%s] image common mode removed from %d levels" % (args.scene, len(snag.feat)), flush=True)

    # C2a text common mode; filled in once classes.txt has been read (below)
    TEXT_MEAN = None

    def templates():
        if args.tpl == "lerf":
            return TEMPLATES
        if args.tpl == "ovs":
            return OVS_TEMPLATES
        return tuple(args.tpl.split("|"))

    def scale_norm(sims):
        if args.norm == "none":
            return sims
        return [(s - s.mean()) / s.std().clamp(min=1e-6) for s in sims]

    def text_sims(prompt):
        if args.text == "canon":
            vlm.encode_text(prompt)
            return [vlm.compute_similarity(f) for f in snag.feat]
        if args.text == "dot":
            v = l2n(vlm.get_text_encoding([prompt]).float())[0]
            return [f @ v for f in snag.feat]
        bank_words = [t.format(prompt) for t in templates()]
        bank = vlm.get_text_encoding(bank_words).float()
        if args.text_center == "global" and TEXT_MEAN is not None:
            # C2a: remove the shared text mode estimated over the class set (deco_ops.text_center)
            bank = l2n(bank - TEXT_MEAN)
        else:
            bank = l2n(bank)
        if args.text_agg == "mean":
            return [(f @ bank.T).mean(dim=1) for f in snag.feat]
        return [lse_bank(f, bank, args.tau) for f in snag.feat]

    def candidates(cls):
        """-> (candidates, n_after_truncation) for one prompt, under --cand-mode."""
        sims = scale_norm(text_sims(cls))
        # cand-mode gap   = shipped behaviour: prefix above the largest similarity gap
        # cand-mode gap2  = same, but without the leading-0 padding tie bug
        # cand-mode thresh= keep every candidate within `cand_margin` of the best one
        # cand-mode all   = keep every candidate (upper bound on recall)
        # cand-mode top1  = keep only the best candidate
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=args.topk, level_until=args.level_until,
            filter_small=args.filter_small,
            remove_small=(args.cand_mode == "gap") and (not args.no_remove_small))
        if args.size_lambda > 0 and cands:
            import math

            def _boost(c):
                try:
                    a = np.asarray(c[4]).reshape(-1)
                    n = float(a[0]) if a.size else 1.0
                except (TypeError, ValueError):
                    n = 1.0
                return float(c[3]) + args.size_lambda * math.log10(max(n, 1.0))

            cands = sorted(cands, key=_boost, reverse=True)
            cands = [tuple(list(c[:3]) + [_boost(c)] + list(c[4:])) for c in cands]
            print("    size-lambda %.3f: re-ranked %d candidates (top size %s, top sim %.4f)"
                  % (args.size_lambda, len(cands),
                     [int(float(c[4])) if c[4] is not None else -1 for c in cands[:5]],
                     float(cands[0][3])), flush=True)
        if args.cand_mode != "gap" and cands:
            if args.cand_mode == "thresh":
                best = max(c[3] for c in cands)
                cands = [c for c in cands if c[3] >= best - args.cand_margin]
            elif args.cand_mode == "gap2":
                # shipped gap filter pads the delta list with a leading 0 and takes
                # .index(max(...)), so a tie at 0 (all candidates equally similar)
                # truncates to the empty set -> IoU 0 on exactly the big/ambiguous
                # classes (e.g. sofa "grey sofa").  Padding gone, prefix inclusive.
                sc = [c[3] for c in cands]
                d = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
                k = (max(range(len(d)), key=lambda i: d[i]) + 1) if d else 1
                cands = cands[:k]
            elif args.cand_mode == "top1":
                cands = cands[:1]
        if args.expand_levels and cands:
            # A class whose selected clusters carry < 50% of the blending weight on its own
            # pixels renders as an empty mask (the failure mode measured on "green lawn" /
            # "wood wall" / "grey sofa").  Climb from each selected cluster to its ancestors
            # at the given levels: a big low-contrast class is covered by its ancestor, even
            # when the ancestor's own similarity is too low to be retrieved directly.
            want = [int(x) for x in str(args.expand_levels).split(",") if x.strip() != ""]
            extra = []
            for L in want:
                if L < 1 or L >= len(snag.labels):
                    continue
                for tup in cands:
                    lvl, idx = int(tup[0]), int(tup[2])
                    members = torch.where(snag.labels[lvl] == idx)[0]
                    if members.numel() == 0:
                        continue
                    for v in torch.unique(snag.labels[L][members]).tolist():
                        extra.append((L, -1, int(v), 0.0, int(members.numel()), 0.0, 0.0))
            if extra:
                cands = list(cands) + extra
                if args.cand_mode == "top1":
                    cands = cands  # ancestors are wanted even with top1
        return cands

    def pool(cls):
        """Untruncated candidate pool, used by the empty-mask fallback only."""
        sims = scale_norm(text_sims(cls))
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=args.topk, level_until=args.level_until,
            filter_small=args.filter_small, remove_small=False)
        return cands

    def to_gt(mask_hw, gt):
        """Prediction is up-sampled to the GT grid, then thresholded (THGS/LangSplat)."""
        return cv2.resize(mask_hw.cpu().numpy().astype(np.uint8),
                          (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0

    def render_rel(cam, rel, force_kill=False):
        """Rasterise a 0/1 Gaussian selection.

        ``blend`` keeps every Gaussian's opacity (shipped ReLaGS behaviour): the selected
        Gaussians are blended with everything else, so a selection that is occluded, thin, or
        behind other geometry picks up < 0.5 of the pixel weight and the thresholded mask is
        empty.  ``kill`` zeroes the opacity of the unselected Gaussians for this render only,
        which makes the mask the screen coverage of the selection itself -- the closest
        Gaussian-side analogue of the dense per-pixel feature field the literature thresholds.
        """
        gaussians._semantics = rel.expand(-1, N_CHANNELS).cuda()
        if args.mask_render == "kill" or force_kill:
            orig = gaussians._opacity
            kept = (rel.reshape(-1).cuda() > 0.5)
            dead = torch.full_like(orig.detach(), -30.0)
            gaussians._opacity = torch.where(kept[:, None], orig.detach(), dead)
            try:
                return render(cam, gaussians, pipe, bg)["semantics"]
            finally:
                gaussians._opacity = orig
        return render(cam, gaussians, pipe, bg)["semantics"]

    def score(pred, gt, view, cls, n_cands):
        inter = float(np.logical_and(pred, gt).sum())
        union = float(np.logical_or(pred, gt).sum())
        iou = inter / union if union > 0 else float("nan")
        rows.append(dict(scene=args.scene, image=view, prompt=cls, iou=round(iou, 6),
                         pred_px=int(pred.sum()), gt_px=int(gt.sum()),
                         n_candidates=n_cands))
        if args.save_masks:
            os.makedirs(args.save_masks, exist_ok=True)
            cv2.imwrite(os.path.join(args.save_masks, "%s__%s.png" % (view, cls)), pred.astype(np.uint8) * 255)
        return iou

    seg_root = os.path.join(args.gt_root, args.scene, "segmentations")
    classes = [c.strip() for c in open(os.path.join(seg_root, "classes.txt")) if c.strip()]
    views = sorted(d for d in os.listdir(seg_root)
                   if os.path.isdir(os.path.join(seg_root, d)))
    if args.text_center == "global":
        TEXT_MEAN = vlm.get_text_encoding(classes).float().mean(0, keepdim=True)
        print("[%s] C2a text common mode from %d classes (||mean||=%.3f), tpl=%s agg=%s"
              % (args.scene, len(classes), float(TEXT_MEAN.norm()), args.tpl, args.text_agg), flush=True)
    print("[%s] %d classes, %d labelled views: %s, mode=%s" % (
        args.scene, len(classes), len(views), views, args.mode), flush=True)

    cams = {c.image_name: c for c in scene.getTrainCameras()}
    rows, per_view = [], []

    with torch.no_grad():
        for view in views:
            cam = cams.get(view)
            if cam is None:
                print("WARN missing camera for view", view, flush=True)
                continue
            h, w = cam.image_height, cam.image_width
            gts, cands_of = {}, {}
            for cls in classes:
                mp = os.path.join(seg_root, view, cls + ".png")
                if not os.path.exists(mp):
                    print("WARN missing mask", mp, flush=True)
                    continue
                gt = read_mask(mp)
                if gt is None:
                    continue
                gts[cls] = gt
                cands_of[cls] = candidates(cls)

            if args.debug_cands:
                n_tot = int(gaussians.get_xyz.shape[0])
                print("N_GAUSSIANS %d" % n_tot, flush=True)
                for L in range(1, len(snag.labels)):
                    print("LEVEL %d: %d clusters" % (L, int(snag.labels[L].max().item()) + 1),
                          flush=True)
                for cls in classes:
                    if cls not in gts:
                        continue
                    ids, s_all = [], text_sims(cls)
                    rows_dbg = [(int(c[0]), int(c[2]), float(c[3]), int(c[4]))
                                for c in cands_of[cls]]
                    print("CAND %-16s n=%d %s" % (cls, len(rows_dbg),
                          [(L, i, round(s, 4), n) for (L, i, s, n) in rows_dbg]), flush=True)
                raise SystemExit(0)

            vals = []
            if args.mode == "oracle":
                # Ceiling test for a *region-level* readout: select EVERY Gaussian whose
                # projected centre falls inside the GT mask, rasterise, threshold.  The
                # resulting IoU is the best any per-Gaussian binary selection can do with
                # this renderer/threshold on this superpoint-free oracle set, i.e. it
                # separates "candidate search is wrong" from "the protocol is".
                xyz = gaussians.get_xyz
                n_g = xyz.shape[0]
                xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
                p_hom = xyz_h @ cam.full_proj_transform
                depth = p_hom[:, 2]
                p_xy = p_hom[:, :2] / (depth[:, None] + 1e-8)
                uu = ((p_xy[:, 0] + 1) * w - 1) / 2
                vv = ((p_xy[:, 1] + 1) * h - 1) / 2
                ui = uu.round().long()
                vi = vv.round().long()
                ok = (depth > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
                for cls in classes:
                    if cls not in gts:
                        continue
                    gt = gts[cls]
                    gth, gtw = gt.shape
                    gi = (vi.float() * gth / h).long().clamp(0, gth - 1).cpu()
                    gj = (ui.float() * gtw / w).long().clamp(0, gtw - 1).cpu()
                    inside = torch.from_numpy(gt[gi.numpy(), gj.numpy()]).to(ok.device)
                    keep = ok & inside
                    rel = torch.zeros((n_g, 1), dtype=torch.float32, device="cuda")
                    rel[keep.bool()] = 1.0
                    gaussians._semantics = rel.expand(-1, N_CHANNELS)
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    small = (sem.reshape(sem.shape[0], -1)[0] > args.pred_thresh).reshape(h, w)
                    vals.append(score(to_gt(small, gt), gt, view, cls, -1))
            elif args.mode == "pixsim":
                # Per-pixel readout on the finest features this pipeline owns.  Every Gaussian
                # inherits the CLIP feature of its level-L cluster, so rasterising gives a
                # per-pixel text-similarity map flux; the binary mask is a threshold on it.
                # This is the closest available analogue of the pixel-level readout the
                # literature reports (LangSplat/ReLaGS blend per-Gaussian CLIP features).
                L = int(args.pixsim_level)
                lab = snag.labels[L].long().cuda().clamp(min=0)
                f_l = snag.feat[L - 1]
                if args.text == "canon":
                    vlm.encode_text(cls)
                    sim_c = vlm.compute_similarity(f_l)
                elif args.text == "dot":
                    v = l2n(vlm.get_text_encoding([cls]).float())[0]
                    sim_c = f_l @ v
                else:
                    bank_words = [t.format(cls) for t in TEMPLATES]
                    bank = l2n(vlm.get_text_encoding(bank_words).float())
                    sim_c = (f_l @ bank.T).mean(dim=1) if args.text_agg == "mean" \
                        else lse_bank(f_l, bank, args.tau)
                sim_c = scale_norm([sim_c])[0]
                per_g = sim_c.reshape(-1)[lab].float().reshape(-1, 1)
                gaussians._semantics = per_g.expand(-1, N_CHANNELS).contiguous()
                sem = render(cam, gaussians, pipe, bg)["semantics"]
                m = sem.reshape(sem.shape[0], -1)[0].reshape(h, w)
                if str(args.pixsim_thr) == "otsu":
                    lo, hi = float(m.min()), float(m.max())
                    m8 = ((m - lo) / (hi - lo + 1e-8) * 255.0).clamp(0, 255).byte().cpu().numpy()
                    t_otsu, _ = cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    small = torch.from_numpy(m8 >= t_otsu).cuda()
                else:
                    small = m > float(args.pixsim_thr)
                vals.append(score(to_gt(small, gt), gt, view, cls, -1))
            elif args.mode == "gthresh":
                # GT-free, scale-free selection: z-score the similarity of every cluster at one
                # hierarchy level and keep the clusters above mean + k*std.  Unlike the shipped
                # topk=5 + gap rule this can keep MANY clusters, so a class that covers most of
                # the scene (green lawn / wood wall / grey sofa) is selected as a large region
                # instead of a handful of small high-contrast leaves.
                L = int(args.gthresh_level)
                lab = snag.labels[L].long().cuda().clamp(min=0)
                s_all = text_sims(cls)
                s = s_all[L - 1]
                s = (s - s.mean()) / s.std().clamp(min=1e-6)
                thr = s.mean() + float(args.gthresh_k) * s.std()
                keep_c = s >= thr
                sel = keep_c[lab]
                rel = torch.zeros((gaussians.get_xyz.shape[0], 1), dtype=torch.float32,
                                  device="cuda")
                rel[sel] = 1.0
                gaussians._semantics = rel.expand(-1, N_CHANNELS)
                sem = render(cam, gaussians, pipe, bg)["semantics"]
                small = (sem.reshape(sem.shape[0], -1)[0] > args.pred_thresh).reshape(h, w)
                vals.append(score(to_gt(small, gt), gt, view, cls, int(keep_c.sum().item())))
            elif args.mode == "multi":
                # Every class competes at once.  Each class mask is rasterised exactly as in
                # prompt mode (so the two readouts share one renderer path); the class map is
                # then the per-pixel argmax over the class weights, weighted by the class's
                # candidate similarity so that overlapping selections resolve to the more
                # confident class.  Pixels no selected Gaussian covers stay background.
                wmaps, wscores = [], []
                for cls in classes:
                    if cls not in gts:
                        continue
                    cands = cands_of[cls]
                    rel = snag._get_related_gaussians(cands)
                    gaussians._semantics = rel.expand(-1, N_CHANNELS).cuda()
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    wmaps.append(sem.reshape(sem.shape[0], -1)[0].reshape(h, w))
                    wscores.append(float(max([c[3] for c in cands], default=0.0)))
                if wmaps:
                    stack = torch.stack([wmaps[i] * wscores[i] for i in range(len(wmaps))])
                    best, idx = stack.max(dim=0)
                    amap = torch.where(best > 1e-4, idx,
                                       torch.full_like(idx, -1))
                    for ci, cls in enumerate(classes):
                        if cls not in gts:
                            continue
                        vals.append(score(to_gt(amap == ci, gts[cls]), gts[cls], view, cls,
                                          len(cands_of[cls])))
            else:
                for cls in classes:
                    if cls not in gts:
                        continue
                    cands = cands_of[cls]
                    rel = snag._get_related_gaussians(cands)
                    if int(rel.sum().item()) == 0:
                        print("    EMPTY-REL %-18s %d candidates %s"
                              % (cls, len(cands),
                                 [(type(c[0]).__name__, c[0], type(c[2]).__name__, c[2])
                                  for c in cands[:3]]), flush=True)
                    sem = render_rel(cam, rel)
                    if args.norm_alpha:
                        cov = render_rel(cam, torch.ones_like(rel))
                        sem = sem / cov.clamp(min=1e-3)
                    small = (sem.reshape(sem.shape[0], -1)[0] > args.pred_thresh).reshape(h, w)
                    if args.empty_fallback == "biggest" and int(small.sum()) == 0:
                        # A class that owns most of the scene can be under-covered by the
                        # similarity-gap truncation: the clusters it needs are large and their
                        # averaged feature scores slightly lower, so the prefix keeps only small
                        # high-contrast clusters that cover < 50% of no pixel -> empty mask.
                        # GT-free repair: take the LARGEST clusters that are still within
                        # --fb-margin of the best similarity and render those.
                        allc_raw = pool(cls)
                        best_sim = float(max(cc[3] for cc in allc_raw)) if allc_raw else 0.0
                        allc = [c for c in allc_raw
                                if c[2] is not None and float(c[3]) >= best_sim - args.fb_margin]
                        if allc:
                            def _npts(c):
                                try:
                                    a = np.asarray(c[4]).reshape(-1)
                                    return int(a[0]) if a.size else 0
                                except (TypeError, ValueError):
                                    return 0
                            allc = sorted(allc, key=_npts, reverse=True)[:max(1, args.fb_k)]
                            rel2 = snag._get_related_gaussians(allc)
                            sem2 = render_rel(cam, rel2)
                            small2 = (sem2.reshape(sem2.shape[0], -1)[0]
                                      > args.pred_thresh).reshape(h, w)
                            if int(small2.sum()) == 0:
                                # Still empty with the biggest plausible clusters: the selection is
                                # occluded by unselected geometry, so its blend weight never reaches
                                # the threshold.  Render the selection as its own scene for this
                                # class only (occlusion removed) instead of for every class.
                                sem2 = render_rel(cam, rel2, force_kill=True)
                                small2 = (sem2.reshape(sem2.shape[0], -1)[0]
                                          > args.pred_thresh).reshape(h, w)
                                print("    fallback-kill %-18s %d px"
                                      % (cls, int(small2.sum())), flush=True)
                            print("    fallback %-18s %d clusters -> %d px"
                                  % (cls, len(allc), int(small2.sum())), flush=True)
                            if int(small2.sum()) > 0:
                                small = small2
                    vals.append(score(to_gt(small, gts[cls]), gts[cls], view, cls, len(cands)))
            m = float(np.mean(vals)) if vals else float("nan")
            per_view.append(m)
            print("  view %s: %d classes, mIoU=%.4f" % (view, len(vals), m), flush=True)

    scene_iou = float(np.mean(per_view)) if per_view else float("nan")
    flat = float(np.mean([r["iou"] for r in rows])) if rows else float("nan")
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scene", "image", "prompt", "iou", "pred_px",
                                           "gt_px", "n_candidates"])
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    os.makedirs(args.out, exist_ok=True)
    print("SCENE_RESULT %s mIoU=%.4f flat=%.4f queries=%d views=%d mode=%s" % (
        args.scene, scene_iou, flat, len(rows), len(per_view), args.mode), flush=True)
    json.dump(dict(scene=args.scene, miou=scene_iou, flat_miou=flat, n_queries=len(rows),
                   per_view=per_view, mode=args.mode,
                   protocol=dict(topk=args.topk, level_until=args.level_until,
                                 filter_small=args.filter_small, center=args.center,
                                 text=args.text, tau=args.tau, norm=args.norm,
                                 renorm=args.renorm, text_agg=args.text_agg,
                                 text_center=args.text_center, tpl=args.tpl,
                                 cand_mode=args.cand_mode, cand_margin=args.cand_margin)),
              open(os.path.join(args.out, "%s_%s_summary.json" % (args.scene, args.tag)), "w"),
              indent=2)


if __name__ == "__main__":
    main()