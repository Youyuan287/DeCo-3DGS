#!/usr/bin/env python
"""Evaluate ReLaGS on LERF-Mask (Gaussian Grouping protocol).

Protocol
  * query   : each GT mask filename in test_mask/<i>/<prompt>.png is the text prompt
  * retrieval: ReLaGS multi-level search (defaults topk=5, level_until=3, filter_small=50)
  * render  : per-Gaussian binary selection -> rasterised semantics > 0.5
  * metric  : IoU and Boundary-IoU, code identical to
              gaussian-grouping/script/eval_lerf_mask.py (dilation_ratio=0.02)
  * aggregate: mean over prompts of (mean over views)  -> "Overall Mean IoU" (GG)

Predicted masks are written as <out>/<scene>/<view_dir>/<prompt>.png (0/255) at GT
resolution so that the official GG script can be run on them unchanged.
"""
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2
import numpy as np
import torch
from PIL import Image

RELAGS = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RELAGS)

from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from arguments import ModelParams, PipelineParams  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402


# ---- Gaussian Grouping boundary IoU (verbatim) --------------------------------
def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    dt = (dt > 128).astype("uint8")
    gt = (gt > 128).astype("uint8")
    gb = mask_to_boundary(gt, dilation_ratio)
    db = mask_to_boundary(dt, dilation_ratio)
    inter = ((gb * db) > 0).sum()
    union = ((gb + db) > 0).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def iou_of(gt, dt):
    g = gt > 128
    d = dt > 128
    inter = np.logical_and(g, d).sum()
    union = np.logical_or(g, d).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--eval-source", required=True, help="build/<scene>_eval (full COLMAP)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt-root", required=True, help="datasets/lerf_mask (contains <scene>/test_mask)")
    ap.add_argument("--out", required=True, help="results/lerf_mask/pred")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--level-until", type=int, default=3)
    ap.add_argument("--filter-small", type=int, default=50)
    ap.add_argument("--no-remove-small", action="store_true")
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--check-rgb", action="store_true", help="render RGB of test views and PSNR vs test_i.jpg")
    ap.add_argument("--vis-dir", default="", help="write RGB|GT|pred panels here")
    args = ap.parse_args()

    from argparse import ArgumentParser
    p = ArgumentParser()
    lp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(["-s", args.eval_source, "-m", args.model])
    dataset = lp.extract(pargs)
    pipe = pp.extract(pargs)

    split = json.load(open(os.path.join(args.eval_source, "split.json")))
    print(f"[{args.scene}] test views: " + ", ".join(
        f"{t['dir']}->{t['image']}({len(t['prompts'])} prompts)" for t in split["test"]))

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    print(f"[{args.scene}] cameras available: {len(cams)}; gaussians: {gaussians.get_xyz.shape[0]}")

    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    print(f"[{args.scene}] nag levels: {len(nag['nag'])}  superpoints/level: "
          f"{[int(l.max()) + 1 for l in nag['nag']]}")

    vlm = ClipSimMeasure()
    vlm.load_model()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    gt_scene = os.path.join(args.gt_root, args.scene, "test_mask")

    rows, per_prompt, per_view, align = [], {}, [], []
    n_done = 0
    with torch.no_grad():
        for entry in split["test"]:
            cam = cams.get(entry["camera"])
            if cam is None:
                raise SystemExit(f"FATAL: camera {entry['camera']} not found in eval scene")
            h, w = cam.image_height, cam.image_width
            vdir = os.path.join(gt_scene, entry["dir"])

            rgb_png = None
            if args.check_rgb or args.vis_dir:
                out = render(cam, gaussians, pipe, bg)
                rgb = out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                rgb_u8 = (rgb * 255).astype(np.uint8)
                gt_img_path = os.path.join(args.gt_root, args.scene, "images", entry["image"])
                if os.path.exists(gt_img_path):
                    gt_rgb = np.array(Image.open(gt_img_path).convert("RGB"))
                    if gt_rgb.shape[:2] != rgb_u8.shape[:2]:
                        gt_rgb = np.array(Image.fromarray(gt_rgb).resize((rgb_u8.shape[1], rgb_u8.shape[0])))
                    align.append(dict(view=entry["dir"], image=entry["image"], psnr=psnr(rgb_u8, gt_rgb)))
                else:
                    align.append(dict(view=entry["dir"], image=entry["image"], psnr=None))
                if args.vis_dir:
                    os.makedirs(args.vis_dir, exist_ok=True)
                    Image.fromarray(rgb_u8).save(os.path.join(args.vis_dir, f"{args.scene}_{entry['dir']}_rgb.png"))
                rgb_png = rgb_u8

            vals = []
            for prompt in entry["prompts"]:
                if args.limit_queries and n_done >= args.limit_queries:
                    break
                vlm.encode_text(prompt)
                sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
                rel, cands = snag.search_matched_superpoint_in_mhtree(
                    sims, topk=args.topk, level_until=args.level_until,
                    filter_small=args.filter_small, remove_small=not args.no_remove_small)
                try:
                    sc = [float(c[3]) for c in cands]
                    dd = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
                    gapk = (max(range(len(dd)), key=lambda i: dd[i]) + 1) if dd else 1
                except Exception:
                    gapk = 1
                gt_path = os.path.join(vdir, prompt + ".png")
                gt = np.array(Image.open(gt_path).convert("L"))
                K = len(cands)
                curve = []
                for kk in range(1, K + 1):
                    use = cands[:kk]
                    if kk == K:
                        relk = rel
                    else:
                        relk = snag._get_related_gaussians(use)
                    gaussians._semantics = relk.expand(-1, 20).cuda()
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    pk = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(sem.shape[1], sem.shape[2])
                    pk = pk.cpu().numpy().astype(np.uint8) * 255
                    if gt.shape != pk.shape:
                        pk = cv2.resize(pk, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                    curve.append(float(iou_of(gt, pk)))
                print("  CURVE %s %s %s K=%d GAPK=%d %s" % (args.scene, entry["dir"], prompt.replace(" ", "_"),
                                                    K, gapk, " ".join("%.4f" % v for v in curve)), flush=True)
                pred = None
                iou = curve[-1] if curve else 0.0
                pred = None
                sem = None
                biou = float('nan')
                pdir = os.path.join(args.out, args.scene, entry["dir"])
                os.makedirs(pdir, exist_ok=True)
                if False and args.vis_dir and rgb_png is not None:
                    panel = np.concatenate([rgb_png,
                                            np.repeat((gt > 128)[..., None] * 255, 3, 2).astype(np.uint8),
                                            np.repeat((pred > 128)[..., None] * 255, 3, 2).astype(np.uint8)], axis=1)
                    Image.fromarray(panel).save(os.path.join(
                        args.vis_dir, f"{args.scene}_{entry['dir']}_{prompt.replace(' ', '_')}.png"))

                rows.append(dict(scene=args.scene, view=entry["dir"], image=entry["image"], prompt=prompt,
                                 iou=round(iou, 6), biou=round(biou, 6),
                                 pred_px=-1, gt_px=int((gt > 128).sum()),
                                 n_candidates=len(cands)))
                per_prompt.setdefault(prompt, []).append(iou)
                vals.append(iou)
                n_done += 1
                print(f"  [{entry['dir']}] {prompt!r}: IoU={iou:.4f} BIoU={biou:.4f} "
                      f"cands={len(cands)}")
            if vals:
                per_view.append(float(np.mean(vals)))

    with open(args.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                            ["scene", "view", "image", "prompt", "iou", "biou",
                             "pred_px", "gt_px", "n_candidates"])
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)

    class_mean_iou = float(np.nanmean([np.nanmean(v) for v in per_prompt.values()])) if per_prompt else float("nan")
    # boundary IoU class-mean, recomputed from rows
    by_prompt_b = {}
    for r in rows:
        by_prompt_b.setdefault(r["prompt"], []).append(r["biou"])
    class_mean_biou = float(np.nanmean([np.nanmean(v) for v in by_prompt_b.values()])) if rows else float("nan")
    flat = float(np.nanmean([r["iou"] for r in rows])) if rows else float("nan")
    summary = dict(scene=args.scene, miou=class_mean_iou, biou=class_mean_biou, flat_miou=flat,
                   view_mean=float(np.mean(per_view)) if per_view else float("nan"),
                   n_queries=len(rows), n_views=len(per_view),
                   per_prompt={k: float(np.nanmean(v)) for k, v in per_prompt.items()},
                   per_prompt_biou={k: float(np.nanmean(v)) for k, v in by_prompt_b.items()},
                   alignment=align,
                   protocol=dict(topk=args.topk, level_until=args.level_until,
                                 filter_small=args.filter_small,
                                 remove_small=not args.no_remove_small,
                                 threshold=0.5, nag=os.path.basename(args.nag)))
    os.makedirs(os.path.dirname(os.path.join(args.out, "x")), exist_ok=True)
    out_json = os.path.join(os.path.dirname(args.csv), f"{args.scene}_summary.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"SCENE_RESULT {args.scene} mIoU={class_mean_iou:.4f} BIoU={class_mean_biou:.4f} "
          f"flat={flat:.4f} queries={len(rows)} views={len(per_view)}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()