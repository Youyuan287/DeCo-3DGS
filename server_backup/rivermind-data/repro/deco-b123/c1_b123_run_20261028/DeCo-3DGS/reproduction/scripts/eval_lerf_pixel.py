#!/usr/bin/env python
# ReLaGS LERF-OVS reproduction — pixel-wise relevance evaluator (ALTERNATIVE protocol)
# 用途: 按论文 Sec.15 字面实现"逐像素 CLIP 相似度 + 阈值", 作为并列口径/消融。
# 协议: 渲染 512 维语言场(光栅化器语义通道编译期固定 20, 故分 26 块渲染再拼接),
#       逐像素 L2 归一化, relevance = min_i softmax(10*[s_q, s_canon_i])[pos]
#       (与仓库 utils/vlm_utils.ClipSimMeasure 及 COS3D 的 relevancy 公式一致),
#       阈值 tau=0.5, 并用 all-ones 语义渲染得到 alpha 覆盖率掩膜排除背景。
# 结论: 该口径在本方法上系统性过选 (mean pred/gt 达 8.75~15.65), 不作主指标; 详见 reports/07。
"""LERF-OVS evaluation with PIXEL-WISE relevance thresholding.

Protocol (ReLaGS paper Sec. 15 + COS3D Eq. for relevance + LERF canonical phrases):
  render the language feature field (512-d per pixel) -> per-pixel CLIP relevance
  R = min_i softmax_2class(10*[s_query, s_canon_i])[pos]  (repo's ClipSimMeasure)
  -> binary prediction at tau (0.5 default) -> IoU against GT polygons.

This differs from eval_lerf_relags.py, which renders a binary cluster mask
(THGS test_lerf.py convention). Both are provided; this one follows the paper text.
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

SEM_CH = 20  # compile-time semantic channel count of the surfel rasterizer


def polygon_to_mask(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--levels", default="max",
                    help="'max' = per-pixel max relevance over hierarchy levels, or e.g. '3'")
    ap.add_argument("--save_masks", action="store_true")
    args = ap.parse_args()

    p = argparse.ArgumentParser()
    lp = ModelParams(p); pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(["-s", args.source, "-m", args.model])
    dataset = lp.extract(pargs); pipe = pp.extract(pargs)

    gaussians = GaussianModel(dataset.sh_degree, 512)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    labels = [l.cuda().long() for l in snag.labels]
    # nag_feat[i] carries the features of hierarchy level i+1 (level 0 = superpoints, no feature)
    pairs = [(i + 1, i) for i in range(len(snag.feat))]
    print("levels:", [(li, tuple(snag.feat[fi].shape)) for li, fi in pairs], flush=True)
    if args.levels == "max":
        use_levels = pairs
    else:
        want = {int(x) for x in args.levels.split(",")}
        use_levels = [p for p in pairs if p[0] in want]

    vlm = ClipSimMeasure(); vlm.load_model()

    gt_dir = os.path.join(args.gt_root, args.scene)
    imgs = sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg"))
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    rows, per_img, rel_stats = [], [], []
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
                print("WARN missing camera", name); continue
            h, w = cam.image_height, cam.image_width
            # render the 512-d language field per level, chunked to the 20-channel kernel
            cache = {}
            for li, fi in use_levels:
                feats = snag.feat[fi]
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                idx = labels[li]
                FH = torch.empty((h * w, feats.shape[1]), dtype=torch.float16, device="cuda")
                for k in range(0, feats.shape[1], SEM_CH):
                    cols = torch.arange(k, min(k + SEM_CH, feats.shape[1]), device="cuda")
                    lg = feats[:, cols][idx]
                    pad = torch.zeros((lg.shape[0], SEM_CH), device="cuda", dtype=lg.dtype)
                    pad[:, :lg.shape[1]] = lg
                    gaussians._semantics = pad.contiguous()
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    FH[:, cols] = sem.reshape(SEM_CH, -1)[:lg.shape[1]].T.half()
                cache[li] = FH
            # alpha coverage: semantics are alpha-weighted sums, so rendering all-ones
            # semantics gives the accumulated opacity per pixel (0 outside the geometry).
            gaussians._semantics = torch.ones((labels[0].shape[0], SEM_CH), device="cuda")
            cov = render(cam, gaussians, pipe, bg)["semantics"][0].reshape(-1)
            vals = []
            for prompt in prompts:
                vlm.encode_text(prompt)
                T = vlm.text_feature                       # (1+Ncanon, 512) normalised
                rel = None
                for li, _ in use_levels:
                    F = cache[li].float()
                    F = F / F.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    logits = F @ T.T                       # (HW, 1+Ncanon)
                    s_pos = logits[:, 0:1]
                    s_can = logits[:, 1:]
                    sims = torch.stack((s_pos.expand_as(s_can), s_can), dim=-1)
                    prob = torch.softmax(10 * sims, dim=-1)[..., 0]
                    r, _ = prob.min(dim=-1)                # min over canonical phrases (COS3D/LERF)
                    rel = r if rel is None else torch.maximum(rel, r)
                keep = (rel > args.tau) & (cov > 0.5)
                pred = keep.reshape(h, w).cpu().numpy().astype(np.uint8)
                gt = np.zeros((h, w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                inter = float(np.logical_and(pred, gt).sum())
                union = float(np.logical_or(pred, gt).sum())
                iou = inter / union if union > 0 else float("nan")
                vals.append(iou)
                rel_stats.append(float(rel.mean().item()))
                rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                                 iou=round(iou, 6), pred_px=int(pred.sum()),
                                 gt_px=int(gt.sum()), n_candidates=len(use_levels)))
                if args.save_masks:
                    d = os.path.join(args.out, "masks_pixel", args.scene, name)
                    os.makedirs(d, exist_ok=True)
                    safe = prompt.replace(" ", "_").replace("/", "_")
                    cv2.imwrite(os.path.join(d, safe + ".png"), pred * 255)
                    cv2.imwrite(os.path.join(d, safe + "_gt.png"), gt * 255)
            m = float(np.mean(vals)) if vals else float("nan")
            per_img.append(m)
            print(f"  {name}: {len(prompts)} queries, mIoU={m:.4f}", flush=True)

    scene_iou = float(np.mean(per_img)) if per_img else float("nan")
    flat = float(np.mean([r["iou"] for r in rows])) if rows else float("nan")
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["scene", "image", "prompt", "iou",
                                           "pred_px", "gt_px", "n_candidates"])
        if f.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    print(f"SCENE_RESULT {args.scene} mIoU={scene_iou:.4f} flat={flat:.4f} "
          f"queries={len(rows)} images={len(per_img)}")
    json.dump(dict(scene=args.scene, miou=scene_iou, flat_miou=flat,
                   n_queries=len(rows), per_image=per_img,
                   mean_relevance=float(np.mean(rel_stats)) if rel_stats else None,
                   protocol=dict(mode="pixel_relevance", tau=args.tau, levels=args.levels)),
              open(os.path.join(args.out, f"{args.scene}_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
