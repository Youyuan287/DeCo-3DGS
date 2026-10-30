"""Mask-feature diagnosis: why can't mask-level CLIP features rank the GT-best mask?

For every query (query view):
  raw sim        : cos(mask CLIP feature, text)
  centered sim   : cos(mask feature - scene-mean feature, text)
  rank_raw       : rank of the GT-best mask under raw sim
  rank_cen       : rank under centered sim
  iou_best       : IoU of the GT-best mask with GT (oracle)
  iou_top1_raw / iou_top1_cen : IoU of the top-ranked mask
  margin_raw     : sim(best) - sim(top1)  (<=0 when selection fails)
  sim_std        : std of sims across masks (noise scale)
  cos_scene      : mean cos(mask feature, scene-mean feature) (scene-sharedness)
GT is used only for evaluation/best-mask identification.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure
from eval_lerf_ovs_v2 import polygon_to_mask

ap = ArgumentParser()
for k in ["scene", "source", "model", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False, vlm_type="clip")
vlm = ClipSimMeasure()
vlm.load_model()

gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# scene-mean feature over all masks (all views, foreground masks)
acc = None
cnt = 0
with torch.no_grad():
    for c in cams.values():
        sem = c.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        seg = c.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = c.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        labels = np.unique(seg[fg])
        if labels.size == 0:
            continue
        smin = int(labels.min())
        rows = [L - smin for L in labels if 0 <= L - smin < sem.shape[0]]
        if not rows:
            continue
        f = unit(sem[rows])
        acc = f.sum(0) if acc is None else acc + f.sum(0)
        cnt += len(rows)
scene_mean = acc / max(cnt, 1)
scene_mean = scene_mean / max(np.linalg.norm(scene_mean), 1e-8)

t_cache = {}
rows_out = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        seg = cam.semantic["seg_map"][3].cpu().numpy().astype(np.int64)
        fg = cam.semantic["fg_mask"][3].cpu().numpy().astype(bool)
        sem = cam.semantic["sem"].detach().cpu().numpy().astype(np.float32)
        labels = np.unique(seg[fg])
        if labels.size == 0:
            continue
        smin = int(labels.min())
        prompt_list = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        mask_arrs = {int(L): ((seg == L) & fg) for L in labels}
        for prompt in prompt_list:
            nq += 1
            if args.limit_queries and nq > args.limit_queries:
                break
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            ious = {}
            for L, m in mask_arrs.items():
                u = np.logical_or(m, gt).sum()
                ious[L] = float(np.logical_and(m, gt).sum() / u) if u else 0.0
            best_L = max(ious, key=ious.get)
            if prompt not in t_cache:
                vlm.encode_text(prompt)
                t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float32)
                t_cache[prompt] = t / max(np.linalg.norm(t), 1e-8)
            t = t_cache[prompt]
            sims_raw, sims_cen, cos_scene = {}, {}, {}
            for L in mask_arrs:
                idx = L  # features are indexed by absolute label id
                if not (0 <= idx < sem.shape[0]):
                    continue
                f = unit(sem[idx])
                sims_raw[L] = float(np.dot(f, t))
                sims_cen[L] = float(np.dot(unit(f - scene_mean), t))
                cos_scene[L] = float(np.dot(f, scene_mean))
            if best_L not in sims_raw:
                continue
            order_raw = sorted(sims_raw, key=lambda L: -sims_raw[L])
            order_cen = sorted(sims_cen, key=lambda L: -sims_cen[L])
            top1_raw, top1_cen = order_raw[0], order_cen[0]
            vals = np.array(list(sims_raw.values()))
            sv = np.sort(vals)[::-1]
            sim_second = float(sv[1]) if len(sv) > 1 else float("nan")
            margin_best_second = float((sims_raw[best_L] - sim_second) / max(vals.std(), 1e-9))
            rows_out.append(dict(
                scene=args.scene, image=name, prompt=prompt, n_masks=len(sims_raw),
                rank_raw=order_raw.index(best_L) + 1, rank_cen=order_cen.index(best_L) + 1,
                iou_best=ious[best_L], iou_top1_raw=ious[top1_raw], iou_top1_cen=ious[top1_cen],
                gt_area=int(gt.sum()), top1_area=int(mask_arrs[top1_raw].sum()),
                top1_recall=float(np.logical_and(mask_arrs[top1_raw], gt).sum() / max(int(gt.sum()), 1)),
                top1_precision=float(np.logical_and(mask_arrs[top1_raw], gt).sum() / max(int(mask_arrs[top1_raw].sum()), 1)),
                margin_raw=sims_raw[best_L] - sims_raw[top1_raw],
                margin_cen=sims_cen[best_L] - sims_cen[top1_cen],
                sim_std=float(vals.std()),
                sim_best=float(sims_raw[best_L]),
                sim_second=sim_second,
                margin_best_second=margin_best_second,
                cos_scene_mean=float(np.mean(list(cos_scene.values()))),
                cos_scene_best=float(cos_scene[best_L])))
        print("IMG", name, len(prompt_list), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_maskdiag.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    wr.writeheader()
    wr.writerows(rows_out)
print("SCENE_DONE", args.scene, len(rows_out))