#!/usr/bin/env python
# ReLaGS LERF-OVS reproduction — cluster-based evaluator (MAIN protocol)
# 用途: 用作者发布的语义层级 (sai_nag.pt) 对 LERF-OVS 的 22 张标注图 / 208 条 query 算 mIoU。
# 协议: 检索 search_matched_superpoint_in_mhtree(topk=5, level_until=3, filter_small=50)
#       (前三者是 ReLaGS 该函数的默认值, level_until!=-1 对应论文 Alg.1 的层级下降),
#       渲染语义掩膜 (semantics[0] > 0.5), 与 GT 多边形算 IoU; 先逐图平均再对图平均。
# 用法: cd ReLaGS && PYTHONPATH=$PWD/ext:$PWD python ../reproduction/scripts/eval_lerf_relags.py \
#         --scene figurines --source <src> --model <hf_scene> --nag <hf_scene>/sai_nag.pt \
#         --gt_root <lerf_ovs/label> --out <out> --csv <out.csv> --topk 5 --level_until 3 --filter_small 50
# 说明: 上游仓库未发布 mIoU 代码, 本脚本为复现实验原创, 与 THGS 的 test_lerf.py 同构。
"""LERF-OVS evaluation harness for ReLaGS.

Provenance:
  * semantic field / hierarchy / retrieval  : ReLaGS official code (nag_data.py, merge_proj output)
  * rendering of binary query mask          : same convention as THGS test_lerf.py
  * GT masks + query set + IoU aggregation  : THGS scripts/eval_seg.py (LERF-OVS labels)
No GT-driven tuning of level/threshold is performed.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--level_until", type=int, default=3)
    ap.add_argument("--filter_small", type=int, default=50)
    ap.add_argument("--no_remove_small", action="store_true")
    ap.add_argument("--save_masks", action="store_true")
    args = ap.parse_args()

    from argparse import ArgumentParser
    p = ArgumentParser()
    lp = ModelParams(p)
    pp = PipelineParams(p)
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(["-s", args.source, "-m", args.model])
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

    gt_dir = os.path.join(args.gt_root, args.scene)
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
            for prompt in prompts:
                vlm.encode_text(prompt)
                sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
                rel, cands = snag.search_matched_superpoint_in_mhtree(
                    sims, topk=args.topk, level_until=args.level_until,
                    filter_small=args.filter_small,
                    remove_small=not args.no_remove_small)
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
                rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                                 iou=round(iou, 6), pred_px=int(pred.sum()),
                                 gt_px=int(gt.sum()), n_candidates=len(cands)))
                if args.save_masks:
                    d = os.path.join(args.out, "masks", args.scene, name)
                    os.makedirs(d, exist_ok=True)
                    safe = prompt.replace(" ", "_").replace("/", "_")
                    cv2.imwrite(os.path.join(d, safe + ".png"), pred * 255)
                    cv2.imwrite(os.path.join(d, safe + "_gt.png"), gt * 255)
            m = float(np.mean(vals)) if vals else float("nan")
            per_img.append(m)
            print(f"  {name}: {len(prompts)} queries, mIoU={m:.4f}")

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
                   protocol=dict(topk=args.topk, level_until=args.level_until,
                                 filter_small=args.filter_small,
                                 remove_small=not args.no_remove_small)),
              open(os.path.join(args.out, f"{args.scene}_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
