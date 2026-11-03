"""Evaluate retrieval at the primitive (level-0) superpoint level on 3D-OVS.

Usage: python eval_3dovs_l0.py --scene bed --source ... --model ... --nag ...
Selects level-0 clusters by text similarity (z-score threshold or top-N),
renders the union, and scores against the GT masks with the stock protocol.
"""
import os, sys, json, argparse
import numpy as np
import torch
import cv2

_REL = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
for p in (_REL, os.path.join(_REL, "ext"), "/root/rivermind-data/deco/DeCo-3DGS", "/root/rivermind-data/deco/DeCo-3DGS/deco"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from nag_data import SemanticNAG
from utils.vlm_utils import ClipSimMeasure

N_CHANNELS = 20


def read_mask(path):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if gt is None else (gt > 0).astype(np.uint8)


def iou(pred, gt):
    u = float(np.logical_or(pred, gt).sum())
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def score_triplet(pred, gt):
    p = cv2.resize((pred.astype(np.uint8) * 255), (gt.shape[1], gt.shape[0]),
                   interpolation=cv2.INTER_NEAREST) > 0
    return iou(p, gt > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model, "-r", "4"])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, N_CHANNELS)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    labels0 = snag.labels[0].long().cuda()
    feat0 = snag.feat[0]
    n_clusters = int(labels0.max().item()) + 1

    seg_root = os.path.join(args.gt_root, args.scene, "segmentations")
    classes = [c.strip() for c in open(os.path.join(seg_root, "classes.txt")) if c.strip()]
    views = sorted(d for d in os.listdir(seg_root) if os.path.isdir(os.path.join(seg_root, d)))
    cams = {c.image_name: c for c in scene.getTrainCameras()}

    variants = [("z%.0f" % k, ("z", k)) for k in (0, 1, 2, 3)] + \
               [("top%d" % n, ("top", n)) for n in (25, 50, 100, 200)]
    rows = {v[0]: [] for v in variants}

    with torch.no_grad():
        for view in views:
            cam = cams.get(view)
            if cam is None:
                continue
            h, w = cam.image_height, cam.image_width
            for cls in classes:
                mp = os.path.join(seg_root, view, cls + ".png")
                if not os.path.exists(mp):
                    continue
                gt = read_mask(mp)
                vlm.encode_text(cls)
                sim = vlm.compute_similarity(feat0)
                sim = sim.detach().float()
                base = float(sim.mean().item()); sd = float(sim.std().item()) or 1.0
                for tag, (mode, val) in variants:
                    if mode == "z":
                        keep = sim >= (base + float(val) * sd)
                    else:
                        k = min(int(val), sim.numel())
                        thr = torch.topk(sim, k).values.min()
                        keep = sim >= thr
                    rel = keep[labels0].float().reshape(-1, 1)
                    gaussians._semantics = rel.expand(-1, N_CHANNELS).cuda()
                    sem = render(cam, gaussians, pipe, bg)["semantics"]
                    pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()
                    rows[tag].append(dict(image=view, prompt=cls,
                                          iou=score_triplet(pred, gt),
                                          n_sel=int(keep.sum().item())))
            print("view", view, "done", flush=True)

    out = {"scene": args.scene, "variants": {}}
    for tag, rs in rows.items():
        rs = [r for r in rs if r["iou"] == r["iou"]]
        by = {}
        for r in rs:
            by.setdefault(r["image"], []).append(r["iou"])
        m = float(np.mean([np.mean(v) for v in by.values()]))
        out["variants"][tag] = dict(mIoU=m, n=len(rs),
                                    mean_sel=float(np.mean([r["n_sel"] for r in rs])))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "l0_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("RESULT", args.scene, json.dumps({k: round(v["mIoU"], 4) for k, v in out["variants"].items()}), flush=True)


if __name__ == "__main__":
    main()