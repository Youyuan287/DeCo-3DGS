#!/usr/bin/env python3
"""ScanNet 3D semantic segmentation evaluation for ReLaGS.

Protocol (paper App. sec. 9 / Table 9):
  * every Gaussian gets the GT label of its **nearest GT point** (L2);
  * prediction per Gaussian comes from the hierarchical language field:
    cluster feature (finest level of sai_nag.pt) x class-name text embedding,
    argmax over the class subset -> cluster class -> every Gaussian of that
    cluster;
  * metrics: 3D mIoU (point-set IoU) and mAcc on the 19/15/10 class subsets.

Ground truth: ScanNet `<scene>_vh_clean_2.labels.ply` (NYU40 label per vertex).

Usage:
  python eval_scannet_3d.py --scene scene0000_00 \
      --data data/scannet/scene0000_00 \
      --model output/scannet/scene0000_00 \
      --gt data/scannet_raw/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply \
      --iteration 0 --out results/scannet/scene0000_00.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

NYU40 = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop",
}
SUBSETS = {
    19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    15: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    10: [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}


def read_ply(path, want=("x", "y", "z")):
    ply = PlyData.read(path)
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    return {n: np.asarray(v[n]) for n in want if n in names}, names, v


def load_gt(path):
    d, names, v = read_ply(path)
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)
    lab = np.asarray(v["label"]) if "label" in names else None
    if lab is None:
        raise SystemExit("no 'label' property in %s" % path)
    return xyz, lab.astype(np.int64)


def load_gaussians(model, iteration, opacity_min=0.0):
    p = os.path.join(model, "point_cloud", "iteration_%d" % iteration, "point_cloud.ply")
    ply = PlyData.read(p)
    v = ply["vertex"]
    names = [pr.name for pr in v.properties]
    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float64)
    op = None
    if "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    keep = np.ones(xyz.shape[0], dtype=bool) if opacity_min <= 0 else (op >= opacity_min)
    return xyz, op, keep, p


def _find_nag(model, prefer=None):
    import glob
    if prefer and os.path.exists(prefer):
        return prefer
    cands = sorted(glob.glob(os.path.join(model, "sai_nag*.pt")))
    if not cands:
        raise SystemExit("no sai_nag*.pt in %s" % model)
    for c in cands:
        if os.path.basename(c) == "sai_nag.pt":
            return c
    return cands[0]


def load_cluster_labels(model, iteration, n_gauss, nag_file=None):
    """Per-Gaussian 0-based cluster ids + matching per-cluster feature level.

    merge_proj saves `saifine_labels.npy` with 1-based final cluster ids
    (1..C); sai_nag.pt holds `nag` (C_level ids, 0-based) and `nag_feat`
    (one 512-d feature per cluster, 3 levels: 4698/937/366 style).
    """
    nag_path = _find_nag(model, nag_file)
    d = torch.load(nag_path, map_location="cpu")
    nag, feats = d["nag"], d["nag_feat"]
    counts = [int(torch.unique(x).numel()) for x in nag]
    print("nag:", os.path.basename(nag_path), "level uniques:", counts,
          "feat shapes:", [tuple(f.shape) for f in feats])

    lab = None
    src = None
    p = os.path.join(model, "saifine_labels.npy")
    if os.path.exists(p):
        lab = np.load(p).astype(np.int64)
        if lab.shape[0] != n_gauss:
            lab = None
        else:
            if lab.min() >= 1:          # merge_proj numbering starts at 1
                lab = lab - 1
            src = "saifine_labels.npy"
    if lab is None:
        idx = int(np.argmin([abs(c - feats[-1].shape[0]) for c in counts]))
        lab = nag[idx].cpu().numpy().astype(np.int64)
        src = "%s[nag][%d]" % (os.path.basename(nag_path), idx)

    uniq = int(lab.max()) + 1
    # pick the feature level whose cluster count matches the label space
    lvl = int(np.argmin([abs(f.shape[0] - uniq) for f in feats]))
    if feats[lvl].shape[0] != uniq:
        print("WARNING: label space %d vs feature rows %d (level %d)"
              % (uniq, feats[lvl].shape[0], lvl))
    return lab, feats[lvl].float(), src, lvl


def text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", prompts=("{}",)):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model.eval().cuda()
    emb = []
    with torch.no_grad():
        for n in names:
            texts = [p.format(n) for p in prompts]
            t = tok(texts).cuda()
            f = model.encode_text(t).float()
            f = f / f.norm(dim=-1, keepdim=True)
            emb.append(f.mean(0))
    emb = torch.stack(emb)
    return torch.nn.functional.normalize(emb, dim=-1)


def metrics(gt, pred, classes):
    """gt/pred use original NYU40 ids; classes = subset of NYU40 ids."""
    k = len(classes)
    remap = {c: i + 1 for i, c in enumerate(classes)}  # 1..k, 0 = ignore/other
    g = np.zeros_like(gt)
    for c, i in remap.items():
        g[gt == c] = i
    p = np.zeros_like(pred)
    for c, i in remap.items():
        p[pred == c] = i
    p[g == 0] = 0
    ious, accs = {}, {}
    inter = np.zeros(k + 1, dtype=np.int64)
    union = np.zeros(k + 1, dtype=np.int64)
    corr = np.zeros(k + 1, dtype=np.int64)
    tot = np.zeros(k + 1, dtype=np.int64)
    for i in range(1, k + 1):
        inter[i] = np.sum((g == i) & (p == i))
        union[i] = np.sum((g == i) | (p == i))
        corr[i] = inter[i]
        tot[i] = np.sum(g == i)
    present = [i for i in range(1, k + 1) if tot[i] > 0]
    for i in present:
        ious[i] = float(inter[i]) / float(union[i]) if union[i] > 0 else 0.0
        accs[i] = float(corr[i]) / float(tot[i])
    mIoU = float(np.mean([ious[i] for i in present])) if present else float("nan")
    mAcc = float(np.mean([accs[i] for i in present])) if present else float("nan")
    valid = g != 0
    oa = float(np.sum((g == p) & valid)) / float(np.sum(valid)) if valid.any() else float("nan")
    return dict(num_classes=k, present=len(present), mIoU=mIoU, mAcc=mAcc, OA=oa,
                per_class={NYU40.get(c, str(c)): dict(iou=ious[remap[c]], acc=accs[remap[c]],
                                                     gt=int(tot[remap[c]]),
                                                     inter=int(inter[remap[c]]),
                                                     union=int(union[remap[c]]))
                           for c in classes if remap[c] in ious})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opacity_min", type=float, default=0.0)
    ap.add_argument("--prompt", default="{}|a photo of a {}|the {}|a {} in a room",
                    help="text prompt templates separated by | ; they are averaged (prompt ensembling)")
    ap.add_argument("--level", type=int, default=0, help="nag feature level (0=finest)")
    ap.add_argument("--clip_type", default="ViT-B-16")
    ap.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    ap.add_argument("--nag_file", default=None)
    ap.add_argument("--center", default="none",
                    choices=["none", "image", "text", "both", "csls"])
    ap.add_argument("--csls_k", type=int, default=10)
    args = ap.parse_args()

    gt_xyz, gt_lab = load_gt(args.gt)
    print("GT points:", gt_xyz.shape[0], "labels>0:", int((gt_lab > 0).sum()))

    g_xyz, op, keep, ply_path = load_gaussians(args.model, args.iteration)
    print("Gaussians:", g_xyz.shape[0], "kept:", int(keep.sum()), "ply:", ply_path)

    lab, feats, lab_src, lvl = load_cluster_labels(args.model, args.iteration, g_xyz.shape[0], args.nag_file)
    print("cluster label source:", lab_src, "level:", lvl,
          "clusters:", int(np.unique(lab).size))

    feats = torch.nn.functional.normalize(feats, dim=-1)
    if args.center in ("image", "both"):
        feats = torch.nn.functional.normalize(feats - feats.mean(0, keepdim=True), dim=-1)
    print("cluster features:", tuple(feats.shape), "center:", args.center)

    out = dict(scene=args.scene, iteration=args.iteration, ply=ply_path,
               label_source=lab_src, n_gauss=int(g_xyz.shape[0]),
               n_kept=int(keep.sum()), n_clusters=int(np.unique(lab).size),
               n_gt=int(gt_xyz.shape[0]), prompt=args.prompt, level=int(lvl),
               clip_type=args.clip_type, clip_pretrained=args.clip_pretrained,
               subsets={})

    # nearest GT point (L2) for every Gaussian -> GT label
    tree = cKDTree(gt_xyz)
    dist, idx = tree.query(g_xyz, k=1, workers=-1)
    gauss_gt = gt_lab[idx]
    out["nearest_dist"] = dict(mean=float(dist.mean()), median=float(np.median(dist)),
                               p95=float(np.percentile(dist, 95)))

    # cluster -> class per subset
    for k, classes in SUBSETS.items():
        names = [NYU40[c] for c in classes]
        tf = text_features(names, model_name=args.clip_type, pretrained=args.clip_pretrained,
                           prompts=tuple(args.prompt.split("|"))).cpu()   # (K,D)
        tf = torch.nn.functional.normalize(tf, dim=-1)
        if args.center in ("text", "both"):
            tf = torch.nn.functional.normalize(tf - tf.mean(0, keepdim=True), dim=-1)
        sim = feats @ tf.T                                              # (C,K)
        if args.center == "csls":
            kk = min(args.csls_k, sim.shape[0], sim.shape[1])
            r_cls = torch.topk(sim, kk, dim=0).values.mean(0)
            r_clu = torch.topk(sim, kk, dim=1).values.mean(1)
            sim = sim - r_cls[None, :] - r_clu[:, None]
        cls_idx = sim.argmax(dim=1).numpy()                             # (C,)
        cluster_class = np.array([classes[i] for i in cls_idx], dtype=np.int64)
        n_clusters = max(int(lab.max()) + 1, int(cluster_class.shape[0]))
        lut = np.zeros(n_clusters + 1, dtype=np.int64)
        lut[:cluster_class.shape[0]] = cluster_class
        pred = lut[np.clip(lab, 0, n_clusters)]
        m = metrics(gauss_gt[keep], pred[keep], classes)
        m["n_gt_points"] = int(((gauss_gt[keep] > 0)).sum())
        out["subsets"][str(k)] = m
        print("K=%2d mIoU %.4f mAcc %.4f OA %.4f (present %d)"
              % (k, m["mIoU"], m["mAcc"], m["OA"], m["present"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()