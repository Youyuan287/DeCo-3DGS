"""Score a ScanNet field with the published simple readout (x @ one_word.T),
so the VCL field can be compared against the off/on arms on equal footing."""
import os, sys, json, glob
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

CODE = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"
REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [CODE, REL, os.path.join(REL, "ext")]
from eval_scannet_3d import load_gt, load_gaussians, metrics, NYU40, SUBSETS, text_features
import score_scannet as published


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    ap = ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_root", required=True)
    args = ap.parse_args()

    gt_path = glob.glob(os.path.join(args.input_root, "*_vh_clean_2.labels.ply"))
    assert gt_path, "no gt ply in %s" % args.input_root
    gtxyz, gtlabel = load_gt(gt_path[0])
    xyz, opacity, keep, ply = load_gaussians(args.model, 0)
    dist, idx = cKDTree(gtxyz).query(xyz, k=1, workers=8)
    gt = gtlabel[idx]

    nag = torch.load(os.path.join(args.model, "sai_nag_3.0.pt"), map_location="cpu")
    labels = nag["nag"][-1].numpy().astype(np.int64)
    feats = nag["nag_feat"][-1].numpy().astype(np.float32)
    cls = SUBSETS[19]
    names = [NYU40[c] for c in cls]
    one_word = text_features(names, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                             prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
    one_word = l2n(one_word)
    x = l2n(feats)
    scores = x @ one_word.T
    pred = np.array(cls)[scores.argmax(1)][labels]
    m = metrics(gt[keep], pred[keep], cls)
    print("RESULT", args.scene, "mIoU %.2f mAcc %.2f present %d" % (
        100 * m["mIoU"], 100 * m["mAcc"], m["present"]), flush=True)


if __name__ == "__main__":
    main()