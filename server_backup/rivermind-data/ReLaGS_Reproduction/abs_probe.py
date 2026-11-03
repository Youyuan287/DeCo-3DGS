"""缺席查询可行性探针：给定场景 + 任意 prompt，看 NAG 分数与候选池是否非空。
用法: python abs_probe.py --scene ramen --prompts "chopsticks,volleyball,banana"
"""
import os, sys, json
import numpy as np
import torch
from argparse import ArgumentParser

R = "/root/rivermind-data"
REPO = os.path.join(R, "github_DeCo-3DGS")
sys.path.insert(0, os.path.join(REPO, "deco", "lerf_ovs"))
sys.path.insert(0, REPO)

from deco_ops import center_torch
from arguments import ModelParams, PipelineParams
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure

ap = ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--source", default=os.path.join(R, "VALA/dataset/3dgs/lerf_ovs"))
ap.add_argument("--model", default=os.path.join(R, "ReLaGS_Reproduction/ref/relags_hf"))
ap.add_argument("--center", default="image")
args = ap.parse_args()

sc = args.scene
p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", os.path.join(args.source, sc), "-m", os.path.join(args.model, sc)])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
nag = torch.load(os.path.join(args.model, sc, "sai_nag.pt"), map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure(); vlm.load_model()
if args.center == "image":
    snag.feat = center_torch(snag.feat, renorm=False)
root_level = len(snag.labels) - 1
leaf_level = root_level - 1
print("LEVELS", len(snag.labels), "root", root_level, "leaf", leaf_level, "gaussians", snag.gaussian_num, flush=True)


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


def gap_len(c):
    d = [0] + [c[i] - c[i + 1] for i in range(len(c) - 1)]
    return d.index(max(d)) if d else 0


with torch.no_grad():
    for prompt in [x.strip() for x in args.prompts.split(",") if x.strip()]:
        vlm.encode_text(prompt)
        sims = [vlm.compute_similarity(f) for f in snag.feat]
        topv, topi = torch.topk(sims[-1], min(5, sims[-1].shape[0]))
        topv = [round(float(x), 4) for x in topv]
        p5 = pool(sims, 5)
        sc_ = [round(float(c[3]), 4) for c in p5]
        g = gap_len(sc_) if sc_ else 0
        # 各层最高分（层 0 是叶）
        lay = [round(float(f.max()), 4) for f in sims]
        print(json.dumps({"prompt": prompt, "root_top5": topv, "pool_scores": sc_,
                          "gap_len": g, "n_pool": len(p5), "layer_max": lay}, ensure_ascii=False), flush=True)