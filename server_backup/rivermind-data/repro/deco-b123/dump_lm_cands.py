"""Dump the LERF-Mask candidate lists per test prompt (GPU side)."""
import os, sys, json, csv
from argparse import ArgumentParser
import numpy as np
import torch

RELAGS = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RELAGS)
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG

ap = ArgumentParser()
for k in ["scene", "eval_source", "model", "nag", "gt_root", "out"]:
    ap.add_argument("--" + k, required=True)
args = ap.parse_args()

p = ArgumentParser(); lp = ModelParams(p); pp = PipelineParams(p)
pargs = p.parse_known_args(["-s", args.eval_source, "-m", args.model])[0]
dataset = lp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
snag.feat = [f.cuda().float() for f in snag.feat]
vlm = ClipSimMeasure(); vlm.load_model()

split = json.load(open(os.path.join(args.eval_source, "split.json")))
prompts = []
for e in split["test"]:
    for pr in e["prompts"]:
        if pr not in prompts:
            prompts.append(pr)

rows = []
with torch.no_grad():
    for pr in prompts:
        vlm.encode_text(pr)
        sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=5, level_until=3, filter_small=50, remove_small=False)
        for i, c in enumerate(cands):
            rows.append(dict(scene=args.scene, prompt=pr, rank=i + 1,
                             level=int(c[0]), index=int(c[2]), sim=float(c[3])))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scene", "prompt", "rank", "level", "index", "sim"])
    w.writeheader(); w.writerows(rows)
print("CAND_DUMP", args.scene, len(prompts), len(rows))
