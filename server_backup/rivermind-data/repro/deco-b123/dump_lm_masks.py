"""Render each candidate's mask in sampled training views (GPU side, correct camera transform)."""
import os, sys, json, csv
from argparse import ArgumentParser
import numpy as np, torch
from PIL import Image
RELAGS = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RELAGS)
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from nag_data import SemanticNAG

ap = ArgumentParser()
for k in ["scene", "eval_source", "model", "nag", "cands", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--views", type=int, default=16)
args = ap.parse_args()
p = ArgumentParser(); lp = ModelParams(p); pp = PipelineParams(p)
pargs = p.parse_known_args(["-s", args.eval_source, "-m", args.model])[0]
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
g = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, g, 0, load_sem=False, load_test=False)
bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device="cuda")
nag = torch.load(args.nag, map_location="cpu")
snag = SemanticNAG(nag["nag"], nag["nag_feat"])
cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
idx = list(range(0, len(cams), max(1, len(cams)//args.views)))[:args.views]
print("train cams", len(cams), "sampled", len(idx))
os.makedirs(args.out, exist_ok=True)
cands = {}
for r in csv.DictReader(open(args.cands)):
    cands.setdefault(r["prompt"], []).append((int(r["level"]), int(r["index"]), int(r["rank"])))
n = 0
with torch.no_grad():
    for prompt, cl in cands.items():
        for level, index, rank in cl:
            tup = [(level, 0, index, 0.0, None, None, 0)]
            rel = snag._get_related_gaussians(tup)
            g._semantics = rel.expand(-1, 20).cuda()
            for ci in idx:
                cam = cams[ci]
                sem = render(cam, g, pipe, bg)["semantics"]
                m = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(sem.shape[1], sem.shape[2])
                m = m.cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(m).save(os.path.join(
                    args.out, "%s__r%02d__%s.png" % (prompt.replace(" ", "_"), rank, cam.image_name)))
                n += 1
        print("  ", prompt, n, flush=True)
print("MASK_DUMP", args.scene, n)
