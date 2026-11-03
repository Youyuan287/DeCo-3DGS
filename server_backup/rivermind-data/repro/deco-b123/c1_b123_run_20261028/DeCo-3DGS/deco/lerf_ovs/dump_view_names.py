"""Dump view_id -> image_name mapping in the same order the SAS dumps use
(scene.getTrainCameras())."""
import os, sys, json
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene

ap = ArgumentParser()
for k in ["scene", "source", "model", "out"]:
    ap.add_argument("--" + k, required=True)
args = ap.parse_args()

p = ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset = lp.extract(pargs)
g = GaussianModel(dataset.sh_degree, 20)
scene = Scene(dataset, g, 0, load_sem=False, load_test=False)
names = [c.image_name for c in scene.getTrainCameras()]
json.dump(names, open(args.out, "w"))
print("saved", len(names), "->", args.out)