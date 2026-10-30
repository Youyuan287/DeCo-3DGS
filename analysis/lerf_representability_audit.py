"""Representability audit: does the GT object exist in the 3DGS reconstruction?

For each query (query camera + GT 2D mask): compute per-Gaussian render weight via
render_point, keep visible Gaussians (weight > 1e-4), project their centers into the
query image, and measure support inside the GT mask:
  n_vis_gt, opac_mass_gt, pix_cover, gt_area, n_vis.
"""
import os, sys, json, csv
import numpy as np
import torch
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_point
from scene import GaussianModel, Scene
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
scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                  dtype=torch.float32, device="cuda")
xyz = (gaussians.get_xyz if hasattr(gaussians, "get_xyz") else gaussians._xyz).detach()
opac = gaussians.get_opacity.detach().flatten().cpu().numpy()

gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
rows = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        rp = render_point(cam, gaussians, pipe, bg)
        wt = rp["weight"].detach().flatten()
        vis = (wt > 1e-4)
        R = torch.from_numpy(np.asarray(cam.R)).float().cuda()
        T = torch.from_numpy(np.asarray(cam.T)).float().cuda()
        pc = xyz @ R.T + T
        z = pc[:, 2]
        front = vis & (z > 1e-4)
        fx = w / (2 * np.tan(cam.FoVx * 0.5))
        fy = h / (2 * np.tan(cam.FoVy * 0.5))
        zs = torch.clamp(z, min=1e-6)
        px = (pc[:, 0] / zs) * fx + w / 2
        py = (pc[:, 1] / zs) * fy + h / 2
        ixn = px.round().long().cpu().numpy()
        iyn = py.round().long().cpu().numpy()
        frontn = front.cpu().numpy()
        inb = frontn & (ixn >= 0) & (ixn < w) & (iyn >= 0) & (iyn < h)
        n_vis = int(frontn.sum())
        for prompt in prompts:
            nq += 1
            if args.limit_queries and nq > args.limit_queries:
                break
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            inside = np.zeros(len(opac), bool)
            inside[inb] = gt[iyn[inb], ixn[inb]]
            ngt = int(inside.sum())
            om = float(opac[inside].sum()) if ngt else 0.0
            pix = np.zeros((h, w), bool)
            if ngt:
                pix[iyn[inside], ixn[inside]] = True
            rows.append(dict(scene=args.scene, image=name, prompt=prompt,
                             gt_area=int(gt.sum()), n_vis=n_vis, n_vis_gt=ngt,
                             opac_mass_gt=om,
                             pix_cover=float(pix.sum()) / max(int(gt.sum()), 1),
                             n_total=int(xyz.shape[0])))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_representability.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))