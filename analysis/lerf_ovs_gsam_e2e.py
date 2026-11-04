"""End-to-end: gate -> Grounded-SAM -> lift to 3D -> render in the query view -> IoU.

For every LERF-OVS query, if the label-free entropy of the pipeline's own score
vector exceeds a threshold fixed on the OTHER scenes (LOSO), replace the readout
with a re-observation:

    Grounding DINO (top-1 box) -> SAM box prompt -> mask with best SAM score
    -> gaussians whose projection in THIS view falls inside that mask
    -> render those gaussians in the same view -> IoU with the GT mask

Everything else keeps the released readout (compared against the baseline per-query
CSV).  Nothing is fitted; DINO and SAM are frozen.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

BASE = "/root/rivermind-data/part_grounding_20261101"
REPO = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS"
for p in (REPO, os.path.join(REPO, "deco"), os.path.join(REPO, "deco", "lerf_ovs")):
    sys.path.insert(0, p)
from arguments import ModelParams, PipelineParams          # noqa: E402
from gaussian_renderer import render, render_point        # noqa: E402
from scene import GaussianModel, Scene                    # noqa: E402
from argparse import ArgumentParser                       # noqa: E402

SRC = "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs"
MODEL = "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf"
GT = "/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label"
DINO = BASE + "/model/grounding-dino-tiny"
SAM_CKPT = "/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth"


def poly2mask(shape, segs):
    im = Image.new("L", (shape[1], shape[0]), 0)
    dr = ImageDraw.Draw(im)
    s = segs
    polys = s if (s and isinstance(s[0], (list, tuple)) and s[0]
                  and isinstance(s[0][0], (list, tuple))) else [s]
    for p in polys:
        a = np.asarray(p, np.float32).reshape(-1, 2)
        if len(a) >= 3:
            dr.polygon([tuple(q) for q in a], fill=1)
    return np.asarray(im) > 0


def gt_of(anno, cat, h, w):
    g = np.zeros((h, w), bool)
    for o in anno["objects"]:
        if o["category"] == cat:
            g = np.logical_or(g, poly2mask((h, w), o["segmentation"]))
    return g


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def load_scene(scene):
    p = ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    a, _ = p.parse_known_args(["-s", os.path.join(SRC, scene), "-m", os.path.join(MODEL, scene)])
    ds, pipe = lp.extract(a), pp.extract(a)
    gs = GaussianModel(ds.sh_degree, 20)
    sc = Scene(ds, gs, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    return gs, pipe, bg, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="figurines,ramen,teatime,waldo_kitchen")
    ap.add_argument("--thr", type=float, default=2.841)
    ap.add_argument("--box-thr", type=float, default=0.08)
    ap.add_argument("--boxes", required=True)
    args = ap.parse_args()

    from segment_anything import SamPredictor, sam_model_registry
    B = np.load(args.boxes)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to("cuda")
    spred = SamPredictor(sam)

    ent = {(r["scene"], r["frame"], r["name"]): float(r["entropy"])
           for r in csv.DictReader(open(args.entropy_csv))}

    rows = []
    for scene in args.scenes.split(","):
        gs, pipe, bg, sc = load_scene(scene)
        cams = {c.image_name: c for c in sc.getTrainCameras()}
        gdir = os.path.join(GT, scene)
        for fn in sorted(f for f in os.listdir(gdir) if f.endswith(".json")):
            fr = fn[:-5]
            cam = cams.get(fr)
            if cam is None:
                continue
            anno = json.load(open(os.path.join(gdir, fn)))
            h, w = cam.image_height, cam.image_width
            img = Image.open(os.path.join(SRC, scene, "images", fr + ".jpg")).convert("RGB")
            spred.set_image(np.asarray(img))
            xyz = (gs.get_xyz if hasattr(gs, "get_xyz") else gs._xyz).detach()
            wt = render_point(cam, gs, pipe, bg)["weight"].detach().flatten()
            vis = wt > 1e-4
            R = torch.from_numpy(np.asarray(cam.R)).float().cuda()
            T = torch.from_numpy(np.asarray(cam.T)).float().cuda()
            pc = xyz @ R + T
            z = torch.clamp(pc[:, 2], min=1e-6)
            fx = w / (2 * np.tan(cam.FoVx * .5))
            fy = h / (2 * np.tan(cam.FoVy * .5))
            ix = ((pc[:, 0] / z) * fx + w / 2).round().long().cpu().numpy()
            iy = ((pc[:, 1] / z) * fy + h / 2).round().long().cpu().numpy()
            inb = (vis & (pc[:, 2] > 1e-4)).cpu().numpy() & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            # occlusion test: keep a gaussian only if it is at the front of its pixel
            zf = pc[:, 2].cpu().numpy()
            zbuf = np.full(h * w, np.inf)
            sel = np.nonzero(inb)[0]
            if len(sel):
                px = (iy[sel] * w + ix[sel])
                o = np.argsort(-zf[sel])
                zbuf[px[o]] = zf[sel][o]
                tol = np.maximum(0.05, 0.03 * zbuf[px])
                inb[sel] = zf[sel] <= zbuf[px] + tol

            def lift(m):
                bm = np.zeros(len(xyz), bool)
                bm[inb] = m[iy[inb], ix[inb]]
                rel = torch.from_numpy(bm).float().view(-1, 1).cuda()
                gs._semantics = rel.expand(-1, 20)
                sem = render(cam, gs, pipe, bg)["semantics"]
                return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()

            print("[%s/%s]" % (scene, fr), flush=True)
            for o in anno["objects"]:
                cat = o["category"]
                gt = gt_of(anno, cat, h, w)
                if gt.sum() < 100:
                    continue
                e = ent.get((scene, fr, cat))
                if e is None:
                    continue
                row = dict(scene=scene, frame=fr, name=cat, entropy=round(e, 3),
                           gate=int(e > args.thr), iou3d=float("nan"))
                if row["gate"]:
                    key = "%s|%s|%s" % (scene, fr, cat)
                    boxes = B[key] if key in B else np.zeros((0, 4))
                    if len(boxes):
                        x0, y0, x1, y1 = boxes[0]
                        masks, scores, _ = spred.predict(
                            box=np.array([[x0, y0, x1, y1]], dtype=np.float32),
                            multimask_output=True)
                        k = int(np.argmax(np.asarray(scores)))
                        row["iou3d"] = iou(lift(masks[k].astype(bool)), gt)
                        row["has_box"] = 1
                    else:
                        row["has_box"] = 0
                rows.append(row)
                print("   %-22s e=%.2f gate=%d  iou3d=%.3f"
                      % (cat, e, row["gate"], row["iou3d"]), flush=True)
        del gs, sc
        torch.cuda.empty_cache()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "e2e_gate.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print("saved", len(rows), flush=True)


if __name__ == "__main__":
    main()