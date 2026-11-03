"""CPU witness dump for LERF-Mask (no GPU on hn01).

For each candidate (level, index) and each sampled TRAINING view:
  mask = (labels[level] == index)            # candidate's Gaussians
  -> project their centres into the view
  -> collect the SAM superpixels the mask covers
  -> average their per-superpixel CLIP features (f.npy) and dot with the text
That is the per-view testimony a_v(c) used by the certificate.
"""
import os, sys, json, glob, csv
from argparse import ArgumentParser
import numpy as np
from plyfile import PlyData
import torch

R = "/root/rivermind-data"
ap = ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--nag", required=True)
ap.add_argument("--featdir", required=True)
ap.add_argument("--cands", required=True)
ap.add_argument("--views", type=int, default=30)
ap.add_argument("--mincover", type=float, default=0.35)
ap.add_argument("--out", required=True)
ap.add_argument("--flip", action="store_true")
args = ap.parse_args()

relags = os.path.join(R, "ReLaGS_Reproduction", "ReLaGS")
sys.path.insert(0, relags)
import open_clip  # noqa: E402

# ---- gaussians
ply = PlyData.read(os.path.join(args.model, "point_cloud", "iteration_0", "point_cloud.ply"))
v = ply["vertex"]
X = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)

nag = torch.load(args.nag, map_location="cpu")
labels = [t.numpy().astype(np.int64) for t in nag["nag"]]
print("gaussians", X.shape[0], "levels", [int(l.max()) + 1 for l in labels])

cams = json.load(open(os.path.join(args.model, "cameras.json")))
cams = [c for c in cams if "test" not in c["img_name"]]
print("train cams", len(cams))

# ---- text encoder (same family as the field)
model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
model.eval()
tok = open_clip.get_tokenizer("ViT-B-16")


def text_feat(prompt):
    with torch.no_grad():
        t = tok([prompt])
        f = model.encode_text(t).float().numpy()[0]
    return f / max(np.linalg.norm(f), 1e-9)


# ---- candidates
cands = {}
for r in csv.DictReader(open(args.cands)):
    cands.setdefault(r["prompt"], []).append((int(r["level"]), int(r["index"]), int(r["rank"])))
print("prompts", len(cands))


def project(cam, pts):
    Rm = np.asarray(cam["rotation"], float).reshape(3, 3)
    C = np.asarray(cam["position"], float)
    p = (pts - C[None, :]) @ Rm.T
    z = p[:, 2]
    ok = z > 1e-6
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["width"] / 2.0, cam["height"] / 2.0
    u = fx * p[:, 0] / np.maximum(z, 1e-9) + cx
    vv = fy * p[:, 1] / np.maximum(z, 1e-9) + cy
    if args.flip:
        vv = cam["height"] - vv
    return u, vv, ok


rows = []
view_idx = list(range(0, len(cams), max(1, len(cams) // args.views)))[: args.views]
for prompt, cl in cands.items():
    t = text_feat(prompt)
    for level, index, rank in cl:
        gi = np.where(labels[level] == index)[0]
        if len(gi) == 0:
            continue
        pts = X[gi]
        for ci in view_idx:
            cam = cams[ci]
            name = cam["img_name"].replace(".jpg", "")
            fp = os.path.join(args.featdir, name + "_f.npy")
            sp = os.path.join(args.featdir, name + "_s.npy")
            if not (os.path.exists(fp) and os.path.exists(sp)):
                continue
            u, vv, ok = project(cam, pts)
            h, w = cam["height"], cam["width"]
            m = ok & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
            if m.sum() < 50:
                continue
            S = np.load(sp)[0]
            if S.shape != (h, w):
                continue
            ui = np.clip(u[m].astype(int), 0, w - 1)
            vi = np.clip(vv[m].astype(int), 0, h - 1)
            lab = S[vi, ui]
            lab = lab[lab >= 0]
            if len(lab) == 0:
                continue
            # keep superpixels covering a reasonable share of the mask
            uniq, cnt = np.unique(lab, return_counts=True)
            keep = uniq[cnt >= args.mincover * len(lab) / max(len(uniq), 1)]
            if len(keep) == 0:
                keep = uniq
            F = np.load(fp)
            if F.shape[0] <= keep.max():
                continue
            f = F[keep].astype(np.float64).mean(0)
            f /= max(np.linalg.norm(f), 1e-9)
            rows.append(dict(scene=args.scene, prompt=prompt, rank=rank,
                             view=name, sim=float(f @ t)))
    print("  done", prompt, len(rows), flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["scene", "prompt", "rank", "view", "sim"])
    w.writeheader(); w.writerows(rows)
print("WITNESS_DUMP", args.scene, len(rows))