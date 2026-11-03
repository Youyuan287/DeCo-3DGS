"""Project a few roots into a camera and save overlay PNGs to verify the
projection convention (y up vs y down) before running the SAM pilot."""
import os, sys, json
import numpy as np
from PIL import Image

sys.path[:0] = ["/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"]
import torch as th

SC = "scene0000_00"
B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
ABL = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029/ablation/v2_on_on/" + SC
MODEL = "/root/verify58_20261029/full10/" + SC + "/model"
IMG = "/root/rivermind-data/datasets/scannet/" + SC + "/images"
CAMS = "/root/rivermind-data/output/scannet/" + SC + "/cameras.json"
OUT = "/root/rivermind-data/repro/deco-b123/sam_debug_20261101"
os.makedirs(OUT, exist_ok=True)

cams = json.load(open(CAMS))
d = np.load(os.path.join(ABL, "predictions_and_scores.npz"))
xyz = d["xyz"].astype(np.float64)
labels = d["labels"].astype(np.int64)
nag = th.load(os.path.join(MODEL, "sai_nag_3.0.pt"), map_location="cpu")
root_of = nag["nag"][-1].numpy().astype(np.int64)

# choose 3 large roots
counts = np.bincount(root_of, minlength=int(root_of.max()) + 1)
cand = [r for r in range(len(counts)) if 2000 <= counts[r] <= 15000]
roots = cand[:3]
print("mid roots", roots, "sizes", counts[roots].tolist())


def project(pts, cam, flip_y):
    R = np.asarray(cam["rotation"], np.float64)
    C = np.asarray(cam["position"], np.float64)
    p = (pts - C[None, :]) @ R.T
    z = p[:, 2]
    ok = z > 1e-6
    x = cam["fx"] * p[:, 0] / np.maximum(z, 1e-6) + cam["width"] / 2
    y = cam["fy"] * p[:, 1] / np.maximum(z, 1e-6) + cam["height"] / 2
    if flip_y:
        y = cam["height"] - y
    return x, y, ok


def draw(img, x, y, ok):
    W, H = img.size
    xi = np.round(x).astype(int)
    yi = np.round(y).astype(int)
    sel = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    arr = np.asarray(img).copy()
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            xx = np.clip(xi[sel] + dx, 0, W - 1)
            yy = np.clip(yi[sel] + dy, 0, H - 1)
            arr[yy, xx] = [255, 0, 0]
    return Image.fromarray(arr), int(sel.sum())


# pick the camera where the first root has the most visible points
best = None
for ci, cam in enumerate(cams):
    g = np.flatnonzero(root_of == roots[0])
    x, y, ok = project(xyz[g], cam, False)
    W, H = cam["width"], cam["height"]
    vis = ok & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if best is None or vis.sum() > best[0]:
        best = (int(vis.sum()), ci)
ci = best[1]
cam = cams[ci]
print("camera", ci, cam["img_name"], "visible", best[0])

img_path = os.path.join(IMG, cam["img_name"] + ".jpg")
img = Image.open(img_path).convert("RGB")
for r in roots:
    g = np.flatnonzero(root_of == r)
    for flip in (False, True):
        x, y, ok = project(xyz[g], cam, flip)
        ov, n = draw(img, x, y, ok)
        tag = "flip" if flip else "std"
        ov.save(os.path.join(OUT, "root%d_%s.png" % (r, tag)))
        print("saved", r, tag, "n", n)
img.save(os.path.join(OUT, "clean.png"))