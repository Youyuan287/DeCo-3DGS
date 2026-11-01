"""Part-whole selector gate.

For each true-part query:
  - enrichment score (local/context/parent/residual late interaction) -> top-K masks;
  - for each candidate: multi-view track consistency (8 nearest views),
    parent/child relation, geometry prior, GT IoU (evaluation only).
Dumps one row per candidate for offline selector analysis.
"""
import os, sys, json, csv, argparse
from collections import defaultdict
import numpy as np
import torch
import cv2
from PIL import Image
import open_clip
from argparse import ArgumentParser as AP

REPO = "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/deco")
sys.path.insert(0, REPO + "/deco/lerf_ovs")
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render_point
from scene import GaussianModel, Scene
from utils.vlm_utils import ClipSimMeasure

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

vlm = ClipSimMeasure()
vlm.load_model()
model, _, pre = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16")
model = model.cuda().eval()

TESTS = [
    ("figurines", "frame_00041", "pirate hat"),
    ("figurines", "frame_00105", "pirate hat"),
    ("figurines", "frame_00152", "pirate hat"),
    ("figurines", "frame_00195", "pirate hat"),
    ("teatime", "frame_00002", "bear nose"),
    ("teatime", "frame_00025", "bear nose"),
    ("teatime", "frame_00107", "bear nose"),
    ("teatime", "frame_00025", "hooves"),
    ("teatime", "frame_00043", "hooves"),
    ("teatime", "frame_00107", "dall-e brand"),
    ("teatime", "frame_00129", "dall-e brand"),
]
K = 50


def poly2mask(shape, segs):
    m = np.zeros(shape, np.uint8)
    if not segs:
        return m
    if isinstance(segs[0][0], (int, float)):
        segs = [segs]
    for p in segs:
        cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 2)], 1)
    return m


def gt_of(anno, cat, h, w):
    g = np.zeros((h, w), np.uint8)
    for o in anno["objects"]:
        if o["category"] == cat:
            g = np.maximum(g, poly2mask((h, w), o["segmentation"]))
    return g.astype(bool)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def crop_feature(im, mask, scale=1.0):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if scale != 1.0:
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        hh, ww = (y1 - y0) * scale / 2, (x1 - x0) * scale / 2
        y0, y1 = max(0, int(cy - hh)), min(im.height, int(cy + hh))
        x0, x1 = max(0, int(cx - ww)), min(im.width, int(cx + ww))
    sub = np.asarray(im.crop((x0, y0, x1, y1))).copy()
    h, w = sub.shape[:2]
    l = max(h, w)
    pad = np.zeros((l, l, 3), np.uint8)
    pad[(l - h) // 2:(l - h) // 2 + h, (l - w) // 2:(l - w) // 2 + w] = sub
    return pad


@torch.no_grad()
def encode_tiles(im, masks, scale, bs=32):
    out = []
    for i in range(0, len(masks), bs):
        x = torch.cat([pre(Image.fromarray(crop_feature(im, m, scale)))[None]
                       for m in masks[i:i + bs]], 0).cuda().half()
        z = model.encode_image(x).float()
        z = z / z.norm(dim=-1, keepdim=True)
        out.append(z.cpu())
    return torch.cat(out, 0).numpy()


def np_norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_scene(scene):
    p = AP(); lp, pp = ModelParams(p), PipelineParams(p)
    a, _ = p.parse_known_args([
        "-s", f"{args.data_root}/VALA/dataset/3dgs/lerf_ovs/{scene}",
        "-m", f"{args.data_root}/ReLaGS_Reproduction/ref/relags_hf/{scene}"])
    ds, pipe = lp.extract(a), pp.extract(a)
    gs = GaussianModel(ds.sh_degree, 20)
    sc = Scene(ds, gs, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if ds.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    return gs, pipe, bg, sc


def load_lang(data_root, scene, img):
    p = f"{data_root}/VALA/dataset/3dgs/lerf_ovs/{scene}/language_features/{img}"
    s = np.load(p + "_s.npy")
    f0 = np.load(p + "_f.npy").astype(np.float32)
    ids = np.unique(s)
    ids = ids[ids >= 0]
    ms, levels = [], []
    for mid in ids:
        lv = int(np.where((s == mid).any(axis=(1, 2)))[0][0])
        ms.append(s[lv] == mid)
        levels.append(lv)
    return s, ids, ms, np.array(levels), np_norm(f0[ids])


def project(cam, gs, pipe, bg):
    xyz = (gs.get_xyz if hasattr(gs, "get_xyz") else gs._xyz).detach()
    wt = render_point(cam, gs, pipe, bg)["weight"].detach().flatten()
    vis = wt > 1e-4
    R = torch.from_numpy(np.asarray(cam.R)).float().cuda()
    T = torch.from_numpy(np.asarray(cam.T)).float().cuda()
    pc = xyz @ R + T
    z = torch.clamp(pc[:, 2], min=1e-6)
    fx = cam.image_width / (2 * np.tan(cam.FoVx * .5))
    fy = cam.image_height / (2 * np.tan(cam.FoVy * .5))
    ix = ((pc[:, 0] / z) * fx + cam.image_width / 2).round().long().cpu().numpy()
    iy = ((pc[:, 1] / z) * fy + cam.image_height / 2).round().long().cpu().numpy()
    inb = (vis & (pc[:, 2] > 1e-4)).cpu().numpy()
    inb &= (ix >= 0) & (ix < cam.image_width) & (iy >= 0) & (iy < cam.image_height)
    return ix, iy, inb


scene_cache = {}
lang_cache = {}
rows = []
for scene, img, prompt in TESTS:
    key = (scene, img)
    if key in lang_cache:
        s, ids, ms, levels, local = lang_cache[key]
    else:
        s, ids, ms, levels, local = load_lang(args.data_root, scene, img)
        lang_cache[key] = (s, ids, ms, levels, local)
    im = Image.open(f"{args.data_root}/VALA/dataset/3dgs/lerf_ovs/{scene}/images/{img}.jpg").convert("RGB")
    anno = json.load(open(f"{args.data_root}/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label/{scene}/{img}.json"))
    gt = gt_of(anno, prompt, s.shape[1], s.shape[2])
    if not gt.any():
        continue
    # enrichment features
    ctx = encode_tiles(im, ms, 1.0)
    ctx15 = encode_tiles(im, ms, 1.5)
    areas = np.array([m.sum() for m in ms])
    parent = np.full(len(ms), -1)
    for i, m in enumerate(ms):
        best, ba = -1, 1e18
        for j, n in enumerate(ms):
            if levels[j] <= levels[i]:
                continue
            inter = (m & n).sum()
            if inter > 0 and inter / m.sum() > 0.8 and n.sum() < ba:
                best, ba = j, n.sum()
        parent[i] = best
    pf = np.zeros_like(local)
    for i in range(len(ms)):
        pf[i] = local[parent[i]] if parent[i] >= 0 else 0.0
    resid = np_norm(local - pf)
    resid[parent < 0] = 0.0
    vlm.encode_text(prompt)
    t = vlm.text_feature[0].detach().cpu().numpy()
    t = t / max(float(np.linalg.norm(t)), 1e-8)
    text_score = np.maximum.reduce([local @ t, ctx @ t, ctx15 @ t, pf @ t, resid @ t])
    order = np.argsort(-text_score)[:K]
    gt_iou = np.array([iou(m, gt) for m in ms])

    if scene not in scene_cache:
        scene_cache[scene] = load_scene(scene)
    gs, pipe, bg, sc = scene_cache[scene]
    cams = sorted(sc.getTrainCameras(), key=lambda c: c.image_name)
    by = {c.image_name: c for c in cams}
    cam = by[img]
    h, w = cam.image_height, cam.image_width
    ix, iy, inb = project(cam, gs, pipe, bg)
    cc = np.array([c.camera_center.detach().cpu().numpy() for c in cams])
    near = np.argsort(np.linalg.norm(cc - cc[cams.index(cam)], axis=1))[1:9]
    views = []
    for vi in near:
        c = cams[vi]
        if c.image_name == img:
            continue
        try:
            s2, ids2, ms2, lv2, f2 = load_lang(args.data_root, scene, c.image_name)
        except FileNotFoundError:
            continue
        ix2, iy2, inb2 = project(c, gs, pipe, bg)
        id2row = {int(mid): k for k, mid in enumerate(ids2)}
        views.append((c.image_name, s2, f2, id2row, ix2, iy2, inb2,
                      c.image_width, c.image_height))

    for rank, j in enumerate(order):
        m = ms[j]
        sel = inb & m[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
        if sel.sum() == 0:
            continue
        covs, coss = [], []
        for name2, s2, f2, id2row, ix2, iy2, inb2, w2, h2 in views:
            q = sel & inb2
            if q.sum() < 10:
                continue
            xx = np.clip(ix2[q], 0, w2 - 1)
            yy = np.clip(iy2[q], 0, h2 - 1)
            vals = s2[:, yy, xx]
            best = None
            for L in range(4):
                u, c = np.unique(vals[L], return_counts=True)
                ok = u >= 0
                if ok.any():
                    k = np.argmax(c * ok)
                    cand = (int(c[k]), int(u[k]))
                    if best is None or cand[0] > best[0]:
                        best = cand
            if best is None:
                continue
            cnt, mid = best
            if mid not in id2row:
                continue
            covs.append(cnt / q.sum())
            coss.append(float(local[j] @ f2[id2row[mid]]))
        if not covs:
            continue
        pa = max(int(areas[parent[j]]) if parent[j] >= 0 else int(areas[j]), 1)
        ys, xs = np.where(m)
        if parent[j] >= 0:
            ysp, xsp = np.where(ms[parent[j]])
            dy = (ys.mean() - ysp.mean()) / max(ysp.max() - ysp.min(), 1)
            dx = (xs.mean() - xsp.mean()) / max(xsp.max() - xsp.min(), 1)
        else:
            dy = dx = 0.0
        rows.append(dict(
            scene=scene, image=img, prompt=prompt,
            rank=int(rank), mask_idx=int(j), level=int(levels[j]),
            text_score=float(text_score[j]), gt_iou=float(gt_iou[j]),
            area=int(areas[j]), area_ratio=float(areas[j] / pa),
            parent_id=int(parent[j]), parent_score=float(pf[j] @ t),
            resid_score=float(resid[j] @ t),
            track_cov_mean=float(np.mean(covs)), track_cov_max=float(np.max(covs)),
            track_cos_mean=float(np.mean(coss)), track_cos_max=float(np.max(coss)),
            track_n=len(covs),
            track_good=int(sum((cc >= .5) and (ss >= .5) for cc, ss in zip(covs, coss))),
            dy=float(dy), dx=float(dx),
        ))
    print("DONE", scene, img, prompt, "cands", len(order), flush=True)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "part_selector_candidates.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("ALL_DONE", len(rows))