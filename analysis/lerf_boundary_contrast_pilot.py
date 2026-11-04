#!/usr/bin/env python
"""Pre-registered pilot: boundary-contrast criterion J on LERF-OVS candidates.

See experiments/cvpr27_8h_discovery/20_BOUNDARY_CONTRAST_PREREG.md (frozen before running).

J = mean_{erode(M,w)} a_v(p) - mean_{dilate(M,w)\\M} a_v(p)
a_v(p) = rendered per-Gaussian text affinity, alpha-normalised.

Reuses the frozen candidate rows from cnone_gate_tf/<scene>_segidentity.csv
(level, cand_id, score, iou, area_ratio); does NOT re-run candidate search.
"""
import os, sys, json, csv, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_dilation

RE = os.environ.get("RELAGS_REPO", "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS")
sys.path.insert(0, RE)

from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG


def parse():
    ap = argparse.ArgumentParser()
    for k in ["scene", "source", "model", "nag", "units_csv", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--w", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse()
    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    # Scene/model args are supplied via env so the shared parser stays untouched.
    argv = ["-s", a.source, "-m", a.model]
    p.add_argument("--path_pred", default="")
    pargs = p.parse_args(argv)
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)

    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(a.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    labels = [l.cpu().numpy().astype(np.int64) for l in snag.labels]
    NG = int(labels[0].shape[0])
    print("gaussians from nag: %d ; labels per level %s" %
          (NG, [int(l.max()) + 1 for l in labels]), flush=True)

    vlm = ClipSimMeasure()
    vlm.load_model()

    cams = {c.image_name: c for c in scene.getTrainCameras()}
    print("cameras: %d" % len(cams), flush=True)

    units = {}
    with open(a.units_csv) as f:
        for r in csv.DictReader(f):
            units.setdefault((r["image"], r["prompt"]), []).append(r)
    keys = sorted(units)
    if a.limit:
        keys = keys[:a.limit]
    print("units: %d" % len(keys), flush=True)

    W = a.w
    out_rows, worst_score = [], 0.0
    with torch.no_grad():
        for ui, key in enumerate(keys):
            im, prompt = key
            cam = cams.get(im)
            if cam is None:
                print("WARN missing camera %s" % im, flush=True)
                continue
            h, w = cam.image_height, cam.image_width
            vlm.encode_text(prompt)
            sims = [vlm.compute_similarity(f.float()) for f in snag.feat]

            # per-level rendered affinity map, alpha-normalised
            amap = {}
            for r in units[key]:
                lev = int(r["level"])
                if lev in amap:
                    continue
                k = lev - 1                       # snag.feat indexes levels 1..3
                a_g = sims[k][torch.from_numpy(labels[lev])].float()
                sem = torch.zeros((NG, 20), device="cuda", dtype=torch.float32)
                sem[:, 0] = a_g
                sem[:, 1] = 1.0
                gaussians._semantics = sem
                o = render(cam, gaussians, pipe, bg)["semantics"]
                o = o.reshape(o.shape[0], -1)
                num, den = o[0].cpu().numpy(), o[1].cpu().numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    av = num / den
                av[den < 1e-3] = np.nan
                amap[lev] = av.reshape(h, w)

            for r in units[key]:
                lev, cid = int(r["level"]), int(r["cand_id"])
                G = labels[lev] == cid
                if G.sum() == 0:
                    continue
                rel = torch.from_numpy(G.astype(np.float32)).view(-1, 1).cuda()
                gaussians._semantics = rel.expand(-1, 20)
                o = render(cam, gaussians, pipe, bg)["semantics"]
                M = (o.reshape(o.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()

                av = amap[lev]
                row = dict(scene=a.scene, image=im, prompt=prompt, rank=int(r["rank"]),
                           level=lev, cand_id=cid, n_points=int(r["n_points"]),
                           score=float(r["score"]), iou=float(r["iou"]),
                           area_ratio=float(r["area_ratio"]),
                           mask_px=int(M.sum()), unit_px=int(h * w))

                def estimate(ww):
                    I = binary_erosion(M, iterations=ww)
                    D = binary_dilation(M, iterations=ww)
                    O = D & ~M
                    if M.sum() < 20 or I.sum() < 20 or O.sum() < 20:
                        return np.nan, np.nan, np.nan, int(I.sum()), int(O.sum())
                    ai, ao = av[I], av[O]
                    if np.isnan(ai).mean() > 0.5 or np.isnan(ao).mean() > 0.5:
                        return np.nan, np.nan, np.nan, int(I.sum()), int(O.sum())
                    mi, mo = float(np.nanmean(ai)), float(np.nanmean(ao))
                    return mi - mo, mi, mo, int(I.sum()), int(O.sum())

                J, a_in, a_ring, ni, no_ = estimate(W)
                J3, _, _, _, _ = estimate(3)
                J9, _, _, _, _ = estimate(9)
                row.update(J=J, J_w3=J3, J_w9=J9, a_in=a_in, a_ring=a_ring,
                           interior_px=ni, ring_px=no_)
                # self-check: CSV score must equal sims[level-1][cand_id]
                ref = float(sims[lev - 1][cid].item())
                row["score_recomputed"] = ref
                worst_score = max(worst_score, abs(ref - float(r["score"])))
                out_rows.append(row)

            if (ui + 1) % 10 == 0:
                print("  unit %d/%d rows=%d worst_score_diff=%.2e"
                      % (ui + 1, len(keys), len(out_rows), worst_score), flush=True)

    fn = ["scene", "image", "prompt", "rank", "level", "cand_id", "n_points",
          "score", "score_recomputed", "iou", "area_ratio", "mask_px", "unit_px",
          "J", "J_w3", "J_w9", "a_in", "a_ring", "interior_px", "ring_px"]
    path = os.path.join(a.out, "%s_units.csv" % a.scene)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fn)
        wr.writeheader()
        for r in out_rows:
            wr.writerow(r)
    print("SCORE_CHECK max|recomputed-csv| = %.3e" % worst_score, flush=True)
    print("WROTE %s rows=%d" % (path, len(out_rows)), flush=True)


if __name__ == "__main__":
    main()