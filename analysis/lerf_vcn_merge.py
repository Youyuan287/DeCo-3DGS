"""VCN grouping experiment on LERF-OVS.

For each query: pool = top-5 roots (stock search), released = gap2 prefix.
Identity similarity between two candidates = max over cross-view pairs of
cosine between their per-view residual features (view-conditional nuisance
removal: r = normalize(f - mean_view(f))). Candidates whose residual identity
similarity to any released member exceeds tau are merged into the selection.
Evaluation: render released / merged(tau) / all-pool, IoU vs the GT polygon.
"""
import os, sys, json, csv
from argparse import ArgumentParser
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402
from eval_lerf_ovs_v2 import polygon_to_mask  # noqa: E402

N_CH = 20


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    ap = ArgumentParser()
    for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--taus", default="0.03,0.06,0.10")
    args = ap.parse_args()

    p = ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, N_CH)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()
    root_level = len(snag.labels) - 1
    leaf_level = root_level - 1

    obs = np.load(os.path.join(args.sas, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float64))
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    mu = {}
    for v in np.unique(view):
        mu[int(v)] = F[view == v].mean(0)
    # residual features per (root, view): unit vectors
    res = {}
    # group rows by root
    for rid in np.unique(reg):
        idx = np.flatnonzero(reg == rid)
        d = {}
        for i in idx:
            v = int(view[i])
            r = F[i] - mu[v]
            n = np.linalg.norm(r)
            if n > 1e-8:
                d.setdefault(v, []).append(r / n)
        if d:
            res[int(rid)] = {v: np.mean(np.stack(xs), 0) for v, xs in d.items()}

    def id_sim(root_a, root_b, topk=3):
        """mean of the top-k cross-view residual similarities (robust to the
        max-over-many-pairs inflation)"""
        A, B = res.get(int(root_a)), res.get(int(root_b))
        if not A or not B:
            return -1.0
        sims = []
        for va, fa in A.items():
            fa = fa / max(np.linalg.norm(fa), 1e-8)
            for vb, fb in B.items():
                if va == vb:
                    continue
                fb = fb / max(np.linalg.norm(fb), 1e-8)
                sims.append(float(np.dot(fa, fb)))
        if not sims:
            return -1.0
        return float(np.mean(sims))

    def sims_for(prompt):
        vlm.encode_text(prompt)
        return [vlm.compute_similarity(f) for f in snag.feat]

    def pool(sims):
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=5, level_until=3, filter_small=50, remove_small=False)
        if not cands:
            return []
        sc = [c[3] for c in cands]
        d = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
        k = (max(range(len(d)), key=lambda i: d[i]) + 1) if d else 1
        return cands[:k], cands

    def render_iou(cands, cam, h, w, gt):
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, N_CH).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()
        u = np.logical_or(pred, gt).sum()
        return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")

    taus = [float(x) for x in args.taus.split(",")]
    tags = ["released", "poolall"] + ["merge%.2f" % t for t in taus]
    rows = []
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    gt_dir = os.path.join(args.gt_root, args.scene)
    with torch.no_grad():
        for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
            name = im[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            cam = cams[name]
            h, w = cam.image_height, cam.image_width
            prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
            for prompt in prompts:
                gt = np.zeros((h, w), np.uint8)
                for o in anno["objects"]:
                    if o["category"] == prompt:
                        gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
                gt = gt.astype(bool)
                sims = sims_for(prompt)
                out = pool(sims)
                if not out:
                    continue
                released, allc = out
                row = dict(image=name, prompt=prompt, n_pool=len(allc), n_rel=len(released))
                row["released"] = render_iou(released, cam, h, w, gt)
                row["poolall"] = render_iou(allc, cam, h, w, gt)
                for t in taus:
                    merged = list(released)
                    for c in allc:
                        if c in merged:
                            continue
                        if any(id_sim(c[1], r[1]) > t for r in released):
                            merged.append(c)
                    row["merge%.2f" % t] = render_iou(merged, cam, h, w, gt)
                    row["merge%.2f_n" % t] = len(merged) - len(released)
                rows.append(row)
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    summary = {k: nested(k) for k in tags + ["merge%.2f_n" % t for t in taus]}
    summary["n"] = len(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_vcnmerge.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: round(v, 4) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()