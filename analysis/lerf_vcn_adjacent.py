"""VCN adjacency grouping on LERF-OVS.

Seed = released selection (gap prefix). Candidates = roots that are spatially
adjacent to the seed (gaussian KD-tree, radius d) but NOT in the seed and not
text-selected. A candidate is merged iff its view-conditional residual identity
similarity to some seed member exceeds tau. Evaluation = rendered IoU vs GT.
"""
import os, sys, json
from argparse import ArgumentParser
import numpy as np
import torch
from scipy.spatial import cKDTree

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
    ap.add_argument("--taus", default="0.02,0.04,0.07")
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

    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    root_lab = snag.labels[root_level].cpu().numpy().astype(np.int64)
    tree = cKDTree(xyz)
    rs = np.random.default_rng(0)
    samp = rs.choice(len(xyz), min(5000, len(xyz)), replace=False)
    dist, _ = tree.query(xyz[samp], k=2)
    d = float(3.0 * np.median(dist[:, 1]))
    print("radius d =", d, flush=True)
    root_size = np.bincount(root_lab)

    obs = np.load(os.path.join(args.sas, "observations_level3.npz"))
    F = l2n(obs["features"].astype(np.float64))
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    mu = {int(v): F[view == v].mean(0) for v in np.unique(view)}
    res = {}
    for rid in np.unique(reg):
        idx = np.flatnonzero(reg == rid)
        d2 = {}
        for i in idx:
            v = int(view[i])
            r = F[i] - mu[v]
            n = np.linalg.norm(r)
            if n > 1e-8:
                d2.setdefault(v, []).append(r / n)
        if d2:
            res[int(rid)] = {v: np.mean(np.stack(xs), 0) for v, xs in d2.items()}

    def id_sim(a, b, max_pairs=1500):
        A, B = res.get(int(a)), res.get(int(b))
        if not A or not B:
            return -1.0
        vs = list(A.items())[:20]
        ws = list(B.items())[:20]
        sims = []
        for va, fa in vs:
            fa = fa / max(np.linalg.norm(fa), 1e-8)
            for vb, fb in ws:
                if va == vb:
                    continue
                fb = fb / max(np.linalg.norm(fb), 1e-8)
                sims.append(float(np.dot(fa, fb)))
                if len(sims) >= max_pairs:
                    break
            if len(sims) >= max_pairs:
                break
        return float(np.mean(sims)) if sims else -1.0

    def sims_for(prompt):
        vlm.encode_text(prompt)
        return [vlm.compute_similarity(f) for f in snag.feat]

    def pool(sims):
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=5, level_until=3, filter_small=50, remove_small=False)
        if not cands:
            return []
        sc = [c[3] for c in cands]
        dd = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
        k = (max(range(len(dd)), key=lambda i: dd[i]) + 1) if dd else 1
        return cands[:k], cands

    def render_iou(cands, cam, h, w, gt):
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, N_CH).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()
        u = np.logical_or(pred, gt).sum()
        return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")

    taus = [float(x) for x in args.taus.split(",")]
    tags = ["released", "adjall"] + ["merge%.2f" % t for t in taus]
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
                released, _all = out
                s0_roots = {int(c[1]) for c in released}
                gmask = np.isin(root_lab, list(s0_roots))
                gi = np.flatnonzero(gmask)
                if len(gi) > 3000:
                    gi = rs.choice(gi, 3000, replace=False)
                cnt = {}
                for p_ in xyz[gi]:
                    for j in tree.query_ball_point(p_, d):
                        rj = int(root_lab[j])
                        if rj not in s0_roots:
                            cnt[rj] = cnt.get(rj, 0) + 1
                cand_roots = [r for r, c_ in cnt.items() if c_ >= 20 and root_size[r] >= 50]
                cands = []
                for r in cand_roots:
                    level = root_level
                    cands.append([level, r, r, 0.0, None, None, int(root_size[r])])
                row = dict(image=name, prompt=prompt, n_adj=len(cands), n_rel=len(released))
                row["released"] = render_iou(released, cam, h, w, gt)
                row["adjall"] = render_iou(released + cands, cam, h, w, gt)
                # identity of each candidate to the seed (max over seed members)
                sims_id = {}
                for c in cands:
                    vals = [id_sim(c[1], m[1]) for m in released]
                    sims_id[int(c[1])] = max(vals) if vals else -1.0
                for t in taus:
                    add = [c for c in cands if sims_id[int(c[1])] > t]
                    row["merge%.2f" % t] = render_iou(released + add, cam, h, w, gt)
                    row["merge%.2f_n" % t] = len(add)
                row["id_mean_adj"] = float(np.mean(list(sims_id.values()))) if sims_id else float("nan")
                rows.append(row)
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    summary = {k: nested(k) for k in tags + ["merge%.2f_n" % t for t in taus]}
    summary["n"] = len(rows)
    summary["id_mean_adj"] = float(np.nanmean([r["id_mean_adj"] for r in rows]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_vcnadj.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: round(v, 4) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()