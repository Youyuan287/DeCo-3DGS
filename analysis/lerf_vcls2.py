"""Corrected VCLS on LERF-OVS.

Neighbourhood = ALL observations in the view (not just the query's pool), and
candidate similarities are computed with the same estimator (SAS observations):
    m_v(c) = mean_{o in view v} <f_o, t_c>
    s_v(r) = mean_{o: region=r, view=v} <f_o, t_c>
    s'_v  = s_v - alpha * m_v
Challenger = top-g by z(NAG) + z(evidence), evidence = mean of top-3 s'_v.
Certificate over calibrated per-view supports decides S_M vs S0.
"""
import os, sys, json
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


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def main():
    ap = ArgumentParser()
    for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--alphas", default="0.25,0.5,1.0")
    ap.add_argument("--lam", type=float, default=1.0)
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

    obs = np.load(os.path.join(args.sas, "observations_level3.npz"))
    F = obs["features"].astype(np.float32)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
    reg = obs["region_ids"].astype(np.int64)
    view = obs["view_ids"].astype(np.int64)
    by_region = {}
    for i in np.argsort(reg, kind="stable"):
        by_region.setdefault(int(reg[i]), []).append(i)

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

    def support(cals):
        out = {}
        for d in cals:
            for v, sc_ in d.items():
                if sc_ > out.get(v, -1e9):
                    out[v] = sc_
        return out

    def certify(A, B, seed=0):
        ea, eb = support(A), support(B)
        vs = sorted(set(ea) & set(eb))
        n = len(vs)
        if n < 5:
            return 0.0
        dd = np.array([eb[v] - ea[v] for v in vs])
        k = max(2, n // 2)
        rng = np.random.default_rng(seed)
        m = np.array([dd[rng.choice(n, k, replace=False)].mean() for _ in range(300)])
        return float((m > 0).mean())

    alphas = [float(x) for x in args.alphas.split(",")]
    tags = ["main", "mask", "audit"]
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
                t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)
                out = pool(sims)
                if not out:
                    continue
                released, allc = out
                g = len(released)
                s_all = F @ t
                view_mean = {}
                for v in np.unique(view):
                    m = view == v
                    view_mean[int(v)] = float(s_all[m].mean())
                # candidate per-view sims (same estimator)
                cand_sims = []
                for c in allc:
                    r = int(c[1])
                    idx = by_region.get(r, [])
                    d = {}
                    for i in idx:
                        d.setdefault(int(view[i]), []).append(float(s_all[i]))
                    cand_sims.append({v: float(np.mean(xs)) for v, xs in d.items()})
                row = dict(image=name, prompt=prompt, n=len(allc))
                row["main"] = render_iou(released, cam, h, w, gt)
                for a in alphas:
                    cals = [{v: s - a * view_mean[v] for v, s in d.items()} for d in cand_sims]
                    ev = np.array([np.mean(sorted(d.values(), reverse=True)[:3]) if d else -1e9 for d in cals])
                    sc = np.array([c[3] for c in allc], float)
                    order = list(np.argsort(-(zs(sc) + args.lam * zs(ev)), kind="stable"))
                    SM = [allc[i] for i in order[:g]]
                    A = [cals[allc.index(c)] for c in released]
                    B = [cals[allc.index(c)] for c in SM]
                    pv = certify(A, B)
                    audit = SM if (set(order[:g]) != set(range(g)) and pv >= 0.5) else released
                    row["mask_a%.2f" % a] = render_iou(SM, cam, h, w, gt)
                    row["audit_a%.2f" % a] = render_iou(audit, cam, h, w, gt)
                rows.append(row)
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    keys = ["main"] + [k for k in rows[0].keys() if k.startswith("mask_a")] + \
           [k for k in rows[0].keys() if k.startswith("audit_a")]
    summary = {k: nested(k) for k in keys}
    summary["n"] = len(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_vcls2.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()