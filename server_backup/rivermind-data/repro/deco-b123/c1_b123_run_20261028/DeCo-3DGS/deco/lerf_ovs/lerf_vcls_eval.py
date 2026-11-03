"""Per-query standardization + view-conditioned local scaling (VCLS) on LERF-OVS.

For each query: pool = top-5 (stock search); released = gap prefix S0.
Per-view local scaling of candidate similarities:
    s'_v(c) = s_v(c) - alpha * mean_{c' in pool, visible in v} s_v(c')
evidence(c) = mean of top-3 calibrated view sims.
Challenger S_M = top-g by z(NAG score) + z(evidence).
Certificate over calibrated per-view supports decides S_M vs S0.
Renders S0 / S_M / audit; IoU vs GT polygons.
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


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def main():
    ap = ArgumentParser()
    for k in ["scene", "source", "model", "nag", "gt_root", "witness", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
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

    wit = {}
    with open(os.path.join(args.witness, "%s_witness.csv" % args.scene)) as f:
        for r in csv.DictReader(f):
            wit.setdefault((r["image"], r["prompt"], int(r["rank"])), {})[int(r["view_id"])] = float(r["sim"])

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

    def calibrate(sim_dicts):
        """si: list of {view: sim} per candidate -> list of calibrated dicts"""
        views = set()
        for d in sim_dicts:
            views |= set(d)
        mean_v = {}
        for v in views:
            vals = [d[v] for d in sim_dicts if v in d]
            mean_v[v] = float(np.mean(vals)) if vals else 0.0
        return [{v: s - args.alpha * mean_v[v] for v, s in d.items()} for d in sim_dicts]

    def support(cals):
        out = {}
        for d in cals:
            for v, s in d.items():
                if s > out.get(v, -1e9):
                    out[v] = s
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
                n = len(allc)
                raw = [wit.get((name, prompt, i + 1), {}) for i in range(n)]
                if all(len(d) == 0 for d in raw):
                    continue
                cals = calibrate(raw)
                ev = np.array([np.mean(sorted(d.values(), reverse=True)[:3]) if d else np.nan
                               for d in cals])
                if np.isfinite(ev).any():
                    ev = np.where(np.isfinite(ev), ev, np.nanmean(ev))
                else:
                    continue
                sc = np.array([c[3] for c in allc], float)
                g = len(released)
                order = list(np.argsort(-(zs(sc) + args.lam * zs(ev)), kind="stable"))
                SM = [allc[i] for i in order[:g]]
                pv = certify([cals[allc.index(c)] for c in released],
                             [cals[allc.index(c)] for c in SM])
                audit = SM if (set(order[:g]) != set(range(g)) and pv >= 0.5) else released
                row = dict(image=name, prompt=prompt,
                           main=render_iou(released, cam, h, w, gt),
                           mask=render_iou(SM, cam, h, w, gt),
                           audit=render_iou(audit, cam, h, w, gt), cert=pv)
                rows.append(row)
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    summary = {k: nested(k) for k in ("main", "mask", "audit")}
    summary["n"] = len(rows)
    summary["fires"] = int(sum(1 for r in rows if r["audit"] == r["mask"] and abs(r["audit"] - r["main"]) + abs(r["mask"] - r["main"]) > 1e-12))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_vcls_lerf.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()