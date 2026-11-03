"""Local-contrast variants on LERF-OVS (ReLaGS field, level-3 regions).

Same candidates / CLIP features / views / fusion as the certified readout; only
the per-view evidence changes.

  E0    : top-k mean of s_v(c)
  Eknn  : top-k mean of s_v(c) - mean_{n in kNN(c), vis(v)} s_v(n)
  Erand : same with a random reference set of the same size
  Ehard : s_v(c) - tau*logmeanexp over kNN referents
  Epair : E0 + eta * mean_n [ median_v delta_v * (2 q_n - 1) ]   (absolute anchor)

Epair keeps the absolute evidence on purpose: a purely relative statistic removes
"does c look like t" and only answers "does c beat its neighbours on t", which is
not what a top-1 selection needs.

Selection: top-g by z(NAG) + lam*z(E), g = gap-prefix length of the released set.
Renders released / each variant, IoU against the GT polygons.
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


def topk_mean(vals, k=3):
    v = sorted(vals, reverse=True)[:k]
    return float(np.mean(v)) if v else -1e9


def main():
    ap = ArgumentParser()
    for k in ["scene", "source", "model", "nag", "gt_root", "sas", "out"]:
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
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
    n_reg = int(reg.max()) + 1

    # region centroid from union of member gaussians
    moff = obs["member_gaussian_offsets"]
    mid = obs["member_gaussian_ids"]
    xyz = gaussians._xyz.detach().cpu().numpy().astype(np.float64)
    cen = np.zeros((n_reg, 3))
    for r in range(n_reg):
        rows = np.flatnonzero(reg == r)
        ids = np.concatenate([mid[moff[i]:moff[i + 1]] for i in rows]) if len(rows) else \
            np.array([], np.int64)
        if len(ids):
            cen[r] = xyz[np.unique(ids)].mean(0)
    d2 = ((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    order = np.argsort(d2, axis=1)
    knn = [order[r][: args.knn] for r in range(n_reg)]
    rng = np.random.default_rng(0)
    rand_ref = {}
    for r in range(n_reg):
        q = rng.choice(n_reg - 1, size=min(args.knn, n_reg - 1), replace=False)
        rand_ref[r] = np.where(q >= r, q + 1, q)

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

    def render_iou(cands, cam, h, w, gt, cache=None):
        key = None
        if cache is not None:
            key = tuple(sorted((int(c[0]), int(c[1]), int(c[2])) for c in cands))
            if key in cache:
                return cache[key]
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, N_CH).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()
        u = np.logical_or(pred, gt).sum()
        v = float(np.logical_and(pred, gt).sum() / u) if u else float("nan")
        if cache is not None:
            cache[key] = v
        return v

    rows = []
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    gt_dir = os.path.join(args.gt_root, args.scene)
    n_done = 0
    with torch.no_grad():
        for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
            if args.limit and n_done >= args.limit:
                break
            name = im[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            cam = cams[name]
            h, w = cam.image_height, cam.image_width
            prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
            icache = {}
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
                S = F @ t
                # per region -> {view: sim}
                reg_sims = {}
                for i in range(len(S)):
                    reg_sims.setdefault(int(reg[i]), {})[int(view[i])] = float(S[i])
                # candidate view table (only regions present in observations)
                cand_sims = [reg_sims.get(int(c[1]), {}) for c in allc]
                if all(len(d) == 0 for d in cand_sims):
                    continue
                tau = max(args.gamma * float(S.std()), 1e-6)

                def fill(ev):
                    ok = np.isfinite(ev)
                    if ok.any():
                        ev = np.where(ok, ev, ev[ok].mean())
                    else:
                        ev = np.zeros_like(ev)
                    return ev

                def contrast(kind):
                    ev = np.full(len(allc), np.nan)
                    for ci, d in enumerate(cand_sims):
                        if not d:
                            continue
                        refs = knn[int(allc[ci][1])] if kind != "rand" else rand_ref[int(allc[ci][1])]
                        per_view = []
                        for v, s in d.items():
                            if kind == "e0":
                                per_view.append(s)
                                continue
                            rv = [reg_sims[int(nb)].get(v) for nb in refs if int(nb) in reg_sims]
                            rv = [x for x in rv if x is not None]
                            if not rv:
                                per_view.append(s)
                                continue
                            if kind == "hard":
                                m = max(rv)
                                lse = m + tau * np.log(np.mean(np.exp((np.array(rv) - m) / tau)))
                                per_view.append(s - lse)
                            else:
                                per_view.append(s - float(np.mean(rv)))
                        ev[ci] = topk_mean(per_view)
                    return fill(ev)

                def pair_ev():
                    B = np.full(len(allc), np.nan)
                    P = np.zeros(len(allc))
                    n_ok = 0
                    for ci, d in enumerate(cand_sims):
                        if not d:
                            continue
                        c = int(allc[ci][1])
                        acc, wsum = 0.0, 0
                        for nb in knn[c]:
                            dn = reg_sims.get(int(nb))
                            if not dn:
                                continue
                            common = [v for v in d if v in dn]
                            if len(common) < 2:
                                continue
                            dv = np.array([d[v] - dn[v] for v in common])
                            acc += float(np.median(dv) * (2.0 * (dv > 0).mean() - 1.0))
                            wsum += 1
                        B[ci] = topk_mean(list(d.values()))
                        P[ci] = (acc / wsum) if wsum else 0.0
                        n_ok += 1 if wsum else 0
                    pair_ev.n_ok = n_ok
                    return fill(B), fill(P)

                sc = np.array([c[3] for c in allc], float)
                levels = [int(c[0]) for c in allc]
                row = dict(image=name, prompt=prompt, n=len(allc),
                           n_l2=sum(1 for x in levels if x == 2),
                           n_l3=sum(1 for x in levels if x == 3),
                           n_root=len({int(c[1]) for c in allc}))
                row["main"] = render_iou(released, cam, h, w, gt, icache)
                for kind, tag in [("e0", "E0"), ("knn", "Eknn"), ("rand", "Erand"),
                                  ("hard", "Ehard")]:
                    ev = contrast(kind)
                    order = list(np.argsort(-(zs(sc) + args.lam * zs(ev)), kind="stable"))
                    row[tag] = render_iou([allc[i] for i in order[:g]], cam, h, w, gt, icache)
                base_ev, pair_t = pair_ev()
                # eta is applied in z-space so it does not depend on the score scale
                ev = zs(base_ev) + args.eta * zs(pair_t)
                order = list(np.argsort(-(zs(sc) + args.lam * zs(ev)), kind="stable"))
                row["Epair"] = render_iou([allc[i] for i in order[:g]], cam, h, w, gt, icache)
                row["pair_cov"] = pair_ev.n_ok / max(len(allc), 1)
                rows.append(row)
            n_done += 1
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    keys = [k for k in rows[0].keys() if k not in ("image", "prompt", "n")]
    summary = {k: nested(k) for k in keys}
    summary["n"] = len(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_local.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()