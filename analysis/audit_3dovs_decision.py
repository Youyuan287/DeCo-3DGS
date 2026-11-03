"""Decision audit on the 3D-OVS reproduction (hn01).

Released selection: topk=5, filter_small=50, cand_mode=gap2 (the reproduced protocol).
Witnesses: per-view mask-language evidence, computed from the scene's own
per-view language features (seg_map scale 3 + per-segment features), for every
train view. Challenger: same-size rerank of the same top-5 pool by witness
evidence. Certificate: paired half-sample majority over witness views.
Evaluation: same scorer / view-first aggregation as audit_3dovs.py.

GT is used only for scoring.
"""
import os, sys, json, csv, argparse
import numpy as np
import torch
import cv2

_REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
_B = "/root/rivermind-data/deco/DeCo-3DGS"
for _p in (_B, os.path.join(_B, "deco"), _REL, os.path.join(_REL, "ext")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402

N_CHANNELS = 20


def read_mask(path):
    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if gt is None else (gt > 0).astype(np.uint8)


def iou(pred, gt):
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return (inter / union) if union > 0 else float("nan")


def score_triplet(pred_render, gt, w, h, res_mode):
    if res_mode == "down":
        g = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return iou(pred_render, g)
    p8 = pred_render.astype(np.uint8) * 255
    if res_mode == "up":
        p = cv2.resize(p8, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    else:
        p = cv2.resize(p8, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR) > 0
    return iou(p, gt > 0)


def agg_rows(rows, mode="view", skip_empty=False):
    rs = [r for r in rows if r["iou"] == r["iou"]]
    if skip_empty:
        rs = [r for r in rs if r["pred_px"] > 0]
    if not rs:
        return float("nan")
    if mode == "view":
        by = {}
        for r in rs:
            by.setdefault(r["image"], []).append(r["iou"])
        return float(np.mean([np.mean(v) for v in by.values()]))
    return float(np.mean([r["iou"] for r in rs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=4)
    ap.add_argument("--level_until", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--pred_thresh", type=float, default=0.5)
    args = ap.parse_args()

    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model, "-r", str(args.resolution)])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, N_CHANNELS)
    scene = Scene(dataset, gaussians, 0, load_sem=True, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    nag = torch.load(args.nag, map_location="cpu")
    snag = SemanticNAG(nag["nag"], nag["nag_feat"])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = ClipSimMeasure()
    vlm.load_model()

    root_level = len(snag.labels) - 1
    leaf_level = root_level - 1

    def text_sims(cls):
        vlm.encode_text(cls)
        sims = [vlm.compute_similarity(f) for f in snag.feat]
        t = vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)
        return sims, t

    def pool(cls):
        """returns full top-5 pool, released gap-prefix length k, text vector"""
        sims, t = text_sims(cls)
        _, cands = snag.search_matched_superpoint_in_mhtree(
            sims, topk=5, level_until=args.level_until, filter_small=50, remove_small=False)
        if not cands:
            return [], 0, t
        sc = [c[3] for c in cands]
        d = [sc[i] - sc[i + 1] for i in range(len(sc) - 1)]
        k = (max(range(len(d)), key=lambda i: d[i]) + 1) if d else 1
        return cands, k, t

    def render_rel(cam, cands):
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, N_CHANNELS).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        h, w = cam.image_height, cam.image_width
        return (sem.reshape(sem.shape[0], -1)[0] > args.pred_thresh).reshape(h, w).cpu().numpy()



    cams = list(scene.getTrainCameras())
    seg_root = os.path.join(args.gt_root, args.scene, "segmentations")
    classes = [c.strip() for c in open(os.path.join(seg_root, "classes.txt")) if c.strip()]
    gt_views = sorted(d for d in os.listdir(seg_root) if os.path.isdir(os.path.join(seg_root, d)))
    cam_by_name = {c.image_name: c for c in cams}
    # witness views: those with language features loaded
    wit_cams = [c for c in cams if getattr(c, "semantic", None)]

    def candidate_evidence(cand, t):
        """view -> evidence for one candidate"""
        ev = {}
        for cam in wit_cams:
            try:
                mask = render_rel(cam, [cand])
            except Exception:
                continue
            if mask.sum() < 30:
                continue
            seg = cam.semantic["seg_map"][3]
            if torch.is_tensor(seg):
                seg = seg.cpu().numpy()
            sem = cam.semantic["sem"]
            if torch.is_tensor(sem):
                sem = sem.cpu().numpy()
            m = mask.astype(bool)
            smin = int(seg[m].min())
            sub = seg[m].astype(np.int64) - (smin - 1)   # VL-Fields segment ids are 1-based
            ok = (sub >= 0) & (sub < sem.shape[0])
            sub = sub[ok]
            if sub.size == 0:
                continue
            cnt = np.bincount(sub, minlength=sem.shape[0])
            ratio = cnt / float(sub.size)
            keep = np.flatnonzero(ratio >= args.tau)
            if len(keep) == 0:
                continue
            fv = (ratio[keep, None] * sem[keep]).sum(0)
            n = np.linalg.norm(fv)
            if n < 1e-8:
                continue
            ev[cam.image_name] = float(np.dot(fv / n, t))
        return ev

    def set_evidence(sel_ev):
        out = {}
        for name, ev in sel_ev.items():
            for v, s in ev.items():
                if s > out.get(v, -1e9):
                    out[v] = s
        return out

    def certify(ev_a, ev_b, seed=0):
        ea = set_evidence(ev_a)
        eb = set_evidence(ev_b)
        vs = sorted(set(ea) & set(eb))
        n = len(vs)
        if n < 5:
            return 0.0
        d = np.array([eb[v] - ea[v] for v in vs])
        k = max(2, n // 2)
        rng = np.random.default_rng(seed)
        means = np.array([d[rng.choice(n, k, replace=False)].mean() for _ in range(300)])
        return float((means > 0).mean())

    def score_selection(cands, cls):
        rows = []
        for view in gt_views:
            cam = cam_by_name.get(view)
            mp = os.path.join(seg_root, view, cls + ".png")
            if cam is None or not os.path.exists(mp):
                continue
            gt = read_mask(mp)
            h, w = cam.image_height, cam.image_width
            pred = render_rel(cam, cands)
            row = dict(image=view, prompt=cls, pred_px=int(pred.sum()), gt_px=int(gt.sum()),
                       n_cands=len(cands))
            row["iou"] = score_triplet(pred, gt, w, h, "up")
            rows.append(row)
        return rows

    out = {"scene": args.scene, "classes": classes, "gt_views": gt_views,
           "n_witness_views": len(wit_cams), "per_class": [], "configs": {}}
    rows_all = {"released": [], "challenger": [], "audit": [], "audit95": []}
    with torch.no_grad():
        for cls in classes:
            cands, k, t = pool(cls)
            if not cands:
                continue
            released = cands[:k]
            ev = {i: candidate_evidence(c, t) for i, c in enumerate(cands)}
            # aggregated candidate evidence = mean of top-3 views
            agg = {i: float(np.mean(sorted(v.values(), reverse=True)[:3])) if v else -1e9
                   for i, v in ev.items()}
            k = max(1, len(released))
            # challenger = fused rerank z(NAG) + lam*z(evidence) (same as the LERF mask rerank)
            sc = np.array([c[3] for c in cands], float)
            ag = np.array([agg[i] for i in range(len(cands))], float)
            def zs(x):
                sd = x.std()
                return (x - x.mean()) / sd if sd > 1e-12 else np.zeros_like(x)
            fused = zs(sc) + 1.0 * zs(ag)
            order = list(np.argsort(-fused, kind="stable"))
            chal_idx = order[:k]
            cert = certify({i: ev[i] for i in range(len(released))},
                           {i: ev[i] for i in chal_idx})
            chal_set = [cands[i] for i in chal_idx]
            sel = {
                "released": released,
                "challenger": chal_set,
                "audit": chal_set if cert >= 0.5 else released,
                "audit95": chal_set if cert >= 0.95 else released,
            }
            for tag, cands in sel.items():
                rows_all[tag] += score_selection(cands, cls)
            out["per_class"].append(dict(cls=cls, pool=len(released), cert=cert,
                                         chal=[int(i) for i in chal_idx],
                                         agg=[round(agg[i], 4) for i in range(len(released))]))
            print("CLASS", cls, "pool", len(released), "cert %.2f" % cert, flush=True)

    for tag, rows in rows_all.items():
        out["configs"][tag] = dict(up=agg_rows(rows, "view"),
                                   up_skip=agg_rows(rows, "view", skip_empty=True),
                                   n=len(rows))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "decision_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("RESULT", args.scene, {k: round(v["up"], 4) for k, v in out["configs"].items()}, flush=True)


if __name__ == "__main__":
    main()