"""Certified query expansion on LERF-OVS (pilot).

For each (image, query):
  * candidates and the released prefix S0 come from the frozen field (same as
    lerf_c_mask_gate.py);
  * per-view witness sims for the query come from <scene>_witness.csv, and for
    each paraphrase from <scene>_witness_para.csv (same candidate set);
  * a paraphrase is admitted only if, on the shared views, its per-view argmax
    over the candidate set agrees with the original query's argmax in at least
    PI of the views (>=5 shared views).  This is the same selectivity idea as
    the decision certificate, moved to the query axis;
  * selection = top-g by the mean z-scored per-candidate evidence over
    {original} U admitted paraphrases.

IoUs are rendered for: released prefix (sanity), unconditional expansion,
certified expansion, and the GT-oracle best single variant (diagnostic only).

Protocol: per (image, query) IoU -> mean over queries within an image -> mean
over images -> mean over scenes.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arguments import ModelParams, PipelineParams  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from nag_data import SemanticNAG  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from utils.vlm_utils import ClipSimMeasure  # noqa: E402
from eval_lerf_ovs_v2 import polygon_to_mask  # noqa: E402

ap = argparse.ArgumentParser()
for k in ["scene", "source", "model", "nag", "gt_root", "witness", "witness_para", "out"]:
    ap.add_argument("--" + k, required=True)
ap.add_argument("--pi", type=float, default=0.5)
ap.add_argument("--limit-queries", type=int, default=0)
args = ap.parse_args()

p = argparse.ArgumentParser()
lp, pp = ModelParams(p), PipelineParams(p)
pargs, _ = p.parse_known_args(["-s", args.source, "-m", args.model])
dataset, pipe = lp.extract(pargs), pp.extract(pargs)
gaussians = GaussianModel(dataset.sh_degree, 20)
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


def sims_for(prompt):
    vlm.encode_text(prompt)
    return [vlm.compute_similarity(f) for f in snag.feat]


def pool(sims, n_roots):
    vals, ids = torch.topk(sims[-1], min(max(10, n_roots * 2), sims[-1].shape[0]))
    keep = [((snag.labels[root_level] == i).sum() > 50).item() for i in ids]
    rc = snag._build_root_candidates_with_leaf_info(ids[keep], vals[keep], sims, leaf_level)[:n_roots]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level), key=lambda x: x[3], reverse=True)


def gap_len(c):
    d = [0] + [c[i] - c[i + 1] for i in range(len(c) - 1)]
    return d.index(max(d)) if d else 0


def zs(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def load_witness(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault((r["image"], r["prompt"], int(r["rank"])), {})[int(r["view_id"])] = float(r["sim"])
    return d


def load_para(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault((r["image"], r["prompt"], r["variant"], int(r["rank"])), {})[int(r["view_id"])] = float(r["sim"])
    return d


def render_mask(cands, cam, h, w):
    rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = rel.expand(-1, 20).cuda()
    sem = render(cam, gaussians, pipe, bg)["semantics"]
    return (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()


def iou(pred, gt):
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


wit = load_witness(args.witness)
para = load_para(args.witness_para)
gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}

rows = []
nq = 0
with torch.no_grad():
    for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
        name = im[:-4]
        anno = json.load(open(os.path.join(gt_dir, name + ".json")))
        cam = cams[name]
        h, w = cam.image_height, cam.image_width
        prompts = list(dict.fromkeys(o["category"] for o in anno["objects"]))
        for prompt in prompts:
            nq += 1
            if args.limit_queries and nq > args.limit_queries:
                break
            gt = np.zeros((h, w), np.uint8)
            for o in anno["objects"]:
                if o["category"] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((h, w), o["segmentation"]))
            gt = gt.astype(bool)
            p5 = pool(sims_for(prompt), 5)
            if not p5:
                continue
            n = len(p5)
            sc = np.array([c[3] for c in p5], float)
            g = gap_len(sc)
            S0 = list(range(g)) if g > 0 else []
            base_obs = [wit.get((name, prompt, i + 1), {}) for i in range(n)]
            base_only = iou(render_mask([p5[i] for i in S0], cam, h, w), gt) if S0 else 0.0
            if any(len(o) == 0 for o in base_obs):
                rows.append(dict(scene=args.scene, image=name, prompt=prompt, n=n, gap_len=g,
                                 n_variants=0, n_admitted=0, admitted="",
                                 iou_main=base_only, iou_uncond=base_only,
                                 iou_cert=base_only, iou_best_variant=base_only))
                continue
            variants = sorted({k[2] for k in para if k[0] == name and k[1] == prompt})

            def agg(obs):
                return np.array([np.mean(list(o.values())) if o else -1e9 for o in obs])

            def set_evidence(sel, obs):
                out = {}
                for i in sel:
                    for v, s in obs[i].items():
                        if s > out.get(v, -1e9):
                            out[v] = s
                return out

            rng = np.random.default_rng(20261102)

            def certify(A, B, nboot=200):
                ea = set_evidence(A, base_obs)
                eb = set_evidence(B, base_obs)
                shared = sorted(set(ea) & set(eb))
                if len(shared) < 5 or not B:
                    return 0.0
                d = np.array([eb[v] - ea[v] for v in shared])
                k = max(2, len(d) // 2)
                return float(np.mean([d[rng.choice(len(d), k, replace=False)].mean() > 0
                                      for _ in range(nboot)]))

            p_obs = {v: [para.get((name, prompt, v, i + 1), {}) for i in range(n)] for v in variants}
            usable = [v for v in variants if all(len(o) > 0 for o in p_obs[v])]

            def sel(s):
                if len(s) == 0:
                    return []
                order = list(np.argsort(-s, kind="stable"))
                return order[:g] if g > 0 else []

            base_s = zs(sc)          # no expansion -> top-g by the field score = S0
            per_variant = {}
            for v in usable:
                s = base_s + zs(agg(p_obs[v]))
                B = sel(s)
                per_variant[v] = (B, certify(S0, B))
            admitted = [v for v in usable if per_variant[v][1] >= args.pi]
            uncond_s = base_s + (np.mean([zs(agg(p_obs[v])) for v in usable], 0) if usable else 0.0)
            cert_s = base_s + (np.mean([zs(agg(p_obs[v])) for v in admitted], 0) if admitted else 0.0)

            variant_ious = {v: iou(render_mask([p5[i] for i in per_variant[v][0]], cam, h, w), gt)
                            for v in usable if per_variant[v][0]}

            rows.append(dict(
                scene=args.scene, image=name, prompt=prompt, n=n, gap_len=g,
                n_variants=len(usable), n_admitted=len(admitted),
                admitted="|".join(admitted),
                iou_main=base_only,
                iou_uncond=iou(render_mask([p5[i] for i in sel(uncond_s)], cam, h, w), gt) if g > 0 else 0.0,
                iou_cert=iou(render_mask([p5[i] for i in sel(cert_s)], cam, h, w), gt) if g > 0 else 0.0,
                iou_best_variant=max([base_only] + list(variant_ious.values())),
            ))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
out_csv = os.path.join(args.out, "%s_expansion.csv" % args.scene)
with open(out_csv, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)


def scene_macro(key):
    per_image = {}
    for r in rows:
        per_image.setdefault(r["image"], []).append(float(r[key]))
    img_means = [np.nanmean(v) for v in per_image.values()]
    return float(np.nanmean(img_means))


print("SCENE", args.scene,
      " ".join("%s %.4f" % (k, scene_macro("iou_" + k))
               for k in ("main", "uncond", "cert", "best_variant")),
      "admitted", sum(r["n_admitted"] for r in rows), "/", sum(r["n_variants"] for r in rows),
      flush=True)