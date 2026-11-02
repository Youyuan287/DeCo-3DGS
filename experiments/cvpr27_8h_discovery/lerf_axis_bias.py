"""Two-axis additive-bias test on LERF-OVS (H1).

Scores S[g,c] = <f_g, t_c>.  A decision that ranks CANDIDATES (fixed query) is
invariant to a column constant, so the only additive term that can move it is a
per-REGION bias a_g.  zR removes the per-CLASS bias b_c, which is exactly the
term LERF cannot see.  This script tests the missing half:

    a_g = <f_g, tbar_V>       tbar_V = mean of a FIXED generic vocabulary
    s'_v(g,c) = s_v(g,c) - alpha * a_g

tbar_V is estimated offline from a word list that has nothing to do with the
evaluation queries, so the per-query open-vocabulary protocol is preserved.

Variants (same candidates / fusion / aggregation for all):
  released   the certified decision
  E0         top-3 view mean of s_v
  E0-a       top-3 view mean of s_v - alpha*a_g          <- the test
  E0-arand   same, with a_g randomly permuted across regions   (size control)
  E0-a2      a_g from a second, disjoint word list              (vocab control)
  E0-sup     top-3 view mean, shrunk by effective view count    (H2 confound ctl)
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

# Fixed generic vocabulary. Disjoint halves are used as a robustness control.
VOCAB_A = """chair table sofa bed desk lamp door window wall floor ceiling rug shelf
cabinet drawer counter sink stove oven fridge microwave kettle cup mug bowl plate
bottle jar box bag basket book paper pen pencil scissors knife fork spoon pot pan
tray cloth towel curtain pillow blanket cushion mirror clock phone screen keyboard
mouse cable switch handle knob wheel ladder box crate barrel pipe valve hinge nail
hammer screwdriver wrench saw drill brush broom mop bucket sponge soap shampoo
apple banana orange lemon grape cherry melon carrot potato onion tomato pepper corn
bread cheese butter egg meat fish rice pasta soup salad cake cookie candy coffee tea
juice water milk wine beer glass lid straw napkin tray basket plant flower tree leaf
grass stone rock sand soil wood metal plastic glass fabric leather rubber ceramic
statue toy doll ball game puzzle card dice figure model painting poster photo frame
candle vase clock radio speaker camera lens tripod stand hook rope tape glue marker
eraser ruler notebook folder envelope stamp coin key wallet watch ring necklace hat
scarf glove shoe boot sock jacket shirt pants skirt dress belt tie apron uniform
helmet mask goggles umbrella suitcase backpack purse bottle cage fence gate stairs
railing pillar roof chimney balcony fireplace heater fan air conditioner radiator
""".split()

VOCAB_B = """dog cat bird horse cow sheep goat pig chicken duck rabbit mouse rat squirrel
bear tiger lion monkey elephant zebra giraffe kangaroo penguin owl eagle shark whale
dolphin snake frog turtle lizard spider ant bee butterfly beetle crab lobster octopus
city town village street road bridge tunnel station airport hospital school library
museum church temple castle tower palace market shop cafe restaurant kitchen bedroom
bathroom garden park forest mountain river lake sea beach island desert field farm
barn factory office studio gallery cinema theater stadium harbor dock pier plaza
alley corridor lobby elevator escalator platform train bus car truck bicycle motorcycle
boat ship plane helicopter tractor rocket scooter skateboard surfboard canoe kayak
cloud rain snow wind storm fog ice fire smoke dust light shadow sound music noise
morning evening night noon summer winter spring autumn holiday festival ceremony
""".split()


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
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--supk", type=float, default=5.0)
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

    # region feature = mean over its observations (same object the NAG scores)
    freg = np.zeros((n_reg, F.shape[1]), np.float64)
    cnt = np.bincount(reg, minlength=n_reg).astype(np.float64)
    for i in range(len(F)):
        freg[reg[i]] += F[i]
    freg /= np.maximum(cnt, 1)[:, None]
    freg /= np.maximum(np.linalg.norm(freg, axis=1, keepdims=True), 1e-8)

    def vocab_dir(words):
        T = []
        for w in words:
            vlm.encode_text(w)
            T.append(vlm.text_feature[0].detach().cpu().numpy().astype(np.float64).ravel())
        T = np.stack(T)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-8)
        m = T.mean(0)
        return m / max(np.linalg.norm(m), 1e-9)

    ta = vocab_dir(VOCAB_A)
    tb = vocab_dir(VOCAB_B)
    a1 = freg @ ta
    a2 = freg @ tb
    nviews = np.bincount(reg, minlength=n_reg)
    def zn(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    a1z, a2z = zn(a1), zn(a2)
    rng = np.random.default_rng(0)
    a_rand = a1[rng.permutation(n_reg)]

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

    def render_iou(cands, cam, h, w, gt, cache):
        key = tuple(sorted((int(c[0]), int(c[1]), int(c[2])) for c in cands))
        if key in cache:
            return cache[key]
        rel = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num, 1)
        gaussians._semantics = rel.expand(-1, N_CH).cuda()
        sem = render(cam, gaussians, pipe, bg)["semantics"]
        pred = (sem.reshape(sem.shape[0], -1)[0] > 0.5).reshape(h, w).cpu().numpy()
        u = np.logical_or(pred, gt).sum()
        v = float(np.logical_and(pred, gt).sum() / u) if u else float("nan")
        cache[key] = v
        return v

    rows = []
    cams = {c.image_name: c for c in scene.getTrainCameras()}
    gt_dir = os.path.join(args.gt_root, args.scene)
    with torch.no_grad():
        for im in sorted(f for f in os.listdir(gt_dir) if f.endswith(".jpg")):
            name = im[:-4]
            anno = json.load(open(os.path.join(gt_dir, name + ".json")))
            cam = cams[name]
            h, w = cam.image_height, cam.image_width
            cache = {}
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
                S = F @ t
                reg_sims = {}
                for i in range(len(S)):
                    reg_sims.setdefault(int(reg[i]), {})[int(view[i])] = float(S[i])
                cands = [int(c[1]) for c in allc]
                if all(c not in reg_sims for c in cands):
                    continue

                def ev(shift, sup=0.0):
                    out = np.full(len(allc), np.nan)
                    for ci, r in enumerate(cands):
                        d = reg_sims.get(r, {})
                        if not d:
                            continue
                        vals = [v - shift[r] for v in d.values()]
                        base = topk_mean(vals)
                        if sup > 0:
                            n = len(vals)
                            base = (n * base) / (n + sup)
                        out[ci] = base
                    ok = np.isfinite(out)
                    if ok.any():
                        out = np.where(ok, out, out[ok].mean())
                    else:
                        out = np.zeros_like(out)
                    return out

                zero = np.zeros(n_reg)
                sc = np.array([c[3] for c in allc], float)
                row = dict(image=name, prompt=prompt, n=len(allc),
                           n_l2=sum(1 for c in allc if int(c[0]) == 2),
                           n_l3=sum(1 for c in allc if int(c[0]) == 3))
                row["main"] = render_iou(released, cam, h, w, gt, cache)
                sd = float(S.std())
                ref_sel = None
                for tag, sh, sup in [("E0", zero, 0.0),
                                     ("E0_a", args.alpha * sd * a1z, 0.0),
                                     ("E0_arand", args.alpha * sd * zn(a_rand), 0.0),
                                     ("E0_a2", args.alpha * sd * a2z, 0.0),
                                     ("E0_sup", zero, args.supk)]:
                    e = ev(sh, sup)
                    order = list(np.argsort(-(zs(sc) + args.lam * zs(e)), kind="stable"))
                    sel = tuple(sorted(order[:g]))
                    if tag == "E0":
                        ref_sel = sel
                    row["diff_" + tag] = 0 if ref_sel is None else int(sel != ref_sel)
                    row[tag] = render_iou([allc[i] for i in order[:g]], cam, h, w, gt, cache)
                rows.append(row)
            print("IMG", name, len(prompts), flush=True)

    def nested(key):
        per = {}
        for r in rows:
            per.setdefault(r["image"], []).append(r[key])
        return float(np.mean([np.mean(v) for v in per.values()]))

    keys = [k for k in rows[0].keys() if k not in ("image", "prompt", "n", "n_l2", "n_l3")
            and not k.startswith("diff_")]
    summary = {k: nested(k) for k in keys}
    for k in [k for k in rows[0].keys() if k.startswith("diff_")]:
        summary[k] = float(np.mean([r.get(k, 0) for r in rows]))
    summary["shift_absmean"] = float(np.mean(np.abs(a1z)))
    summary["n"] = len(rows)
    summary["corr_a1_nviews"] = float(np.corrcoef(a1, nviews)[0, 1])
    summary["corr_a1_a2"] = float(np.corrcoef(a1, a2)[0, 1])
    summary["a1_std"] = float(a1.std())
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s_axis.json" % args.scene), "w") as f:
        json.dump(dict(scene=args.scene, summary=summary, rows=rows), f, indent=2)
    print("RESULT", args.scene, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                            for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()