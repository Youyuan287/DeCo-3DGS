"""SAM-witness pilot for the ScanNet query-level audit (self-contained views).

Pipeline per scene:
  1. root -> project into all cameras -> keep top-k views by visible-point count
  2. per view: SAM embedding once; for each assigned root, box prompt from the
     projected mask -> best-IoU SAM mask -> CLIP(masked, square crop)
  3. per-view verdicts = CLIP . text bank
  4. challenger = top-|I_c| roots by SAM evidence; certificate as in the audit
GT is only used to compute IoU.
"""
import os, sys, json, argparse
import numpy as np
import torch as th
from PIL import Image

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path[:0] = [REL, os.path.join(REL, "ext"),
                "/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet"]
from eval_scannet_3d import SUBSETS


def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def project(pts, cam):
    R = np.asarray(cam["rotation"], np.float64)
    C = np.asarray(cam["position"], np.float64)
    p = (pts - C[None, :]) @ R.T
    z = p[:, 2]
    ok = z > 1e-6
    x = cam["fx"] * p[:, 0] / np.maximum(z, 1e-6) + cam["width"] / 2
    y = cam["fy"] * p[:, 1] / np.maximum(z, 1e-6) + cam["height"] / 2
    return x, y, z, ok


def build_scene_zbuf(cam, pts_all):
    W, H = cam["width"], cam["height"]
    x, y, z, ok = project(pts_all, cam)
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    sel = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    idx = yi[sel] * W + xi[sel]
    zz = z[sel]
    order = np.argsort(-zz)
    zb = np.full(H * W, np.inf, np.float64)
    zb[idx[order]] = zz[order]
    return zb.reshape(H, W)


def visible_mask(cam, pts_root, zbuf, rel_tol=0.03, abs_tol=0.05):
    W, H = cam["width"], cam["height"]
    x, y, z, ok = project(pts_root, cam)
    xi = np.round(x).astype(np.int64)
    yi = np.round(y).astype(np.int64)
    sel = ok & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    m = np.zeros((H, W), np.uint8)
    if sel.sum() == 0:
        return m
    zref = zbuf[yi[sel], xi[sel]]
    tol = np.maximum(abs_tol, rel_tol * zref)
    keep = np.isfinite(zref) & (z[sel] <= zref + tol)
    xs, ys = xi[sel][keep], yi[sel][keep]
    m[ys, xs] = 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--abl", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views-per-root", type=int, default=3)
    ap.add_argument("--min-visible", type=int, default=40)
    ap.add_argument("--max-roots", type=int, default=0)
    ap.add_argument("--debug-masks", type=int, default=0)
    args = ap.parse_args()

    from segment_anything import sam_model_registry, SamPredictor
    import open_clip

    dev = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint="/root/rivermind-data/VALA/ckpts/sam_vit_h_4b8939.pth").to(dev)
    pred = SamPredictor(sam)
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.to(dev).eval()
    mean = th.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)[None, :, None, None]
    std = th.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)[None, :, None, None]

    p = os.path.join(args.abl, args.scene)
    d = np.load(os.path.join(p, "predictions_and_scores.npz"))
    gt = d["gt"].astype(np.int64)
    labels = d["labels"].astype(np.int64)
    S0 = d["scores_off"].astype(np.float64)
    xyz = d["xyz"].astype(np.float64)
    cls = SUBSETS[19]
    base_idx = S0.argmax(1)
    n_root = S0.shape[0]
    valid = np.isin(gt, cls)

    tb = np.load(os.path.join(p, "text_bank.npz"))
    one_word = l2n(tb["textbank"].astype(np.float32))
    if one_word.ndim == 3:
        one_word = one_word.reshape(-1, one_word.shape[-1])
    banks = one_word.reshape(one_word.shape[0] // 19, 19, -1) if one_word.shape[0] % 19 == 0 and one_word.shape[0] != 19 else one_word[None]
    banks_t = [th.tensor(b, device=dev) for b in banks]

    # root labels per gaussian come from the model next to the certified inputs
    model_nag = "/root/verify58_20261029/full10/%s/model/sai_nag_3.0.pt" % args.scene
    nag = th.load(model_nag, map_location="cpu")
    root_of = nag["nag"][-1].numpy().astype(np.int64)

    cams = json.load(open(args.cams))
    by_name = {c["img_name"]: i for i, c in enumerate(cams)}

    # root -> top-k views by visible count
    plan = {}
    root_list = list(range(n_root))[:args.max_roots] if args.max_roots else list(range(n_root))
    for rid in root_list:
        g = np.flatnonzero(root_of == rid)
        if len(g) == 0:
            plan[rid] = []
            continue
        pts = xyz[g]
        vis = []
        for ci, cam in enumerate(cams):
            x, y, z0, ok = project(pts, cam)
            W, H = cam["width"], cam["height"]
            cnt = int((ok & (x >= 0) & (x < W) & (y >= 0) & (y < H)).sum())
            if cnt >= args.min_visible:
                vis.append((cnt, ci))
        vis.sort(reverse=True)
        plan[rid] = [ci for _, ci in vis[:args.views_per_root]]
    print("planned roots", sum(1 for v in plan.values() if v), flush=True)

    # group by view
    by_view = {}
    for rid, vs in plan.items():
        for ci in vs:
            by_view.setdefault(ci, []).append(rid)
    print("unique views", len(by_view), flush=True)

    ev = {}   # rid -> view -> np.array(19)
    for ci, roots in by_view.items():
        cam = cams[ci]
        img_path = os.path.join(args.images, cam["img_name"] + ".jpg")
        if not os.path.exists(img_path):
            continue
        img = np.asarray(Image.open(img_path).convert("RGB"))
        pred.set_image(img)
        W, H = cam["width"], cam["height"]
        zbuf = build_scene_zbuf(cam, xyz)
        for rid in roots:
            g = np.flatnonzero(root_of == rid)
            mask = visible_mask(cam, xyz[g], zbuf)
            if mask.sum() < 20:
                continue
            ys, xs = np.nonzero(mask)
            pad = 8
            box = np.array([max(0, xs.min() - pad), max(0, ys.min() - pad),
                            min(W - 1, xs.max() + pad), min(H - 1, ys.max() + pad)])
            masks, _, _ = pred.predict(box=box[None, :], multimask_output=True)
            best, bi = -1, 0
            for i, mk in enumerate(masks):
                inter = np.logical_and(mk, mask).sum()
                uni = np.logical_or(mk, mask).sum()
                iou = inter / max(uni, 1)
                if iou > best:
                    best, bi = iou, i
            mk = masks[bi]
            if args.debug_masks > 0:
                dbg = img.copy()
                dbg[mask.astype(bool)] = (0.5 * dbg[mask.astype(bool)] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                edge = np.zeros_like(mk)
                edge[1:-1, 1:-1] = mk[1:-1, 1:-1] & ~(mk[:-2, 1:-1] & mk[2:, 1:-1] & mk[1:-1, :-2] & mk[1:-1, 2:])
                dbg[edge] = [0, 255, 0]
                Image.fromarray(dbg).save(os.path.join(args.out, "dbg_%s_root%d.jpg" % (cam["img_name"], rid)))
                args.debug_masks -= 1
            ys, xs = np.nonzero(mk)
            if len(xs) < 20:
                continue
            out = img.copy()
            out[~mk] = 127
            crop = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = crop.shape[:2]
            s = max(h, w)
            sq = np.full((s, s, 3), 127, np.uint8)
            sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
            t = th.tensor(sq.transpose(2, 0, 1), device=dev).float()[None] / 255.0
            t = th.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            t = (t - mean) / std
            with th.no_grad():
                f = clip.encode_image(t)
            fv = l2n(f[0].detach().cpu().numpy()[None, :])
            scores = None
            for b in banks_t:
                s2 = b @ th.tensor(fv[0], device=dev)
                scores = s2 if scores is None else th.maximum(scores, s2)
            ev.setdefault(rid, {})[ci] = scores.detach().cpu().numpy()
        print("view", ci, cam["img_name"], "roots", len(roots), flush=True)

    present = [c for c in range(19) if (gt == cls[c]).any()]
    sets = {c: [r for r in range(n_root) if base_idx[r] == c] for c in present}

    def set_evidence(S, c):
        out = {}
        for r in S:
            for v, arr in ev.get(r, {}).items():
                s = float(arr[c])
                if s > out.get(v, -1e9):
                    out[v] = s
        return out

    def certify(Sa, Sb, c, seed=0):
        ea, eb = set_evidence(Sa, c), set_evidence(Sb, c)
        vs = sorted(set(ea) & set(eb))
        n = len(vs)
        if n < 5:
            return 0.0
        dd = np.array([eb[v] - ea[v] for v in vs])
        k = max(2, n // 2)
        rng = np.random.default_rng(seed)
        means = np.array([dd[rng.choice(n, k, replace=False)].mean() for _ in range(300)])
        return float((means > 0).mean())

    def ev_score(r, c):
        arrs = [float(a[c]) for a in ev.get(r, {}).values()]
        if not arrs:
            return -1e9
        return float(np.mean(sorted(arrs, reverse=True)[:3]))

    results = {}
    fires = {"audit": 0, "audit95": 0}
    for c in present:
        inc = [r for r in sets[c] if r in ev]
        ranked = sorted([r for r in ev if ev[r]], key=lambda r: ev_score(r, c), reverse=True)
        chal = ranked[:max(len(inc), 1)]
        pv = certify(inc, chal, c)
        if pv >= 0.5:
            fires["audit"] += 1
        if pv >= 0.95:
            fires["audit95"] += 1
        mem = lambda S: np.isin(labels, np.array(S, np.int64)) if S else np.zeros(len(gt), bool)
        for tag, S in (("incumbent", sets[c]), ("chal_sam", chal),
                       ("audit", chal if pv >= 0.5 else sets[c]),
                       ("audit95", chal if pv >= 0.95 else sets[c])):
            results.setdefault(tag, {})[c] = mem(S) & valid

    def miou(pm):
        ious = []
        for c in present:
            g = gt == cls[c]
            pp = pm[c]
            u = np.logical_or(g, pp).sum()
            ious.append(float(np.logical_and(g, pp).sum() / u) if u else 0.0)
        return float(np.mean(ious))

    base = {c: (base_idx[labels] == c) & valid for c in present}
    out = {"scene": args.scene, "baseline": miou(base), "fires": fires,
           "n_roots_with_sam": len(ev), "n_classes": len(present)}
    for k in ("incumbent", "chal_sam", "audit", "audit95"):
        out[k] = miou(results[k])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sampilot_%s.json" % args.scene), "w") as f:
        json.dump(out, f, indent=2)
    print("SCENE", args.scene, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()