gt_dir = os.path.join(args.gt_root, args.scene)
cams = {c.image_name: c for c in scene.getTrainCameras()}
rows = []
nq = 0


def candidate_sel(c):
    level, root_id, cand_id, score = c[0], c[1], c[2], c[3]
    lab = lab_root if level == root_level else lab_leaf
    G = np.flatnonzero(lab == cand_id)
    if len(G) == 0:
        return None
    ridx = region_obs.get(int(root_id), np.array([], np.int64))
    if level == root_level:
        e_obs = e_g = e_w = None
        base_obs = ridx
        sel = base_obs
        cvals_sel = np.ones(len(base_obs))
    else:
        e_idx, e_g = entries_for(G)
        if e_idx is not None:
            e_obs = obs_sorted[e_idx]
            e_w = sw_sorted[e_idx]
            mass = np.bincount(e_obs, weights=e_w, minlength=No)
            co = np.clip(mass / np.where(sup_total > 0, sup_total, np.inf), 0, 1)
            cvals = co[ridx] if len(ridx) else np.array([])
        else:
            e_obs = e_w = None
            cvals = np.ones(len(ridx))
        base_obs = ridx
        sel_ = base_obs[cvals >= args.tau] if len(base_obs) else base_obs
        if len(sel_):
            sel_, cvals_sel = dedupe_views(sel_, cvals[np.isin(base_obs, sel_)])
        else:
            cvals_sel = np.array([])
        sel = sel_
    fallback = 0
    if len(sel) < 3:
        sel = base_obs
        cvals_sel = np.ones(len(sel))
        fallback = 1
    return dict(level=int(level), root_id=int(root_id), cand_id=int(cand_id),
                score=float(score), G=G, sel=sel, cvals=cvals_sel,
                e_obs=e_obs, e_g=e_g, e_w=e_w, fallback=fallback)


def identity_feats(X, cvals, t):
    a = np.asarray(cvals, float)
    a = a / max(a.sum(), 1e-12)
    fmean = (a[:, None] * X).sum(0)
    sims = X @ t
    Fc = X - raw_mean[None, :]
    fmean_c = (a[:, None] * Fc).sum(0)
    n = len(X)
    Gm = X @ X.T
    agree = float((Gm.sum() - n) / (n * (n - 1))) if n > 1 else float("nan")
    return dict(id_mean=float(np.dot(fmean / max(np.linalg.norm(fmean), 1e-12), t)),
                id_top3=float(np.mean(np.sort(sims)[-3:])),
                agree=agree, agree_c=agree,
                id_mean_c=float(np.dot(fmean_c, t)))


def srem_from(X, s, b, stats, lam, mu, kmax=8):
    """Global z-scored base, so scores are comparable across candidates."""
    n = len(X)
    if n == 0:
        return float("nan"), 0
    zs = (s - stats["s_mu"]) / stats["s_sd"]
    zb = (b - stats["b_mu"]) / stats["b_sd"]
    base = zs - lam * zb
    Gm = X @ X.T
    chosen = []
    for _ in range(min(kmax, n)):
        best, bestg = -1, -1e9
        for i in range(n):
            if i in chosen:
                continue
            red = max([float(Gm[i, j]) for j in chosen], default=0.0)
            g = float(base[i]) - mu * red
            if g > bestg:
                bestg, best = g, i
        if bestg <= 0:
            break
        chosen.append(best)
    if not chosen:
        chosen = [int(np.argmax(base))]
    score = float(np.mean([base[i] for i in chosen]))
    if len(chosen) > 1:
        red = np.mean([Gm[a, b] for a in chosen for b in chosen if a < b])
        score -= mu * float(red)
    return score, len(chosen)


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
            sims, t = sims_for(prompt)
            p5 = pool(sims, 5)
            if not p5:
                continue
            infos = []
            for i, c in enumerate(p5):
                info = candidate_sel(c)
                if info is not None:
                    info["rank"] = i + 1
                    info["c"] = c
                    infos.append(info)
            if not infos:
                continue
            toks = [F[info["sel"]] if len(info["sel"]) else np.zeros((0, F.shape[1]), np.float32)
                    for info in infos]
            pre = []
            for ii, info in enumerate(infos):
                X = F[info["sel"]].astype(np.float32)
                if len(X) == 0:
                    continue
                Bs = [toks[j] for j in range(len(infos))
                      if j != ii and infos[j]["root_id"] != info["root_id"] and len(toks[j])]
                if not Bs:
                    Bs = [toks[j] for j in range(len(infos)) if j != ii and len(toks[j])]
                B = np.concatenate(Bs, 0) if Bs else np.zeros((0, F.shape[1]), np.float32)
                if len(B) > 1200:
                    B = B[np.linspace(0, len(B) - 1, 1200).astype(int)]
                s = X @ t
                b = (X @ B.T).max(1) if len(B) else np.zeros(len(X), np.float32)
                pre.append(dict(info=info, X=X, s=s, b=b))
            if not pre:
                continue
            all_s = np.concatenate([p["s"] for p in pre]).astype(np.float64)
            all_b = np.concatenate([p["b"] for p in pre]).astype(np.float64)
            stats = dict(s_mu=float(all_s.mean()), s_sd=float(all_s.std() + 1e-8),
                         b_mu=float(all_b.mean()), b_sd=float(all_b.std() + 1e-8))
            for p in pre:
                info, X, s, b = p["info"], p["X"], p["s"], p["b"]
                s11, k11 = srem_from(X, s, b, stats, 1.0, 1.0)
                s10, k10 = srem_from(X, s, b, stats, 1.0, 0.0)
                s01, k01 = srem_from(X, s, b, stats, 0.0, 1.0)
                fe = identity_feats(X, info["cvals"], t)
                G = info["G"]
                if info["e_obs"] is None:
                    cf = float((gcnt[G] > 0).mean())
                else:
                    cf, _ = cover_stats(info["sel"], G, info["e_obs"], info["e_g"], info["e_w"])
                mask = render_mask([info["c"]], cam, h, w)
                rows.append(dict(
                    scene=args.scene, image=name, prompt=prompt, rank=info["rank"],
                    n_cands=len(p5), level=info["level"], is_leaf=int(info["level"] == leaf_level),
                    root_id=info["root_id"], cand_id=info["cand_id"], score=info["score"],
                    n_points=int(len(G)), area_ratio=float(mask.sum()) / (h * w),
                    cover_frac=cf, cover_views=int(len(np.unique(view[info["sel"]]))) if len(info["sel"]) else 0,
                    fallback=int(info["fallback"]),
                    id_mean=fe["id_mean"], id_top3=fe["id_top3"], agree=fe["agree"],
                    id_mean_c=fe["id_mean_c"], agree_c=fe["agree_c"],
                    srem=s11, srem_k=k11, srem_l1m0=s10, srem_k10=k10,
                    srem_l0m1=s01, srem_k01=k01,
                    mean_text=float(s.mean()), mean_bg=float(b.mean()), iou=iou(mask, gt)))
        print("IMG", name, len(prompts), flush=True)
        if args.limit_queries and nq >= args.limit_queries:
            break

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "%s_srem.csv" % args.scene), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print("SCENE_DONE", args.scene, len(rows))