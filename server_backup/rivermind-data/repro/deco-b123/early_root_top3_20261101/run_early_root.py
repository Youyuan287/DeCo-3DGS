"""Move existing root observation top-3 readout before the root shortlist.

Only root retrieval changes. Existing root size, leaf promotion, raw NAG
candidate scores, gap selection, Gaussian supports and rendering stay fixed.
"""
from pathlib import Path
import hashlib
import time

REPO = Path('/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS')
original = REPO / 'deco/lerf_ovs/lerf_identity_dump.py'
source = original.read_text()
# Reuse the exact legacy initialization, including its frozen NAG preprocessing.
# No new centering, consistency score, penalty or feature encoder is introduced.
exec(compile(source.split('obs_npz = ')[0], str(original), 'exec'), globals())

started = time.time()
ROOT = Path('/root/rivermind-data/repro/deco-b123')
Path(args.out).mkdir(parents=True, exist_ok=True)
obs = np.load(Path(args.sas) / 'observations_level3.npz')
features = obs['features'].astype(np.float64)
features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
regions = obs['region_ids'].astype(np.int64)
assert len(features) == len(regions)
assert np.isfinite(features).all()
nroots = len(snag.feat[-1])
assert (regions >= 0).all() and (regions < nroots).all()
root_obs = {int(r): np.flatnonzero(regions == r) for r in np.unique(regions)}
counts = torch.bincount(snag.labels[root_level], minlength=nroots).cpu().numpy()
eligible_roots = np.flatnonzero(counts > 50)
references = list(csv.DictReader((ROOT / 'lerf_pooldepth_20261030' /
                                f'{args.scene}_pooldepth.csv').open()))
with (ROOT / 'evidence_attribution_20261101/results' /
      f'{args.scene}_attribution.csv').open() as f:
    families = {(row['image'], row['prompt']): row['family'] for row in csv.DictReader(f)}

def original_pool(sims):
    vals, ids = torch.topk(sims[-1], min(10, len(sims[-1])))
    keep = [counts[int(i)] > 50 for i in ids]
    ids, vals = ids[keep], vals[keep]
    rc = snag._build_root_candidates_with_leaf_info(ids, vals, sims, leaf_level)[:5]
    return sorted(snag._analyze_leaf_candidates(rc, leaf_level),
                  key=lambda c: c[3], reverse=True), [int(c[1]) for c in rc]

def early_pool(sims, text):
    similarities = features @ text
    obs_scores = {}
    for rid in eligible_roots:
        ix = root_obs.get(int(rid), ())
        if len(ix) >= 3:
            obs_scores[int(rid)] = float(np.sort(similarities[ix])[-3:].mean())
    # Missing evidence is not a negative score. Fill remaining shortlist slots
    # by the original raw root order only if fewer than five valid roots exist.
    chosen = sorted(obs_scores, key=lambda r: (-obs_scores[r], r))[:5]
    fallback_slots = 0
    if len(chosen) < 5:
        for rid in sorted(eligible_roots, key=lambda r: (-float(sims[-1][r]), int(r))):
            if int(rid) not in chosen:
                chosen.append(int(rid))
                fallback_slots += 1
            if len(chosen) == 5:
                break
    ids = torch.as_tensor(chosen, dtype=torch.long, device=sims[-1].device)
    rc = snag._build_root_candidates_with_leaf_info(ids, sims[-1][ids], sims, leaf_level)
    candidates = sorted(snag._analyze_leaf_candidates(rc, leaf_level),
                        key=lambda c: c[3], reverse=True)
    return candidates, chosen, len(obs_scores), fallback_slots, obs_scores

def gap_selection(candidates):
    scores = [float(c[3]) for c in candidates]
    gaps = [0.] + [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    return candidates[:int(np.argmax(gaps))] if scores else []

def candidate_key(c):
    return int(c[0]), int(c[2])

def rendered(candidates, camera, h, w):
    support = (snag._get_related_gaussians(candidates) if candidates else
               torch.zeros(snag.gaussian_num, 1))
    gaussians._semantics = support.expand(-1, 20).cuda()
    sem = render(camera, gaussians, pipe, bg)['semantics']
    return (sem.reshape(sem.shape[0], -1)[0] > .5).reshape(h, w).cpu().numpy()

def mask_iou(pred, gt):
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else float('nan')

cameras = {c.image_name: c for c in scene.getTrainCameras()}
query_cache = {}
mask_cache = {}
header = ['scene', 'image', 'prompt', 'A_iou', 'B_iou', 'A_pool_oracle',
          'B_pool_oracle', 'A_pool_n', 'B_pool_n', 'root_changed', 'B_obs_roots',
          'B_fallback_slots', 'baseline_error', 'family']
out = Path(args.out)
with (out / f'{args.scene}_early_root.csv').open('w', newline='') as f, \
     (out / f'{args.scene}_choices.jsonl').open('w') as details, torch.no_grad():
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for qi, ref in enumerate(references):
        if args.limit_queries and qi >= args.limit_queries:
            break
        image, prompt = ref['image'], ref['prompt']
        if prompt not in query_cache:
            vlm.encode_text(prompt)
            sims = [vlm.compute_similarity(feat) for feat in snag.feat]
            text = vlm.text_feature[0].detach().cpu().numpy().astype(np.float64)
            a, ar = original_pool(sims)
            b, br, valid, fallback, obs_scores = early_pool(sims, text)
            query_cache[prompt] = a, ar, b, br, valid, fallback, obs_scores
        a, ar, b, br, valid, fallback, obs_scores = query_cache[prompt]
        selections = [gap_selection(a), gap_selection(b)]
        camera = cameras[image]
        h, w = camera.image_height, camera.image_width
        # Final selections are fixed before any GT mask is loaded.
        predictions = [rendered(sel, camera, h, w) for sel in selections]
        annotation = json.load((Path(args.gt_root) / args.scene / f'{image}.json').open())
        gt = np.zeros((h, w), np.uint8)
        for obj in annotation['objects']:
            if obj['category'] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), obj['segmentation']))
        gt = gt.astype(bool)
        values = [mask_iou(pred, gt) for pred in predictions]
        error = abs(values[0] - float(ref['gap5']))
        if error > 1e-6:
            raise RuntimeError(f'BASELINE_MISMATCH {image} {prompt}: {error}')
        oracles = []
        candidate_ious = {}
        for candidates in (a, b):
            ious = []
            for c in candidates:
                key = (image, *candidate_key(c))
                if key not in mask_cache:
                    mask_cache[key] = np.packbits(rendered([c], camera, h, w))
                pred = np.unpackbits(mask_cache[key], count=h*w).reshape(h, w).astype(bool)
                candidate_iou = mask_iou(pred, gt)
                ious.append(candidate_iou)
                candidate_ious[f'{int(c[0])}:{int(c[2])}'] = candidate_iou
            oracles.append(max(ious, default=0.))
        writer.writerow(dict(scene=args.scene, image=image, prompt=prompt,
            A_iou=values[0], B_iou=values[1], A_pool_oracle=oracles[0],
            B_pool_oracle=oracles[1], A_pool_n=len(a), B_pool_n=len(b),
            root_changed=int(set(ar) != set(br)), B_obs_roots=valid,
            B_fallback_slots=fallback, baseline_error=error,
            family=families.get((image, prompt), 'unknown')))
        f.flush()
        details.write(json.dumps(dict(image=image, prompt=prompt, A_roots=ar, B_roots=br,
            A_candidates=[candidate_key(c) for c in a],
            B_candidates=[candidate_key(c) for c in b],
            A_selected=[candidate_key(c) for c in selections[0]],
            B_selected=[candidate_key(c) for c in selections[1]],
            candidate_ious=candidate_ious,
            B_root_scores={str(r): obs_scores.get(r) for r in br})) + '\n')
        print('QUERY', qi + 1, image, prompt, values, oracles, flush=True)

(out / f'{args.scene}_metadata.json').write_text(json.dumps(dict(
    scene=args.scene, elapsed_seconds=time.time()-started, obs_features=list(features.shape),
    eligible_roots=len(eligible_roots), observations_sha256=hashlib.sha256(
        (Path(args.sas)/'observations_level3.npz').read_bytes()).hexdigest(),
    original_script_sha256=hashlib.sha256(source.encode()).hexdigest()), indent=2))
print('SCENE_DONE', args.scene, time.time()-started, flush=True)
