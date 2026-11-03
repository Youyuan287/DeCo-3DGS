"""Same candidates/features: isolate canonical reference and sigmoid/gap effects."""
from pathlib import Path
import time

ROOT = Path('/root/rivermind-data/repro/deco-b123')
REPO = ROOT / 'c1_b123_run_20261028/DeCo-3DGS'
original = REPO / 'deco/lerf_ovs/lerf_identity_dump.py'
# Exact validated initialization and legacy NAG preprocessing, no new centering.
exec(compile(original.read_text().split('obs_npz = ')[0], str(original), 'exec'), globals())
started = time.time()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
previous = ROOT / 'early_root_top3_20261101/results'
refs = list(csv.DictReader((previous / f'{args.scene}_early_root.csv').open()))
choices = {}
with (previous / f'{args.scene}_choices.jsonl').open() as f:
    for line in f:
        c = json.loads(line)
        choices[(c['image'], c['prompt'])] = c
cameras = {c.image_name: c for c in scene.getTrainCameras()}
cache = {}

def gap_order(order, score):
    if not order:
        return []
    gaps = [0.] + [float(score[order[i]]-score[order[i+1]]) for i in range(len(order)-1)]
    return order[:int(np.argmax(gaps))]

def render_selection(keys, indexes, camera, h, w):
    candidates = [(keys[i][0], -1, keys[i][1], 0., None, None, 0) for i in indexes]
    support = snag._get_related_gaussians(candidates) if candidates else torch.zeros(snag.gaussian_num, 1)
    gaussians._semantics = support.expand(-1, 20).cuda()
    sem = render(camera, gaussians, pipe, bg)['semantics']
    return (sem.reshape(sem.shape[0], -1)[0] > .5).reshape(h, w).cpu().numpy()

def iou(pred, gt):
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum()/union) if union else float('nan')

header = ['scene', 'image', 'prompt', 'family', 'n', 'g_A', 'g_D', 'g_E',
          'A_iou', 'B_iou', 'C_iou', 'D_iou', 'E_iou', 'baseline_error',
          'control_set_equal', 'AC_set_changed', 'AD_set_changed',
          'pairs', 'raw_correct_pairs', 'margin_correct_pairs',
          'canon_hurts_pairs', 'canon_helps_pairs']
with (out/f'{args.scene}_score.csv').open('w', newline='') as f, \
     (out/f'{args.scene}_details.jsonl').open('w') as detail, torch.no_grad():
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for qi, ref in enumerate(refs):
        if args.limit_queries and qi >= args.limit_queries:
            break
        image, prompt = ref['image'], ref['prompt']
        old = choices[(image, prompt)]
        keys = [tuple(map(int, key)) for key in old['A_candidates']]
        old_selected = {tuple(map(int, key)) for key in old['A_selected']}
        if prompt not in cache:
            vlm.encode_text(prompt)
            similarities = [vlm.compute_similarity(feat) for feat in snag.feat]
            logits = [feat @ vlm.text_feature.T for feat in snag.feat]
            s = np.asarray([float(similarities[l-1][cid]) for l, cid in keys])
            r = np.asarray([float(logits[l-1][cid, 0]) for l, cid in keys])
            b = np.asarray([float(logits[l-1][cid, 1:].max()) for l, cid in keys])
            m = r-b
            if not all(np.isfinite(x).all() for x in (s, r, b, m)):
                raise RuntimeError(f'NONFINITE_SCORE {prompt}')
            order_A = sorted(range(len(keys)), key=lambda i: (-s[i], i))
            order_B = sorted(range(len(keys)), key=lambda i: (-m[i], i))
            order_C = sorted(range(len(keys)), key=lambda i: (-r[i], i))
            a = gap_order(order_A, s)
            if {keys[i] for i in a} != old_selected:
                raise RuntimeError(f'LEGACY_SELECTION_MISMATCH {prompt}')
            g = len(a)
            methods = dict(A=a, B=order_B[:g], C=order_C[:g],
                           D=gap_order(order_B, m), E=gap_order(order_C, r))
            if {keys[i] for i in methods['A']} != {keys[i] for i in methods['B']}:
                raise RuntimeError(f'MONOTONIC_CONTROL_MISMATCH {prompt}')
            cache[prompt] = keys, r, b, m, s, methods
        saved_keys, r, b, m, s, methods = cache[prompt]
        assert saved_keys == keys
        assert {keys[i] for i in methods['A']} == old_selected
        camera = cameras[image]
        h, w = camera.image_height, camera.image_width
        rendered = {}
        predictions = {}
        for name, indexes in methods.items():
            support_key = tuple(sorted(keys[i] for i in indexes))
            if support_key not in rendered:
                rendered[support_key] = render_selection(keys, indexes, camera, h, w)
            predictions[name] = rendered[support_key]
        # GT and archived candidate IoUs are used only after inference selections.
        annotation = json.load((Path(args.gt_root)/args.scene/f'{image}.json').open())
        gt = np.zeros((h, w), np.uint8)
        for obj in annotation['objects']:
            if obj['category'] == prompt:
                gt = np.maximum(gt, polygon_to_mask((h, w), obj['segmentation']))
        gt = gt.astype(bool)
        values = {name: iou(pred, gt) for name, pred in predictions.items()}
        error = abs(values['A']-float(ref['A_iou']))
        if not all(np.isfinite(value) for value in values.values()) or not np.isfinite(error):
            raise RuntimeError(f'NONFINITE_IOU {image} {prompt}')
        if error > 1e-6:
            raise RuntimeError(f'BASELINE_MISMATCH {image} {prompt} {error}')
        truth = np.asarray([float(old['candidate_ious'][f'{l}:{cid}']) for l,cid in keys])
        positives, negatives = np.flatnonzero(truth >= .5), np.flatnonzero(truth <= .1)
        pairs = [(int(p), int(n)) for p in positives for n in negatives]
        raw_ok = sum(r[p] > r[n] for p,n in pairs)
        margin_ok = sum(m[p] > m[n] for p,n in pairs)
        hurts = sum(r[p] > r[n] and m[p] < m[n] for p,n in pairs)
        helps = sum(r[p] < r[n] and m[p] > m[n] for p,n in pairs)
        sets = {name: {keys[i] for i in indexes} for name,indexes in methods.items()}
        writer.writerow(dict(scene=args.scene, image=image, prompt=prompt, family=ref['family'],
            n=len(keys), g_A=len(methods['A']), g_D=len(methods['D']), g_E=len(methods['E']),
            **{name+'_iou':value for name,value in values.items()}, baseline_error=error,
            control_set_equal=int(sets['A']==sets['B']),
            AC_set_changed=int(sets['A']!=sets['C']), AD_set_changed=int(sets['A']!=sets['D']),
            pairs=len(pairs), raw_correct_pairs=int(raw_ok), margin_correct_pairs=int(margin_ok),
            canon_hurts_pairs=int(hurts), canon_helps_pairs=int(helps)))
        f.flush()
        detail.write(json.dumps(dict(scene=args.scene, image=image, prompt=prompt,
            candidates=keys, r=r.tolist(), b=b.tolist(), m=m.tolist(), s=s.tolist(),
            candidate_ious=truth.tolist(), selections={name:[keys[i] for i in indexes]
                for name,indexes in methods.items()}))+'\n')
        print('QUERY', qi+1, image, prompt, values, flush=True)
print('SCENE_DONE', args.scene, time.time()-started, flush=True)
