"""Frozen-pool LERF-OVS pilot. Independent inference, followed by GT evaluation.

Never writes the upstream repository or existing results. No training, SAM,
feature encoding or NAG construction. Proposal embeddings are existing caches.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from quality_aware_scoring import quality_from_hierarchy

from support_ops import evidence_scores, gap_select, mask_iou, proposal_permutation, rescore, validate_run_inputs

R = Path('/root/rivermind-data')
RA = R/'ReLaGS_Reproduction'
MODES = ('baseline', 'visibility', 'spatial', 'permutation', 'quality')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def nodes(pool):
    return [[int(t[0]), int(t[1]), int(t[2]), float(t[3]), int(t[6])] for t in pool]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sanity', action='store_true', help='First prompt only, not a scene score')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('Refusing existing output directory: '+str(a.out))
    for path, reserve in [('/', 5), (R, 10)]:
        if shutil.disk_usage(path).free < reserve*2**30:
            raise SystemExit('Disk reserve violated: '+str(path))
    a.out.mkdir(parents=True)
    start = time.time()
    import cv2
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    spec = importlib.util.spec_from_file_location('ovs_original_eval', RA/'scripts/eval_lerf_relags.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    model = RA/'ref/relags_hf/figurines'
    source = RA/'ref/hf_only/figurines'
    gt_dir = RA/'ref/hf_lerf/lerf_ovs/label/figurines'
    cache = R/'VALA/dataset/3dgs/lerf_ovs/figurines/langsplat/language_features'
    repo = RA/'ReLaGS'
    # Query vocabulary and evaluation image names are public task inputs. Do not
    # pass polygon annotations into the inference context below.
    query_by_image = {}
    for path in sorted(gt_dir.glob('*.jpg')):
        objects = json.loads(path.with_suffix('.json').read_text())['objects']
        query_by_image[path.stem] = list(dict.fromkeys(o['category'] for o in objects))
    del objects
    prompts = list(dict.fromkeys(q for qs in query_by_image.values() for q in qs))
    if a.sanity:
        prompts = prompts[:1]
        query_by_image = {im: [q for q in qs if q in prompts] for im, qs in query_by_image.items()}
    # Strict non-evaluation views only. Choice cannot depend on query or GT IoU.
    historical=list(csv.DictReader((R/'ReLaGS_run_20261023/lerf_query_metrics.csv').open()))
    frame_names = validate_run_inputs([p.name[:-6] for p in cache.glob('*_f.npy')],query_by_image,historical)
    assert len(frame_names)>=8
    support_names = [frame_names[i] for i in np.linspace(0, len(frame_names)-1, 8).astype(int)]
    manifest = dict(dataset='LERF-OVS', scene='figurines', sanity=a.sanity,
                    baseline_origin='author_released_PT_not_self_generated',
                    topk=5, level_until=3, filter_small=50, remove_small=True,
                    threshold=0.5, support_downsample=4, support_seed=0,
                    support_views=support_names, query_by_image=query_by_image,
                    score_rule='raw_score * mean(V * IoU); visibility=raw_score*mean(V)',
                    fallback='All modes use raw scores ONLY if entire pre-gap pool unobserved; zero-IoU observed pool remains zero',
                    input_sha256={str(p): sha(p) for p in [model/'sai_nag.pt',
                        model/'point_cloud/iteration_0/point_cloud.ply', repo/'nag_data.py',
                        repo/'utils/vlm_utils.py', repo/'gaussian_renderer/__init__.py',
                        RA/'scripts/eval_lerf_relags.py', Path(__file__), Path(__file__).with_name('support_ops.py')]})
    dump(a.out/'manifest.json', manifest)
    (a.out/'command.txt').write_text(' '.join(sys.argv)+'\n', encoding='utf-8')
    p = argparse.ArgumentParser()
    lp, pp = ev.ModelParams(p), ev.PipelineParams(p)
    p.add_argument('--path_pred', default='')
    pargs = p.parse_args(['-s', str(source), '-m', str(model)])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = ev.GaussianModel(dataset.sh_degree, 20)
    scene = ev.Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0], dtype=torch.float32, device='cuda')
    cams = {c.image_name:c for c in scene.getTrainCameras()}
    assert all(im in cams for im in query_by_image), 'Evaluation camera missing'
    assert all(im in cams for im in support_names), 'Support camera missing'
    nag = torch.load(model/'sai_nag.pt', map_location='cpu')
    snag = ev.SemanticNAG(nag['nag'], nag['nag_feat'])
    quality_by_level = {
        level: quality_from_hierarchy(nag['nag_feat'], nag['nag'], level)
        for level in (2, 3)
    }
    snag.feat = [f.cuda().float() for f in snag.feat]
    assert gaussians.get_xyz.shape[0] == snag.gaussian_num
    vlm = ev.ClipSimMeasure()
    vlm.load_model()
    banks = {}
    for name in support_names:
        fp, sp = cache/(name+'_f.npy'), cache/(name+'_s.npy')
        f, seg = np.load(fp), np.load(sp)
        assert f.shape[1] == snag.feat[0].shape[1]
        cam = cams[name]
        assert seg.shape[1:] == (cam.image_height, cam.image_width), 'Camera/cache pixel mismatch'
        h, w = seg.shape[1:]
        mapping, stats = proposal_permutation(seg, len(f), seed=int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
        small = np.stack([cv2.resize(layer.astype(np.float32),(max(1,w//4),max(1,h//4)),interpolation=cv2.INTER_NEAREST).astype(np.int32) for layer in seg])
        banks[name] = dict(features=torch.from_numpy(f.astype(np.float32)).cuda(), seg=small,
                           valid=np.unique(seg[seg>=0]), mapping=mapping, stats=stats)
        manifest['input_sha256'][str(fp)] = sha(fp)
        manifest['input_sha256'][str(sp)] = sha(sp)
    manifest['permutation_stats'] = {name:b['stats'] for name,b in banks.items()}
    dump(a.out/'manifest.json', manifest)

    projection_cache = {}
    predictions = {}
    selections = {}
    query_records = []
    with torch.no_grad():
        for qi, prompt in enumerate(prompts):
            print(f'QUERY {qi+1}/{len(prompts)} {prompt}', flush=True)
            vlm.encode_text(prompt)
            sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
            vals, ids = snag._get_root_candidates(sims, 50)
            roots = snag._build_root_candidates_with_leaf_info(ids, vals, sims, len(snag.labels)-2)[:5]
            pool = snag._analyze_leaf_candidates(roots, len(snag.labels)-2)
            pool_hash = jhash(nodes(pool))
            selected_base = gap_select(pool, [float(t[3]) for t in pool])
            official_rel, official_nodes = snag.search_matched_superpoint_in_mhtree(sims, topk=5, level_until=3, filter_small=50, remove_small=True)
            assert nodes(selected_base) == nodes(official_nodes), 'Pool replay not original selector'
            assert torch.equal(snag._get_related_gaussians(selected_base), official_rel), 'Baseline Gaussian union differs'
            supports, support_log = {}, []
            for name, b in banks.items():
                pscores = vlm.compute_similarity(b['features']).cpu().numpy()
                # Full-resolution valid IDs, including tiny masks possibly lost on downsample.
                ids_valid = b['valid']
                feature_id = int(ids_valid[np.argmax(pscores[ids_valid])])
                perm_id = int(b['mapping'][feature_id])
                real = np.any(b['seg']==feature_id, axis=0)
                perm = np.any(b['seg']==perm_id, axis=0)
                supports[name] = (real, perm)
                support_log.append(dict(view=name, feature_id=feature_id, perm_mask_id=perm_id,
                                        id_changed=feature_id!=perm_id, mask_changed=not np.array_equal(real,perm),
                                        feature_score=float(pscores[feature_id]), real_pixels=int(real.sum()), perm_pixels=int(perm.sum())))
            visibility, spatial, permutation = [], [], []
            per_candidate = []
            for candidate in pool:
                level, _, index = map(int,candidate[:3])
                key = (level,index)
                if key not in projection_cache:
                    gaussians._semantics = (snag.labels[level]==index).float().unsqueeze(1).expand(-1,20).contiguous()
                    projected = {}
                    for name in support_names:
                        sem = ev.render(cams[name],gaussians,pipe,bg)['semantics'][0]
                        binary = (sem>0.5).cpu().numpy().astype(np.uint8)
                        h,w = banks[name]['seg'].shape[1:]
                        projected[name] = cv2.resize(binary,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
                    projection_cache[key] = projected
                vis, real_iou, perm_iou = [], [], []
                for name in support_names:
                    bmask = projection_cache[key][name]
                    vis.append(bool(bmask.any()))
                    real_iou.append(mask_iou(bmask,supports[name][0]))
                    perm_iou.append(mask_iou(bmask,supports[name][1]))
                vv, ss = evidence_scores(real_iou,vis)
                _, ps = evidence_scores(perm_iou,vis)
                visibility.append(vv); spatial.append(ss); permutation.append(ps)
                per_candidate.append(dict(node=[level,index], visibility=vv, spatial=ss, permutation=ps,
                                           per_view_visible=vis, per_view_real_iou=real_iou, per_view_perm_iou=perm_iou))
            all_unobserved = not any(visibility)
            raw_scores = [float(t[3]) for t in pool]
            quality_scores = []
            for t in pool:
                level, candidate_id = int(t[0]), int(t[2])
                if level not in quality_by_level or candidate_id >= len(quality_by_level[level]):
                    raise AssertionError('quality candidate outside frozen hierarchy')
                quality_scores.append(raw_scores[len(quality_scores)] * float(quality_by_level[level][candidate_id]))
            mode_scores = dict(baseline=raw_scores,
                               visibility=rescore(raw_scores,visibility,all_unobserved),
                               spatial=rescore(raw_scores,spatial,all_unobserved),
                               permutation=rescore(raw_scores,permutation,all_unobserved),
                               quality=quality_scores)
            record = dict(prompt=prompt,pool=nodes(pool),pool_sha256=pool_hash,
                          supports=support_log,evidence=per_candidate,all_unobserved=all_unobserved,modes={})
            for mode in MODES:
                # Every mode consumes exactly the same immutable query-specific pool.
                assert jhash(nodes(pool)) == pool_hash
                chosen = gap_select(pool,mode_scores[mode])
                rel = snag._get_related_gaussians(chosen)
                union = rel[:,0].numpy().astype(np.uint8)
                record['modes'][mode] = dict(pool_sha256=pool_hash,scores=mode_scores[mode],
                                             selected=nodes(chosen),gaussian_count=int(union.sum()),
                                             gaussian_union_sha256=hashlib.sha256(union.tobytes()).hexdigest())
                selections[(mode,prompt)] = record['modes'][mode]
                gaussians._semantics = rel.expand(-1,20).cuda()
                for image, qs in query_by_image.items():
                    if prompt not in qs: continue
                    sem = ev.render(cams[image],gaussians,pipe,bg)['semantics'][0]
                    pred = (sem>0.5).cpu().numpy().astype(np.uint8)
                    d = a.out/mode/'masks'/image
                    d.mkdir(parents=True,exist_ok=True)
                    safe = prompt.replace(' ','_').replace('/','_')
                    cv2.imwrite(str(d/(safe+'.png')),pred*255)
                    predictions[(mode,image,prompt)] = pred
            query_records.append(record)
            dump(a.out/'query_diagnostics.json',query_records)
            print(f'QUERY_DONE pool={len(pool)} projection_nodes_cached={len(projection_cache)}', flush=True)
        # Inference complete. GT polygons become relevant only now.
        rows = []
        for image, qs in query_by_image.items():
            anno = json.loads((gt_dir/(image+'.json')).read_text())
            h,w = cams[image].image_height,cams[image].image_width
            for prompt in qs:
                gt = np.zeros((h,w),np.uint8)
                for obj in anno['objects']:
                    if obj['category']==prompt:
                        gt = np.maximum(gt,ev.polygon_to_mask((h,w),obj['segmentation']))
                for mode in MODES:
                    pred = predictions[(mode,image,prompt)]
                    union = np.logical_or(gt,pred).sum()
                    iou = float(np.logical_and(gt,pred).sum()/union) if union else float('nan')
                    selection = selections[(mode,prompt)]
                    rows.append(dict(mode=mode,scene='figurines',image=image,prompt=prompt,iou=iou,
                                     pred_px=int(pred.sum()),gt_px=int(gt.sum()),n_candidates=len(selection['selected']),
                                     gaussian_union_sha256=selection['gaussian_union_sha256'],
                                     pool_sha256=selection['pool_sha256']))
        with (a.out/'per_query.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        # Verify against existing OVS baseline without tuning to its score.
        historical={(x['image'],x['prompt']):x for x in historical if x['scene']=='figurines'}
        base_rows=[x for x in rows if x['mode']=='baseline']
        differences=[dict(image=x['image'],prompt=x['prompt'],iou=x['iou'],historical=historical[(x['image'],x['prompt'])]['iou']) for x in base_rows if abs(x['iou']-float(historical[(x['image'],x['prompt'])]['iou']))>0.00000051 or x['pred_px']!=int(historical[(x['image'],x['prompt'])]['pred_px'])]
        dump(a.out/'baseline_recheck.json',dict(n_queries=len(base_rows),differences=differences))
        assert not differences, 'OVS original baseline replay differs; no method conclusion permitted'
        summary=[]
        for mode in MODES:
            rr=[x for x in rows if x['mode']==mode]
            means=[np.mean([x['iou'] for x in rr if x['image']==im]) for im in query_by_image if any(x['image']==im for x in rr)]
            summary.append(dict(mode=mode,miou=float(np.mean(means)),flat_miou=float(np.mean([x['iou'] for x in rr])),n_queries=len(rr)))
        status=dict(status='complete',dataset='LERF-OVS',sanity=a.sanity,summary=summary,
                    wall_seconds=time.time()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                    projection_nodes=len(projection_cache),new_artifact_bytes=sum(p.stat().st_size for p in a.out.rglob('*') if p.is_file()))
        dump(a.out/'FINAL_STATUS.json',status)
        print('FINAL_RESULT '+json.dumps(status),flush=True)


if __name__=='__main__':
    main()
