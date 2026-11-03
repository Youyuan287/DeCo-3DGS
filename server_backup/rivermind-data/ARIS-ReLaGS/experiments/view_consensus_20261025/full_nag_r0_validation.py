"""Independent same-readout-R0 A/B runner.

The default and only locally exercised mode is ``--dry-run``. It loads the
fixed R0 on CPU, validates full level-2/3 candidate/feature mappings, and
records the A/B selector contract without reading GT or launching GPU.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def sha_bytes(a):
    return hashlib.sha256(np.asarray(a, dtype=np.uint8).tobytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def load_query_manifest(path):
    """Load only frozen image/prompt keys; reject GT/evaluation fields."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, list) or not raw:
        raise ValueError('query manifest must be a non-empty list')
    allowed = {'image', 'prompt'}
    forbidden = {'polygon', 'segmentation', 'gt', 'iou', 'mask', 'annotation'}
    out = []
    for rec in raw:
        if not isinstance(rec, dict) or set(rec) != allowed:
            keys = set(rec) if isinstance(rec, dict) else set()
            if keys & forbidden:
                raise ValueError('GT/evaluation fields are forbidden in query manifest')
            raise ValueError('query manifest records must contain exactly image and prompt')
        if not isinstance(rec['image'], str) or not isinstance(rec['prompt'], str):
            raise ValueError('image and prompt must be strings')
        out.append({'image': rec['image'], 'prompt': rec['prompt']})
    return out


def load_exact_evaluator(path):
    """Load only the approved evaluator module; caller must still authorize execution."""
    # spt exposes a top-level ``utils`` package; preload the exact ReLaGS
    # package before importing the renderer to prevent namespace shadowing.
    relags_repo = os.environ.get('RELAGS_REPO', '/root/rivermind-data/ReLaGS_Reproduction/ReLaGS')
    if relags_repo not in sys.path:
        sys.path.insert(0, relags_repo)
    importlib.import_module('utils.system_utils')
    spec = importlib.util.spec_from_file_location('full_nag_exact_evaluator', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load evaluator: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('SemanticNAG', 'ClipSimMeasure'):
        if not hasattr(module, name):
            raise RuntimeError(f'exact evaluator missing {name}')
    return module


def real_score_contract():
    return {
        'evaluator_classes': ['SemanticNAG', 'ClipSimMeasure'],
        'text_call_sequence': ['ClipSimMeasure.encode_text(prompt)',
                               'ClipSimMeasure.compute_similarity(nag_feat[level-1])'],
        'same_r0_for_A_B': True,
        'same_query_manifest_for_A_B': True,
        'same_selector': 'official gap_select / search_matched_superpoint_in_mhtree',
        'gt_opened_by_runner': False,
        'render_and_gt_evaluation': 'separate post-materialization evaluator only',
    }


def candidate_table(labels, feats):
    out = {}
    for level in (2, 3):
        fi = level - 1
        ids = np.unique(labels[level]).astype(int).tolist()
        rows = []
        for cid in ids:
            parent_ids = np.unique(labels[3][labels[level] == cid]).astype(int).tolist() if level == 2 else []
            rows.append({'level': level, 'root_id': cid if level == 3 else None,
                         'candidate_id': cid, 'gaussian_count': int((labels[level] == cid).sum()),
                         'feature_index': fi, 'feature_row': cid < int(feats[fi].shape[0]),
                         'parent_ids': parent_ids, 'parent_mapping_valid': level == 3 or len(parent_ids) == 1,
                         'union_sha256': sha_bytes((labels[level] == cid).astype(np.uint8))})
        out[str(level)] = rows
    return out


def gap_select(candidates, scores):
    if len(candidates) != len(scores): raise ValueError('candidate/score mismatch')
    ranked = sorted(zip(candidates, scores), key=lambda x: (-float(x[1]), tuple(x[0])))
    gaps = [0.0] + [ranked[i][1] - ranked[i + 1][1] for i in range(len(ranked) - 1)] if ranked else []
    cut = gaps.index(max(gaps)) if gaps else 0
    return [x[0] for x in ranked[:cut]]


def exact_root_candidates(root_scores, labels, filter_small=50, topk=5):
    """CPU equivalent of remote SemanticNAG._get_root_candidates (lines 22-32)."""
    root_ids = np.argsort(-np.asarray(root_scores, dtype=np.float32), kind='stable')
    root_ids = [int(x) for x in root_ids[:min(10, len(root_ids))]
                if int((labels[-1] == int(x)).sum()) > filter_small]
    return root_ids[:topk]


def union_record(selected, labels):
    mask = np.zeros(labels[0].shape[0], dtype=bool)
    for item in selected:
        level, candidate_id = int(item[0]), int(item[2])
        mask |= labels[level] == candidate_id
    return {'selected': [[int(x[0]), int(x[1]), int(x[2])] for x in selected],
            'gaussian_count': int(mask.sum()),
            'gaussian_union_sha256': sha_bytes(mask.astype(np.uint8)),
            'empty_output': bool(not mask.any())}


def render_selected_masks(args, evaluator, labels, records):
    """Render only materialized A/B unions; this function never opens GT."""
    if not args.source or not args.model:
        raise SystemExit('real mode rendering requires --source and --model')
    parser = argparse.ArgumentParser(add_help=False)
    lp = evaluator.ModelParams(parser)
    pp = evaluator.PipelineParams(parser)
    parser.add_argument('--path_pred', default='')
    parsed = parser.parse_args(['-s', str(args.source), '-m', str(args.model)])
    dataset, pipe = lp.extract(parsed), pp.extract(parsed)
    gaussians = evaluator.GaussianModel(dataset.sh_degree, 20)
    scene = evaluator.Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    cameras = {c.image_name: c for c in scene.getTrainCameras()}
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')
    root = Path(args.out)
    paths = {'A': root / 'A' / 'masks', 'B': root / 'B' / 'masks'}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=False)
    rendered = []
    with torch.no_grad():
        for rec in records:
            image = rec['image']
            if image not in cameras:
                raise SystemExit(f'NO_GO: query image missing from exact Scene cameras: {image}')
            cam = cameras[image]
            for arm in ('A', 'B'):
                mask = np.zeros(labels[0].shape[0], dtype=bool)
                for level, _, cid in rec[arm]['selected']:
                    mask |= labels[int(level)] == int(cid)
                gaussians._semantics = torch.from_numpy(mask).to('cuda').float().unsqueeze(1).expand(-1, 20).contiguous()
                sem = evaluator.render(cam, gaussians, pipe, bg)['semantics'][0]
                pred = (sem > 0.5).detach().cpu().numpy().astype(np.uint8)
                name = f"q{int(rec['query_index']):04d}_{Path(image).stem}.npy"
                target = paths[arm] / name
                np.save(target, pred)
                rendered.append({'query_index': rec['query_index'], 'arm': arm,
                                 'image': image, 'path': str(target),
                                 'shape': list(pred.shape), 'pixels': int(pred.sum()),
                                 'sha256': sha(target)})
    return rendered


def run_real(args):
    """Approved-only scorer; never reads GT and never overwrites output."""
    if not args.approve_real_run:
        raise SystemExit('FAIL_CLOSED: pass --approve-real-run for real mode')
    if Path(args.out).exists():
        raise SystemExit(f'Refusing existing output directory: {args.out}')
    if not args.evaluator:
        raise SystemExit('real mode requires --evaluator exact eval_lerf_relags.py')
    queries = load_query_manifest(args.queries)
    evaluator = load_exact_evaluator(args.evaluator)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    obj = torch.load(args.r0, map_location='cpu')
    labels = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in obj['nag']]
    snag = evaluator.SemanticNAG(obj['nag'], obj['nag_feat'])
    snag.feat = [f.cuda().float() for f in snag.feat]
    vlm = evaluator.ClipSimMeasure()
    vlm.load_model()
    table = candidate_table(labels, obj['nag_feat'])
    all_rows = [r for level in ('2', '3') for r in table[level]
                if r['feature_row'] and r['parent_mapping_valid']]
    if len(all_rows) != 646:
        raise SystemExit(f'NO_GO: B candidate count/mapping gate failed: {len(all_rows)}')
    records = []
    for qi, query in enumerate(queries):
        prompt = query['prompt']
        vlm.encode_text(prompt)
        sims = [vlm.compute_similarity(f.float()) for f in snag.feat]
        vals, ids = snag._get_root_candidates(sims, 50)
        roots = snag._build_root_candidates_with_leaf_info(
            ids, vals, sims, len(snag.labels)-2)[:5]
        pool_a = snag._analyze_leaf_candidates(roots, len(snag.labels)-2)
        selected_a = gap_select(pool_a, [float(x[3]) for x in pool_a])
        scores_b = {2: sims[1].detach().cpu().numpy(),
                    3: sims[2].detach().cpu().numpy()}
        pool_b = []
        for row in all_rows:
            level, cid = int(row['level']), int(row['candidate_id'])
            root_id = int(row['root_id']) if row['root_id'] is not None else cid
            pool_b.append((level, root_id, cid, float(scores_b[level][cid]),
                           int(row['gaussian_count'])))
        selected_b = gap_select(pool_b, [float(x[3]) for x in pool_b])
        rec = {'query_index': qi, 'image': query['image'], 'prompt': prompt,
               'A': {'pool_size': len(pool_a),
                     'pool_hash': json_hash(pool_a),
                     'real_text_score': True,
                     **union_record(selected_a, labels)},
               'B': {'pool_size': len(pool_b),
                     'pool_hash': json_hash(pool_b),
                     'real_text_score': True,
                     **union_record(selected_b, labels)}}
        records.append(rec)
    Path(args.out).mkdir(parents=True, exist_ok=False)
    rendered = render_selected_masks(args, evaluator, labels, records)
    input_paths = [Path(args.r0), Path(args.cameras), Path(args.queries),
                   Path(args.evaluator), Path(__file__)]
    input_paths += [Path(x) for x in (args.ply, args.labels, args.text_checkpoint) if x]
    manifest = {'status': 'REAL_PREDICTIONS_AND_MASKS', 'scene': 'figurines',
                'gt_opened': False, 'gpu_used': True,
                'same_r0_for_A_B': True, 'same_query_manifest_for_A_B': True,
                'selector': 'gap_select', 'query_count': len(queries),
                'B_candidate_count': len(all_rows),
                'render_manifest': {'A_mask_dir': str(Path(args.out) / 'A' / 'masks'),
                                   'B_mask_dir': str(Path(args.out) / 'B' / 'masks'),
                                   'records': rendered, 'gt_opened': False},
                'input_sha256': {str(p): sha(p) for p in input_paths},
                'source_sha256': {str(Path(__file__)): sha(Path(__file__))},
                'gt_evaluator': 'separate post-materialization command only'}
    (Path(args.out) / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (Path(args.out) / 'per_query.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (Path(args.out) / 'command.txt').write_text(' '.join(sys.argv) + '\n', encoding='utf-8')
    return manifest


def dry_run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    obj = torch.load(args.r0, map_location='cpu')
    labels = [x.detach().cpu().numpy().astype(np.int64, copy=False) for x in obj['nag']]
    feats = obj['nag_feat']
    table = candidate_table(labels, feats)
    invalid = [r for rows in table.values() for r in rows if not r['feature_row']]
    cameras = json.loads(Path(args.cameras).read_text(encoding='utf-8'))
    queries = load_query_manifest(args.queries) if args.queries else None
    all_nodes = [r for rows in table.values() for r in rows]
    parent_invalid = [r for r in all_nodes if not r['parent_mapping_valid']]
    manifest = {
        'status': 'DRY_RUN_SCHEMA_PASS' if not invalid else 'DRY_RUN_SCHEMA_FAIL',
        'gt_used': False, 'gpu_used': False, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scene': 'figurines', 'r0': str(args.r0), 'camera_count': len(cameras),
        'GT_opened': False,
        'query_manifest': ({'path': str(args.queries), 'count': len(queries),
                            'keys': ['image', 'prompt'], 'gt_fields_rejected': True}
                           if queries is not None else
                           {'status': 'DEFERRED_REQUIRED_FOR_REAL_RUN',
                            'keys': ['image', 'prompt'], 'GT_opened': False}),
        'camera_schema': sorted(cameras[0].keys()) if cameras else [],
        'label_schema': [{'level': i, 'shape': list(x.shape), 'min': int(x.min()), 'max': int(x.max()), 'unique': int(np.unique(x).size)} for i, x in enumerate(labels)],
        'feature_schema': [list(x.shape) for x in feats],
        'candidate_counts': {k: len(v) for k, v in table.items()},
        'invalid_feature_rows': invalid[:20], 'invalid_feature_row_count': len(invalid),
        'parent_mapping_invalid_count': len(parent_invalid),
        'B_union_mapping': {'nodes_checked': len(all_nodes), 'invalid_parent_count': len(parent_invalid), 'union_hashes_recorded': True, 'related_gaussians_contract': 'labels[level]==candidate_id'},
        'A': {'definition': 'same-R0 exact SemanticNAG._get_root_candidates then _build_root_candidates_with_leaf_info/_analyze_leaf_candidates', 'scores': 'DEFERRED_GPU_REAL_TEXT', 'selector': 'official_gap_select', 'root_candidate_contract': 'top-10 root scores, size>filter_small=50, then top-5; no proxy score supplied', 'candidate_union_parity': 'DEFERRED_REAL_TEXT_AND_SELECTOR'},
        'B': {'definition': 'same-R0 complete level-2/3 enumeration', 'scores': 'DEFERRED_GPU_REAL_TEXT', 'selector': 'official_gap_select'},
        'text_score_interface': 'exact evaluator ClipSimMeasure.encode_text(prompt); compute_similarity(nag_feat[level-1]); same evaluator/text checkpoint required for A/B',
        'real_score_contract': real_score_contract(),
        'old_author_pool_used': False, 'old_pool_input': None, 'proxy_scores_used': False,
        'input_sha256': {str(p): sha(p) for p in [Path(args.r0), Path(args.cameras)]},
        'source_sha256': {str(Path(__file__)): sha(Path(__file__))},
        'mapping_contract': 'level=tuple[0], candidate_id=tuple[2], gaussian_mask=labels[level]==candidate_id, feature_index=level-1',
        'selector_parity': 'DEFERRED_UNTIL_REAL_TEXT_SCORE_RUN',
        'next_gate': 'Codex review required before any GPU/text/evaluator execution',
    }
    Path(args.out).mkdir(parents=True, exist_ok=False)
    (Path(args.out) / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (Path(args.out) / 'candidate_table.json').write_text(json.dumps(table, indent=2), encoding='utf-8')
    (Path(args.out) / 'command.txt').write_text(' '.join(sys.argv) + '\n', encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--r0', required=True); p.add_argument('--cameras', required=True)
    p.add_argument('--out', required=True); p.add_argument('--queries')
    p.add_argument('--evaluator'); p.add_argument('--ply'); p.add_argument('--labels')
    p.add_argument('--source'); p.add_argument('--model')
    p.add_argument('--text-checkpoint'); p.add_argument('--real-run', action='store_true')
    p.add_argument('--approve-real-run', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.real_run:
        result = run_real(a)
    elif not a.dry_run:
        raise SystemExit('FAIL_CLOSED: use --dry-run or explicit --real-run --approve-real-run')
    else:
        result = dry_run(a)
    print(json.dumps({k: result[k] for k in ('status','camera_count','candidate_counts','invalid_feature_row_count','gpu_used','gt_used')}, ensure_ascii=False))


if __name__ == '__main__': main()
