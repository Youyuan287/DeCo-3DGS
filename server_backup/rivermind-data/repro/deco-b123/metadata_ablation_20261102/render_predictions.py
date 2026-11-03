"""精确渲染冻结模型的两列贡献置零对照；不训练或改变预测。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/rivermind-data/repro/deco-b123')
REPO = ROOT / 'c1_b123_run_20261028/DeCo-3DGS'
PREVIOUS = ROOT / 'pairwise_semantic_20261101'
sys.path[:0] = [str(REPO), str(REPO / 'deco'), str(REPO / 'deco/lerf_ovs')]
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from eval_lerf_ovs_v2 import polygon_to_mask

ARMS = ('A', 'S', 'J', 'S0', 'J0')
SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')
EXPECTED_VIEWS = dict(zip(SCENES, (56, 71, 59, 22)))
EXPECTED_PROMPTS = dict(zip(SCENES, (21, 14, 14, 18)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_predictions(path):
    predictions = {}
    with path.open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            row = json.loads(text)
            key = row['scene'], row['prompt']
            require(key not in predictions, f'duplicate prediction {key} at {path}:{line}')
            require(row['scene'] in SCENES and row['heldout_scene'] == row['scene'],
                    f'invalid heldout scene {key}')
            require(sorted(row['train_scenes']) == sorted(set(SCENES) - {row['scene']}),
                    f'non-LOSO prediction {key}')
            predictions[key] = row
    require(len(predictions) == 67, f'expected 67 OOF predictions, got {len(predictions)}')
    for scene, count in EXPECTED_PROMPTS.items():
        require(sum(key[0] == scene for key in predictions) == count,
                f'incomplete prediction scene {scene}')
    return predictions


def main():
    ap = argparse.ArgumentParser()
    for name in ('scene', 'source', 'model', 'gt_root', 'out', 'predictions'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--limit-queries', type=int, default=0)
    args = ap.parse_args()
    require(args.scene in SCENES, f'unknown scene {args.scene}')
    require(args.limit_queries >= 0, 'limit-queries must be nonnegative')
    predictions = read_predictions(Path(args.predictions))
    original_predictions = read_predictions(PREVIOUS / 'predictions.jsonl')
    with (PREVIOUS / 'results' / f'{args.scene}_pairwise.csv').open(newline='') as handle:
        refs = list(csv.DictReader(handle))
    require(len(refs) == EXPECTED_VIEWS[args.scene], 'incomplete previous render CSV')
    ref_keys = [(row['image'], row['prompt']) for row in refs]
    require(len(set(ref_keys)) == len(ref_keys), 'duplicate previous query-view')
    require(all(row['scene'] == args.scene for row in refs), 'previous render scene mismatch')
    require({row['prompt'] for row in refs} == {q for scene, q in predictions if scene == args.scene},
            'prediction/render prompt coverage differs')

    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs, _ = p.parse_known_args(['-s', args.source, '-m', args.model])
    dataset, pipe = lp.extract(pargs), pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 0, load_sem=False, load_test=False)
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device='cuda')
    nag = torch.load(Path(args.model) / 'sai_nag.pt', map_location='cpu', weights_only=True)
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = ['scene', 'image', 'prompt', 'family', 'g_A', 'g_S', 'g_J', 'g_S0', 'g_J0',
               'A_iou', 'S_iou', 'J_iou', 'S0_iou', 'J0_iou',
               'baseline_error', 'old_S_error', 'old_J_error']
    # Query-independent image/set cache shared by all arms and prompts. Stored in RAM.
    mask_cache = {}
    with (out / f'{args.scene}_metadata.csv').open('w', newline='') as handle, torch.no_grad():
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for qi, ref in enumerate(refs):
            if args.limit_queries and qi >= args.limit_queries:
                break
            image, prompt = ref['image'], ref['prompt']
            pred = predictions[(args.scene, prompt)]
            previous = original_predictions[(args.scene, prompt)]
            candidates = [tuple(key) for key in pred['candidates']]
            require(candidates == [tuple(key) for key in previous['candidates']],
                    f'candidate IDs/order changed: {prompt}')
            candidate_set = set(candidates)
            require(len(candidate_set) == len(candidates), f'duplicate candidate: {prompt}')
            selections = {name: [tuple(key) for key in pred['selections'][name]] for name in ARMS}
            g = len(selections['A'])
            require(int(pred['g_A']) == g == int(previous['g_A']) == int(float(ref['g_A'])),
                    f'g_A differs: {prompt}')
            for name, selected in selections.items():
                require(len(selected) == g and len(set(selected)) == g and set(selected) <= candidate_set,
                        f'fixed-count/pool violation: {prompt}/{name}')
            for name in ('A', 'S', 'J'):
                require(set(selections[name]) == {tuple(key) for key in previous['selections'][name]},
                        f'original {name} selection changed: {prompt}')
            for name in ('S', 'J', 'S0', 'J0'):
                scores = np.asarray(pred['scores'][name], dtype=float)
                require(scores.shape == (len(candidates),) and np.isfinite(scores).all(),
                        f'invalid score vector: {prompt}/{name}')
                order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
                require(selections[name] == [candidates[i] for i in order[:g]],
                        f'score/selection mismatch: {prompt}/{name}')

            camera = cameras[image]
            height, width = camera.image_height, camera.image_width
            masks = {}
            for name, selected in selections.items():
                key = image, tuple(sorted(selected))
                if key not in mask_cache:
                    cands = [(level, -1, cid, 0., None, None, 0) for level, cid in selected]
                    support = (snag._get_related_gaussians(cands) if cands else
                               torch.zeros(snag.gaussian_num, 1))
                    gaussians._semantics = support.expand(-1, 20).cuda()
                    sem = render(camera, gaussians, pipe, bg)['semantics']
                    mask = (sem.reshape(sem.shape[0], -1)[0] > .5).cpu().numpy()
                    mask_cache[key] = np.packbits(mask)
                masks[name] = np.unpackbits(mask_cache[key], count=height * width).reshape(height, width).astype(bool)

            # All predictions/selections and their renders are fixed before current GT is read.
            annotation = json.load((Path(args.gt_root) / args.scene / f'{image}.json').open())
            gt = np.zeros((height, width), np.uint8)
            for obj in annotation['objects']:
                if obj['category'] == prompt:
                    gt = np.maximum(gt, polygon_to_mask((height, width), obj['segmentation']))
            gt = gt.astype(bool)
            values = {}
            for name, mask in masks.items():
                union = np.logical_or(mask, gt).sum()
                values[name] = float(np.logical_and(mask, gt).sum() / union) if union else float('nan')
            errors = {name: abs(values[name] - float(ref[name + '_iou'])) for name in ('A', 'S', 'J')}
            require(all(np.isfinite(value) for value in values.values()) and
                    all(np.isfinite(error) and error <= 1e-6 for error in errors.values()),
                    f'BASELINE_OR_FINITE_FAILURE {image} {prompt}: {errors}')
            writer.writerow(dict(scene=args.scene, image=image, prompt=prompt, family=ref['family'],
                **{'g_' + name: len(selections[name]) for name in ARMS},
                **{name + '_iou': value for name, value in values.items()},
                baseline_error=errors['A'], old_S_error=errors['S'], old_J_error=errors['J']))
            handle.flush()
            print('QUERY', qi + 1, image, prompt, values, 'reproduction_errors', errors, flush=True)
    print('SCENE_DONE', args.scene, 'mask_cache_entries', len(mask_cache), flush=True)


if __name__ == '__main__':
    main()
