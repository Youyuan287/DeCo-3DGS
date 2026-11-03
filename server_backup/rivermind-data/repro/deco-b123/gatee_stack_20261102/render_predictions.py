"""Exact fixed-branch stack evaluation; GT never selects a branch."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/rivermind-data/repro/deco-b123')
REPO = ROOT / 'c1_b123_run_20261028/DeCo-3DGS'
OLD = ROOT / 'metadata_ablation_20261102'
sys.path[:0] = [str(REPO), str(REPO/'deco'), str(REPO/'deco/lerf_ovs')]
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from eval_lerf_ovs_v2 import polygon_to_mask

ARMS = ('A', 'E', 'J', 'J0', 'EJ', 'EJ0')
COUNTS = dict(zip(('figurines','ramen','teatime','waldo_kitchen'), (56,71,59,22)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser()
    for name in ('scene','source','model','gt_root','out','predictions'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--limit-queries', type=int, default=0)
    args = ap.parse_args()
    require(args.scene in COUNTS and args.limit_queries >= 0, 'invalid scene/limit')
    preds = {}
    for line in Path(args.predictions).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        key = row['scene'], row['image'], row['prompt']
        require(key not in preds, 'duplicate predicted query-view')
        preds[key] = row
    require(len(preds)==208, 'expected 208 predicted query-views')
    require(all(sum(k[0]==s for k in preds)==n for s,n in COUNTS.items()), 'scene count mismatch')
    old_preds = {(r['scene'],r['prompt']):r for r in map(json.loads,
                 (OLD/'predictions.jsonl').read_text().splitlines())}
    refs = list(csv.DictReader((OLD/'results'/f'{args.scene}_metadata.csv').open()))
    anchors = {(r['image'],r['prompt']):r for r in csv.DictReader(
        (ROOT/'gatee_20261031'/f'{args.scene}_cmaskgate_image.csv').open())}
    require(len(refs)==len(anchors)==COUNTS[args.scene], 'reference count mismatch')
    require({(args.scene,r['image'],r['prompt']) for r in refs} ==
            {k for k in preds if k[0]==args.scene}, 'reference/prediction key mismatch')
    p = argparse.ArgumentParser()
    lp, pp = ModelParams(p), PipelineParams(p)
    pargs,_ = p.parse_known_args(['-s',args.source,'-m',args.model])
    dataset,pipe = lp.extract(pargs),pp.extract(pargs)
    gaussians = GaussianModel(dataset.sh_degree,20)
    scene = Scene(dataset,gaussians,0,load_sem=False,load_test=False)
    bg = torch.tensor([1,1,1] if dataset.white_background else [0,0,0],
                      dtype=torch.float32,device='cuda')
    nag = torch.load(Path(args.model)/'sai_nag.pt',map_location='cpu',weights_only=True)
    snag = SemanticNAG(nag['nag'],nag['nag_feat'])
    cameras = {c.image_name:c for c in scene.getTrainCameras()}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    fields = ['scene','image','prompt','family'] + ['g_'+a for a in ARMS] + \
        [a+'_iou' for a in ARMS] + ['route_EJ','route_EJ0','baseline_error','old_E_error','old_J_error','old_J0_error']
    cache = {}
    with (out/f'{args.scene}_stack.csv').open('w',newline='') as handle,torch.no_grad():
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for qi,ref in enumerate(refs):
            if args.limit_queries and qi >= args.limit_queries:
                break
            image,prompt = ref['image'],ref['prompt']
            pred = preds[(args.scene,image,prompt)]
            previous = old_preds[(args.scene,prompt)]
            anchor = anchors[(image,prompt)]
            candidates = [tuple(k) for k in pred['candidates']]
            require(candidates==[tuple(k) for k in previous['candidates']], 'candidate pool/order changed')
            pool = set(candidates)
            selections = {a:[tuple(k) for k in pred['selections'][a]] for a in ARMS}
            require(all(len(v)==len(set(v)) and set(v)<=pool for v in selections.values()), 'invalid selected set')
            g = int(pred['g_A'])
            require(g==int(previous['g_A'])==int(float(ref['g_A'])), 'original cardinality differs')
            for a in ('A','J','J0'):
                require(selections[a]==[tuple(k) for k in previous['selections'][a]] and len(selections[a])==g,
                        'old selection differs '+a)
            require(len(selections['E'])==int(anchor['gatee_len']), 'E cardinality differs')
            for branch,parent in (('EJ','J'),('EJ0','J0')):
                flag = pred['route_'+branch]
                require(flag in (0,1,False,True), 'invalid route flag')
                expected = selections[parent] if flag else selections['E']
                require(selections[branch]==expected, 'stack differs from routed parent')
            camera = cameras[image]
            height,width = camera.image_height,camera.image_width
            masks = {}
            for a,selected in selections.items():
                key = image,tuple(sorted(selected))
                if key not in cache:
                    cands = [(l,-1,c,0.,None,None,0) for l,c in selected]
                    support = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num,1)
                    gaussians._semantics = support.expand(-1,20).cuda()
                    sem = render(camera,gaussians,pipe,bg)['semantics']
                    cache[key] = np.packbits((sem.reshape(sem.shape[0],-1)[0]>.5).cpu().numpy())
                masks[a] = np.unpackbits(cache[key],count=height*width).reshape(height,width).astype(bool)
            annotation = json.load((Path(args.gt_root)/args.scene/f'{image}.json').open())
            gt = np.zeros((height,width),np.uint8)
            for obj in annotation['objects']:
                if obj['category']==prompt:
                    gt = np.maximum(gt,polygon_to_mask((height,width),obj['segmentation']))
            gt = gt.astype(bool)
            values = {}
            for a,mask in masks.items():
                union = np.logical_or(mask,gt).sum()
                values[a] = float(np.logical_and(mask,gt).sum()/union) if union else float('nan')
            errors = {'baseline_error':abs(values['A']-float(ref['A_iou'])),
                      'old_E_error':abs(values['E']-float(anchor['gatee_iou'])),
                      'old_J_error':abs(values['J']-float(ref['J_iou'])),
                      'old_J0_error':abs(values['J0']-float(ref['J0_iou']))}
            require(all(np.isfinite(v) for v in values.values()) and
                    all(np.isfinite(e) and e<=1e-6 for e in errors.values()),
                    f'BASELINE_OR_FINITE_FAILURE {image} {prompt}: {errors}')
            writer.writerow(dict(scene=args.scene,image=image,prompt=prompt,family=ref['family'],
                **{'g_'+a:len(v) for a,v in selections.items()},
                **{a+'_iou':v for a,v in values.items()},route_EJ=int(pred['route_EJ']),
                route_EJ0=int(pred['route_EJ0']),**errors))
            handle.flush()
            print('QUERY',qi+1,image,prompt,values,'errors',errors,flush=True)
    print('SCENE_DONE',args.scene,'cache_entries',len(cache),flush=True)


if __name__=='__main__':
    main()
