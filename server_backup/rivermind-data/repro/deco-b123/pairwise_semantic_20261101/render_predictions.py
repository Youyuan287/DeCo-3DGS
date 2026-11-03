"""Evaluate frozen out-of-fold rankings by the original exact renderer."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/rivermind-data/repro/deco-b123')
REPO = ROOT/'c1_b123_run_20261028/DeCo-3DGS'
sys.path[:0] = [str(REPO),str(REPO/'deco'),str(REPO/'deco/lerf_ovs')]
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from nag_data import SemanticNAG
from scene import GaussianModel, Scene
from eval_lerf_ovs_v2 import polygon_to_mask

def main():
    ap = argparse.ArgumentParser()
    for name in ('scene','source','model','gt_root','out','predictions'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--limit-queries',type=int,default=0)
    args = ap.parse_args()
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
    predictions = {}
    for line in Path(args.predictions).read_text().splitlines():
        row = json.loads(line)
        if row['scene'] == args.scene:
            assert row['heldout_scene'] == args.scene
            assert args.scene not in row['train_scenes']
            predictions[row['prompt']] = row
    old = {}
    for line in (ROOT/'score_readout_20261101/results'/f'{args.scene}_details.jsonl').read_text().splitlines():
        row = json.loads(line)
        old[(row['image'],row['prompt'])] = row
    refs = list(csv.DictReader((ROOT/'score_readout_20261101/results'/f'{args.scene}_score.csv').open()))
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    columns = ['scene','image','prompt','family','g_A','g_S','g_J','g_Q',
               'A_iou','S_iou','J_iou','Q_iou','baseline_error']
    with (out/f'{args.scene}_pairwise.csv').open('w',newline='') as f,torch.no_grad():
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for qi,ref in enumerate(refs):
            if args.limit_queries and qi >= args.limit_queries:
                break
            image,prompt = ref['image'],ref['prompt']
            pred = predictions[prompt]
            before = old[(image,prompt)]
            candidates = {tuple(k) for k in before['candidates']}
            assert {tuple(k) for k in pred['candidates']} == candidates
            selections = {name:[tuple(k) for k in pred['selections'][name]] for name in 'ASJQ'}
            assert set(selections['A']) == {tuple(k) for k in before['selections']['A']}
            g = len(selections['A'])
            assert int(pred['g_A']) == g
            for name,selected in selections.items():
                assert len(selected)==g and len(set(selected))==g and set(selected)<=candidates
            camera = cameras[image]
            h,w = camera.image_height,camera.image_width
            rendered, masks = {},{}
            for name,selected in selections.items():
                key = tuple(sorted(selected))
                if key not in rendered:
                    cands = [(l,-1,c,0.,None,None,0) for l,c in selected]
                    support = snag._get_related_gaussians(cands) if cands else torch.zeros(snag.gaussian_num,1)
                    gaussians._semantics = support.expand(-1,20).cuda()
                    sem = render(camera,gaussians,pipe,bg)['semantics']
                    rendered[key] = (sem.reshape(sem.shape[0],-1)[0]>.5).reshape(h,w).cpu().numpy()
                masks[name] = rendered[key]
            # Out-of-fold scores/selections are fixed before current GT is read.
            annotation = json.load((Path(args.gt_root)/args.scene/f'{image}.json').open())
            gt = np.zeros((h,w),np.uint8)
            for obj in annotation['objects']:
                if obj['category']==prompt:
                    gt = np.maximum(gt,polygon_to_mask((h,w),obj['segmentation']))
            gt = gt.astype(bool)
            vals = {}
            for name,mask in masks.items():
                union = np.logical_or(mask,gt).sum()
                vals[name] = float(np.logical_and(mask,gt).sum()/union) if union else float('nan')
            error = abs(vals['A']-float(ref['A_iou']))
            if not all(np.isfinite(v) for v in vals.values()) or not np.isfinite(error) or error>1e-6:
                raise RuntimeError(f'BASELINE_OR_FINITE_FAILURE {image} {prompt}: {error}')
            writer.writerow(dict(scene=args.scene,image=image,prompt=prompt,family=ref['family'],
                **{'g_'+name:len(selections[name]) for name in 'ASJQ'},
                **{name+'_iou':value for name,value in vals.items()},baseline_error=error))
            f.flush()
            print('QUERY',qi+1,image,prompt,vals,flush=True)
    print('SCENE_DONE',args.scene,flush=True)

if __name__=='__main__':
    main()
