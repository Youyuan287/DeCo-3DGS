"""Render only two missing cells from immutable cached scores; GT evaluated last."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rank_count_ops import build_cells
from run_ovs_support import sha,jhash,dump

MODES=('rank1_count0','rank0_count1')
R=Path('/root/rivermind-data')
RA=R/'ReLaGS_Reproduction'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise SystemExit('Refusing existing output')
    for path,reserve in [('/',5),(R,10)]:
        if shutil.disk_usage(path).free<reserve*2**30:raise SystemExit('Disk reserve violated')
    prior=json.loads((a.prior/'manifest.json').read_text())
    records=json.loads((a.prior/'query_diagnostics.json').read_text())
    assert prior['dataset']=='LERF-OVS' and prior['scene']=='figurines' and not prior['sanity']
    # Assert every old source/model/cache is identical before consuming old scores.
    for name,digest in prior['input_sha256'].items():
        assert sha(Path(name))==digest, 'Changed input: '+name
    assert len(records)==21 and len({r['prompt'] for r in records})==21
    for r in records:
        assert jhash(r['pool'])==r['pool_sha256']
        build_cells(r)
    a.out.mkdir(parents=True)
    start=time.time()
    manifest=dict(dataset='LERF-OVS',scene='figurines',baseline_origin=prior['baseline_origin'],
                  prior=str(a.prior),query_by_image=prior['query_by_image'],
                  old_inputs_verified=True,threshold=prior['threshold'],new_modes=list(MODES),
                  scores_recomputed=False,baseline_gpu_rerun=False,
                  input_sha256={str(a.prior/f):sha(a.prior/f) for f in ('manifest.json','query_diagnostics.json','per_query.csv')},
                  source_sha256={str(HERE/f):sha(HERE/f) for f in ('run_rank_count.py','rank_count_ops.py')},
                  environment={k:os.environ.get(k,'') for k in ('CUDA_VISIBLE_DEVICES','RELAGS_REPO','PYTHONPATH','HF_HUB_OFFLINE')})
    dump(a.out/'manifest.json',manifest)
    (a.out/'command.txt').write_text(' '.join(sys.argv)+'\n')
    import cv2
    import numpy as np
    import torch
    torch.manual_seed(0);np.random.seed(0)
    spec=importlib.util.spec_from_file_location('original_ovs_evaluator',RA/'scripts/eval_lerf_relags.py')
    ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
    model=RA/'ref/relags_hf/figurines';source=RA/'ref/hf_only/figurines'
    gt_dir=RA/'ref/hf_lerf/lerf_ovs/label/figurines'
    p=argparse.ArgumentParser();lp,pp=ev.ModelParams(p),ev.PipelineParams(p)
    p.add_argument('--path_pred',default='')
    args=p.parse_args(['-s',str(source),'-m',str(model)])
    dataset,pipe=lp.extract(args),pp.extract(args)
    gs=ev.GaussianModel(dataset.sh_degree,20)
    scene=ev.Scene(dataset,gs,0,load_sem=False,load_test=False)
    bg=torch.tensor([1,1,1] if dataset.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    cams={c.image_name:c for c in scene.getTrainCameras()}
    nag=torch.load(model/'sai_nag.pt',map_location='cpu')
    snag=ev.SemanticNAG(nag['nag'],nag['nag_feat'])
    assert gs.get_xyz.shape[0]==snag.gaussian_num
    query_by_image=prior['query_by_image']
    assert all(im in cams for im in query_by_image)
    predictions={}; selections={}; diagnostics=[]; checked_old_unions=0; reused_checks=0
    with torch.no_grad():
        for qi,record in enumerate(records):
            prompt=record['prompt'];cells=build_cells(record);outrec=dict(prompt=prompt,pool_sha256=record['pool_sha256'],modes={})
            rels={};unions={}
            for mode,chosen in cells.items():
                restored=[(n[0],n[1],n[2],n[3],None,None,n[4]) for n in chosen]
                rel=snag._get_related_gaussians(restored)
                union=rel[:,0].numpy().astype(np.uint8)
                digest=hashlib.sha256(union.tobytes()).hexdigest()
                if mode in ('baseline','spatial'):
                    assert digest==record['modes'][mode]['gaussian_union_sha256'], 'Prior Gaussian union mismatch'
                    assert int(union.sum())==record['modes'][mode]['gaussian_count']
                    checked_old_unions+=1
                rels[mode]=rel;unions[mode]=digest
                if mode in MODES:
                    outrec['modes'][mode]=dict(selected=chosen,gaussian_union_sha256=digest,
                                              gaussian_count=int(union.sum()),pool_sha256=record['pool_sha256'])
            for mode in MODES:
                selections[(mode,prompt)]=outrec['modes'][mode]
                gs._semantics=rels[mode].expand(-1,20).cuda()
                for image,qs in query_by_image.items():
                    if prompt not in qs:continue
                    sem=ev.render(cams[image],gs,pipe,bg)['semantics'][0]
                    pred=(sem>prior['threshold']).cpu().numpy().astype(np.uint8)
                    safe=prompt.replace(' ','_').replace('/','_')+'.png'
                    for old in ('baseline','spatial'):
                        if unions[mode]==unions[old]:
                            oldmask=cv2.imread(str(a.prior/old/'masks'/image/safe),cv2.IMREAD_GRAYSCALE)
                            assert oldmask is not None and np.array_equal(pred,oldmask>0), 'Identical union render replay mismatch'
                            reused_checks+=1
                    d=a.out/mode/'masks'/image;d.mkdir(parents=True,exist_ok=True)
                    assert cv2.imwrite(str(d/safe),pred*255)
                    predictions[(mode,image,prompt)]=pred
            diagnostics.append(outrec)
            dump(a.out/'query_diagnostics.json',diagnostics)
            print(f'QUERY_DONE {qi+1}/{len(records)} {prompt} k0={len(cells["baseline"])} k1={len(cells["spatial"])}',flush=True)
        # Only after all predictions are frozen do polygon annotations enter evaluation.
        rows=[]
        for image,qs in query_by_image.items():
            anno=json.loads((gt_dir/(image+'.json')).read_text())
            h,w=cams[image].image_height,cams[image].image_width
            for prompt in qs:
                gt=np.zeros((h,w),np.uint8)
                for obj in anno['objects']:
                    if obj['category']==prompt:gt=np.maximum(gt,ev.polygon_to_mask((h,w),obj['segmentation']))
                for mode in MODES:
                    pred=predictions[(mode,image,prompt)];sel=selections[(mode,prompt)]
                    union=np.logical_or(pred,gt).sum()
                    assert union>0
                    rows.append(dict(mode=mode,scene='figurines',image=image,prompt=prompt,
                                     iou=float(np.logical_and(pred,gt).sum()/union),pred_px=int(pred.sum()),gt_px=int(gt.sum()),
                                     n_candidates=len(sel['selected']),pool_sha256=sel['pool_sha256'],
                                     gaussian_union_sha256=sel['gaussian_union_sha256']))
        with (a.out/'per_query.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
        summary=[]
        for mode in MODES:
            rr=[x for x in rows if x['mode']==mode];assert len(rr)==56
            means=[np.mean([x['iou'] for x in rr if x['image']==im]) for im in query_by_image]
            summary.append(dict(mode=mode,miou=float(np.mean(means)),flat_miou=float(np.mean([x['iou'] for x in rr])),n_queries=len(rr)))
        status=dict(status='complete',dataset='LERF-OVS',scene='figurines',summary=summary,
                    old_union_checks=checked_old_unions,identical_union_mask_checks=reused_checks,
                    wall_seconds=time.time()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
        assert checked_old_unions==42 and reused_checks>0
        dump(a.out/'FINAL_STATUS.json',status)
        print('FINAL_RESULT '+json.dumps(status),flush=True)


if __name__=='__main__':main()
