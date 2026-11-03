"""New matched observation bank on immutable R0 final NAG; no region growing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from capture_contracts import validate_features,validate_labels,check_budget,geometry_groups,audit_calls,sha

BASE=Path('/root/rivermind-data')
REPO=BASE/'ReLaGS_Reproduction/ReLaGS'
R0=BASE/'ReLaGS_Reproduction/runs/readout_20261024/figurines/model_r0'
SOURCE=BASE/'VALA/dataset/3dgs/lerf_ovs/figurines'
OUT=BASE/'ARIS-ReLaGS/runs/view_consensus_capture_20261025'
EXPECTED='8597e7fe4c33165859b638ebd3a2981bd7953a4c93d4f5e87cc10a4777c4ccef'

def dump(name,value):
    with (OUT/name).open('x') as f: json.dump(value,f,indent=2)

def preflight():
    import torch
    audit_calls(Path(__file__).read_text())
    assert sha(R0/'sai_nag_3.0.pt')==EXPECTED
    nag=torch.load(R0/'sai_nag_3.0.pt',map_location='cpu')
    n=len(nag['nag'][0]);ks=[len(f) for f in nag['nag_feat']]
    for labels,k in zip(nag['nag'][1:],ks): validate_labels(labels.numpy(),k,n)
    cameras=sorted(json.loads((R0/'cameras.json').read_text()),key=lambda c:c['img_name'])
    assert len(cameras)==299 and len({c['img_name'] for c in cameras})==299
    inputs={str(R0/'sai_nag_3.0.pt'):EXPECTED,str(R0/'cameras.json'):sha(R0/'cameras.json'),
            str(R0/'point_cloud/iteration_0/point_cloud.ply'):sha(R0/'point_cloud/iteration_0/point_cloud.ply')}
    for i,c in enumerate(cameras):
        path=SOURCE/'language_features'
        sp=path/(c['img_name']+'_s.npy');fp=path/(c['img_name']+'_f.npy')
        s=np.load(sp,mmap_mode='r');f=np.load(fp,mmap_mode='r');validate_features(s,f)
        assert s.shape[0]>=4
        for level in (1,2,3):
            valid=s[level][s[level]>=0]
            assert len(valid)>0
        inputs[str(sp)]=sha(sp);inputs[str(fp)]=sha(fp)
        if i%50==0: print('PREFLIGHT',i,flush=True)
    planned=4*299*sum(ks)*512 + 299*sum(ks)*5 + 60*2**20
    check_budget(shutil.disk_usage(BASE).free,planned)
    assert shutil.disk_usage('/').free>=5*2**30
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'preflight.json').exists()
    groups=geometry_groups(cameras)
    dump('geometry_groups.json',groups)
    dump('cameras_frozen.json',cameras)
    dump('preflight.json',{'status':'pass','dataset':'LERF-OVS','scene':'figurines','baseline':'new mapping on frozen R0 NAG, not R0 replay',
       'gaussians':n,'k':ks,'views':299,'planned_bytes':planned,'inputs':inputs,'tau':3.,
       'source_order':'image_name ascending; original historical shuffle unavailable; frozen identically for all new arms',
       'seed_scheme':'20261025+1000*level+sorted_view_index','groups_sha256':sha(OUT/'geometry_groups.json'),
       'source_hashes':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('capture_contracts.py'),REPO/'merge_proj.py',REPO/'scene/__init__.py',REPO/'scene/dataset_readers.py']}})
    print('PREFLIGHT_PASS',planned,groups['sizes'],flush=True)

def capture():
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(REPO))
    from merge_proj import SAI3D
    from scene import Scene,GaussianModel
    from arguments import ModelParams
    from gaussian_renderer import render_point
    meta=json.loads((OUT/'preflight.json').read_text())
    assert sha(OUT/'geometry_groups.json')==meta['groups_sha256']
    for path,digest in meta['inputs'].items(): assert sha(path)==digest,'changed input '+path
    parser=argparse.ArgumentParser();mp=ModelParams(parser)
    args=mp.extract(parser.parse_args(['-s',str(SOURCE),'-m',str(R0)]))
    gs=GaussianModel(3,0)
    scene=Scene(args,gs,load_iteration=0,shuffle=False,load_img=False,load_sem=True,load_test=False,vlm_type='clip')
    views=sorted(scene.getTrainCameras(),key=lambda c:c.image_name)
    cams=json.loads((OUT/'cameras_frozen.json').read_text())
    assert [v.image_name for v in views]==[c['img_name'] for c in cams]
    assert len(gs.get_xyz)==meta['gaussians']
    # Validate actual camera pose against frozen JSON before projections.
    for v,c in zip(views,cams):
        pose=np.eye(4);pose[:3,:3]=v.R.T;pose[:3,3]=v.T;pose=np.linalg.inv(pose)
        assert np.allclose(pose[:3,3],c['position'],atol=1e-5)
        assert np.allclose(pose[:3,:3],c['rotation'],atol=1e-5)
    shell=argparse.Namespace(max_neighbor_distance=2,view_freq=5,dis_decay=.5,tau=3.)
    agent=SAI3D(shell)
    model=torch.load(R0/'sai_nag_3.0.pt',map_location='cpu')
    source_records=[];bank_records=[]
    start=time.monotonic()
    with torch.no_grad():
        for level,(labels,k) in enumerate(zip(model['nag'][1:],meta['k']),1):
            check_budget(shutil.disk_usage(BASE).free,4*len(views)*k*512+len(views)*k*5+20*2**20)
            bankpath=OUT/f'bank_l{level}.npy'
            if bankpath.exists(): raise RuntimeError('existing bank; no overwrite')
            bank=np.lib.format.open_memmap(bankpath,mode='w+',dtype=np.float32,shape=(len(views),k,512))
            vis=np.zeros((len(views),k),np.float32)
            gau2sp=labels.cuda().long();counts=torch.bincount(gau2sp,minlength=k)
            for vi,view in enumerate(views):
                t=time.monotonic();seed=20261025+1000*level+vi
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                mapped=agent.extract_gaussian_features(gs,[view],level).T[0]
                sem=view.semantic['sem'].cuda().float();seg=view.semantic['seg_map'][level]
                foreground=view.semantic['fg_mask'][level];low=seg[foreground].min().item();high=seg[foreground].max().item()
                ratios=agent._get_superpoint_mask_ratio(gau2sp,mapped,high-low+2)
                assert ratios.shape==(k,high-low+1),'ratio/semantic row alignment failure'
                ratios[ratios<.3]=0
                features=ratios@sem[low:high+1]
                weight=render_point(view,gs,agent.pipeline,agent.background)['weight']
                visible=weight>.0001
                fraction=torch.bincount(gau2sp[visible],minlength=k)/counts
                x=F.normalize(features,p=2,dim=-1)*fraction[:,None]
                bank[vi]=x.cpu().numpy();vis[vi]=fraction.cpu().numpy()
                source_records.append({'level':level,'view_index':vi,'view_id':view.image_name,'seed':seed,
                  'camera_sha256':__import__('hashlib').sha256(json.dumps(cams[vi],sort_keys=True).encode()).hexdigest(),
                  'source_identity':f'{level}:{view.image_name}:{seed}',
                  'mapping_sha256':__import__('hashlib').sha256(mapped.cpu().numpy().tobytes()).hexdigest()})
                if level==1 and vi==0:
                    torch.cuda.synchronize()
                    sanity={'seconds':time.monotonic()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                            'finite':bool(torch.isfinite(x).all()),'nonzero_nodes':int(x.any(dim=1).sum())}
                    dump('first_view_sanity.json',sanity)
                    assert sanity['finite'] and sanity['nonzero_nodes']>0
                    print('FIRST_VIEW_PASS',json.dumps(sanity),flush=True)
                if vi%50==0: print('CAPTURE',level,vi,'elapsed',time.monotonic()-start,flush=True)
                if time.monotonic()-start>900: raise RuntimeError('15 minute capture budget exceeded')
            bank.flush();valid=np.any(bank!=0,axis=2)
            np.save(OUT/f'valid_l{level}.npy',valid);np.save(OUT/f'visibility_l{level}.npy',vis)
            bank_records.append({'level':level,'path':str(bankpath),'sha256':sha(bankpath),'valid_observations':int(valid.sum())})
            del bank
            if shutil.disk_usage(BASE).free<10*2**30: raise RuntimeError('reserve violation')
    assert sha(R0/'sai_nag_3.0.pt')==EXPECTED
    dump('source_identities.json',source_records)
    dump('capture_done.json',{'banks':bank_records,'seconds':time.monotonic()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','capture']);a=p.parse_args()
    (preflight if a.mode=='preflight' else capture)()
