#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch

def poly_mask(shape, pts):
    m=np.zeros(shape,np.uint8); cv2.fillPoly(m,[np.asarray(pts,np.int32)],1); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scene',required=True); ap.add_argument('--source',required=True); ap.add_argument('--model',required=True); ap.add_argument('--nag',required=True); ap.add_argument('--gt_root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--chunk',type=int,default=20); a=ap.parse_args()
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from nag_data import SemanticNAG
    from scene import GaussianModel, Scene
    root=Path(a.out); maskdir=root/'root_region_masks'; maskdir.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser(); lp=ModelParams(p); pp=PipelineParams(p); pargs,_=p.parse_known_args(['-s',a.source,'-m',a.model]); ds=lp.extract(pargs); pipe=pp.extract(pargs)
    gauss=GaussianModel(ds.sh_degree,20); scene=Scene(ds,gauss,0,load_sem=False,load_test=False)
    bg=torch.tensor([1,1,1] if ds.white_background else [0,0,0],dtype=torch.float32,device='cuda')
    d=torch.load(a.nag,map_location='cpu'); snag=SemanticNAG(d['nag'],d['nag_feat']); labels=snag.labels[-1]; rids=torch.unique(labels).sort().values; rids_cpu=rids.detach().cpu().numpy().astype(np.int64)
    cams={c.image_name:c for c in scene.getTrainCameras()}; gt_dir=Path(a.gt_root)/a.scene; rows=[]
    with torch.no_grad():
      for jf in sorted(gt_dir.glob('*.json')):
        name=jf.stem; cam=cams.get(name)
        if cam is None: continue
        anno=json.loads(jf.read_text()); h,w=cam.image_height,cam.image_width; qg={}
        for o in anno.get('objects',[]): qg.setdefault(o['category'],np.zeros((h,w),np.uint8)); qg[o['category']]=np.maximum(qg[o['category']],poly_mask((h,w),o['segmentation']))
        masks=[]
        for st in range(0,len(rids_cpu),a.chunk):
          ids=rids_cpu[st:st+a.chunk]; sem=torch.zeros((labels.shape[0],len(ids)),dtype=torch.float32,device='cuda')
          for j,rid in enumerate(ids): sem[:,j]=(labels==int(rid)).float()
          if sem.shape[1]<a.chunk: sem=torch.cat([sem,torch.zeros((sem.shape[0],a.chunk-sem.shape[1]),device='cuda')],1)
          gauss._semantics=sem
          out=render(cam,gauss,pipe,bg)['semantics']; out=out.reshape(out.shape[0],-1)[:len(ids)].reshape(len(ids),h,w).detach().cpu().numpy()
          masks.append((out>0.5).astype(np.uint8))
        rm=np.concatenate(masks,0); np.savez_compressed(maskdir/(name+'.npz'),region_ids=rids_cpu,masks=rm)
        for qi,(query,gt) in enumerate(qg.items()):
          for ri,rid in enumerate(rids_cpu):
            reg=rm[ri].astype(bool); g=gt.astype(bool); inter=int((reg&g).sum()); rp=int(reg.sum()); gp=int(g.sum()); union=int((reg|g).sum())
            rows.append(dict(scene=a.scene,image=name,query=query,region_id=int(rid),gt_px=gp,region_px=rp,intersection_px=inter,precision=inter/rp if rp else 0.,recall=inter/gp if gp else 0.,iou=inter/union if union else 0.))
    pd.DataFrame(rows).to_parquet(root/'query_region_support.parquet',index=False)
    print('rows',len(rows),'regions',len(rids_cpu),'images',len(cams))
if __name__=='__main__': main()
