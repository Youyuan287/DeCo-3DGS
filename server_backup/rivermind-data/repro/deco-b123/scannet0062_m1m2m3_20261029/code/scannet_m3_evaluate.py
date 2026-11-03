"""ScanNet published DeCo protocol, with only fixed/adaptive auxiliary weights changed."""
from pathlib import Path
import os,sys,json,csv,hashlib,itertools
import numpy as np
import torch
torch.set_num_threads(4)
from scipy.spatial import cKDTree

ROOT=Path('/root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029')
CODE=ROOT/'code'
REL=Path('/root/rivermind-data/repro/deco-b123/frozen_ramen_relags')
MODEL=Path('/root/rivermind-data/output/scannet/scene0062_00')
SRC=Path('/root/rivermind-data/datasets/scannet/scene0062_00')
os.environ['HF_HUB_OFFLINE']='1';os.environ['RIVERMIND_ROOT']=str(ROOT/'reference_root')
sys.path[:0]=[str(CODE),str(REL),str(REL/'ext')]
from eval_scannet_3d import load_gt,load_gaussians,metrics,NYU40,SUBSETS
from evidence_readout_v2 import load_bank_v2,load_mode_support,auxiliary_correction
from evidence_readout_weight_ablation import fixed_weight_correction
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def writecsv(p,rows):
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@torch.no_grad()
def main():
 assert (ROOT/'BANK_READY').exists()
 manifest=json.loads((ROOT/'EXPORT_MANIFEST.json').read_text())
 assert all(sha(p)==h for p,h in manifest['inputs'].items())
 gt_path=ROOT/'scene0062_00_vh_clean_2.labels.ply';gtxyz,gtlabel=load_gt(str(gt_path))
 xyz,opacity,keep,ply=load_gaussians(str(MODEL),0)
 dist,idx=cKDTree(gtxyz).query(xyz,k=1,workers=8);gt=gtlabel[idx]
 nag_path=MODEL/'sai_nag_3.0.pt';nag=torch.load(nag_path,map_location='cpu')
 labels=nag['nag'][-1].numpy();features=nag['nag_feat'][-1].numpy()
 assert len(labels)==len(xyz) and len(features)==int(labels.max())+1
 # Execute the original released ScanNet scorer unchanged against a new GT cache.
 rr=ROOT/'reference_root';(rr/'clip_cache').mkdir(parents=True,exist_ok=True);(rr/'output/scannet').mkdir(parents=True,exist_ok=True)
 link=rr/'output/scannet/scene0062_00'
 if not link.exists():link.symlink_to(MODEL,target_is_directory=True)
 np.savez_compressed(rr/'clip_cache/scene0062_00.npz',gt=gt,keep=keep)
 import score_scannet as published
 sys.argv=['score_scannet.py','--renorm','0','--out',str(ROOT/'original_deco_reference')]
 published.main()
 reference=json.loads((ROOT/'original_deco_reference/scene0062_00__r0.json').read_text())
 import open_clip
 model,_,_=open_clip.create_model_and_transforms('ViT-B-16',pretrained='laion2b_s34b_b88k')
 model.eval();tok=open_clip.get_tokenizer('ViT-B-16')
 def emb(words):
  z=model.encode_text(tok(list(words)));return (z/z.norm(dim=-1,keepdim=True)).float().cpu().numpy()
 l2=published.l2n;classes=SUBSETS[19];names=[NYU40[c] for c in classes]
 templates=np.stack([emb([p.format(n) for n in names]) for p in published.PROMPTS]);tpl=l2(templates-templates.mean(1,keepdims=True))
 v=max(len(published.LEX[n]) for n in names)
 lex=np.transpose(np.stack([emb(published.LEX[n]+[published.LEX[n][0]]*(v-len(published.LEX[n]))) for n in names]),(1,0,2))
 lex=l2(lex-lex.mean(1,keepdims=True));textbank=np.concatenate([tpl,lex],0)
 np.savez_compressed(ROOT/'text_bank.npz',textbank=textbank,class_ids=classes)
 x=l2(features);mean=x.mean(0,keepdims=True);xc=x-mean
 def c2(descriptor):
  sc=np.stack([descriptor@b.T for b in textbank]);m=sc.max(0)
  return published.TAU*np.log(np.exp((sc-m)/published.TAU).mean(0))+m
 main_scores=c2(xc);pred_root_off=np.array(classes)[main_scores.argmax(1)];pred_off=pred_root_off[labels]
 offmetric=metrics(gt[keep],pred_off[keep],classes)
 assert offmetric==reference['subsets']['19'],'OFF mismatch against unchanged published scorer'
 print('BASELINE_PARITY_PASS',offmetric['mIoU'],flush=True)
 bank,bmeta=load_bank_v2(str(ROOT/'B1.npz'),str(nag_path),len(x),512,str(ROOT/'observations_level3.npz'),str(ROOT/'B1.members.json'))
 _,_,view_count,eligible=load_mode_support(str(ROOT/'observations_level3.npz'),str(ROOT/'B1.members.json'),bank)
 slot=c2((bank['features']-mean.reshape(1,1,512)).reshape(-1,512)).reshape(*bank['priors'].shape,len(classes))
 roots={'off':main_scores};details={};trace=[]
 for arm,func in [('fixed',fixed_weight_correction),('adaptive',auxiliary_correction)]:
  out=main_scores.copy()
  for ci,c in enumerate(classes):
   new,d=func(torch.from_numpy(main_scores[:,ci].copy()),torch.from_numpy(slot[:,:,ci].copy()),bank,eligible,np.ones_like(bank['priors']),temperature=.08,coefficient=.25)
   out[:,ci]=new.numpy();details[(arm,c)]=d
   for ri,rid in enumerate(bank['region_ids']):
    for k in range(bank['priors'].shape[1]):
     trace.append(dict(arm=arm,class_id=c,query=NYU40[c],root_id=int(rid),mode_id=k,valid=bool(bank['valid'][ri,k]),eligible=bool(eligible[ri,k]),views=int(view_count[ri,k]),prior=float(bank['priors'][ri,k]),main_score=float(main_scores[rid,ci]),mode_score=float(slot[ri,k,ci]),advantage=float(d['advantage'][ri,k]),weight=float(d['query_weights'][ri,k]),reliability=1.,delta=float(d['delta'][ri]),correction=float(d['correction'][ri])))
  roots[arm]=out
 for c in classes:
  a=details[('fixed',c)];b=details[('adaptive',c)]
  torch.testing.assert_close(a['advantage'],b['advantage'],rtol=0,atol=0)
  torch.testing.assert_close(a['eligible'],b['eligible'],rtol=0,atol=0)
 pred_roots={arm:np.array(classes)[scores.argmax(1)] for arm,scores in roots.items()}
 preds={arm:p[labels] for arm,p in pred_roots.items()}
 np.savez_compressed(ROOT/'predictions_and_scores.npz',gt=gt,labels=labels,xyz=xyz,**{f'pred_{k}':v for k,v in preds.items()},**{f'scores_{k}':v for k,v in roots.items()})
 results={a:metrics(gt[keep],p[keep],classes) for a,p in preds.items()}
 # Classwise binary diagnostic: ScanNet uses class argmax, not the LERF candidate-gap selector.
 valid=np.isin(gt,classes);per=[];changes=[]
 for c in classes:
  row=dict(class_id=c,query=NYU40[c],gt_count=int(np.count_nonzero(gt==c)))
  masks={a:p==c for a,p in preds.items()}
  for arm,p in preds.items():
   mask=masks[arm];tp=int(np.count_nonzero(mask&(gt==c)&valid));fp=int(np.count_nonzero(mask&(gt!=c)&valid));fn=int(np.count_nonzero(~mask&(gt==c)&valid))
   row.update({arm+'_TP':tp,arm+'_FP':fp,arm+'_FN':fn,arm+'_IoU':tp/(tp+fp+fn) if tp+fp+fn else '',arm+'_selected_roots':json.dumps(np.flatnonzero(pred_roots[arm]==c).tolist()),arm+'_pred_gaussians_all':int(mask.sum())})
  for a,b in [('off','fixed'),('off','adaptive'),('fixed','adaptive')]:
   row[a+'_vs_'+b+'_gaussian_xor']=int(np.count_nonzero(masks[a]!=masks[b]));row[a+'_to_'+b+'_gaussians_added']=int(np.count_nonzero(~masks[a]&masks[b]));row[a+'_to_'+b+'_gaussians_removed']=int(np.count_nonzero(masks[a]&~masks[b]))
  per.append(row)
 for rid in range(len(x)):
  for ci,c in enumerate(classes):
   changes.append(dict(root_id=rid,class_id=c,query=NYU40[c],off_score=float(roots['off'][rid,ci]),fixed_score=float(roots['fixed'][rid,ci]),adaptive_score=float(roots['adaptive'][rid,ci]),off_class=int(pred_roots['off'][rid]),fixed_class=int(pred_roots['fixed'][rid]),adaptive_class=int(pred_roots['adaptive'][rid]),gaussian_count=int(np.count_nonzero(labels==rid))))
 writecsv(ROOT/'SCANNET_M3_PER_CLASS.csv',per);writecsv(ROOT/'SCANNET_M3_WEIGHT_TRACE.csv',trace);writecsv(ROOT/'SCANNET_M3_CANDIDATE_CHANGE.csv',changes)
 summary=dict(scene='scene0062_00',num_gaussians=len(xyz),root_count=len(x),class_count=19,gt_present=offmetric['present'],metrics=results,original_deco_parity=True,protocol='released ScanNet both_lse, C1 renorm=0, 19-class argmax, nearest-GT-point per Gaussian; auxiliary T=.08 coefficient=.25 M2 off',nearest_gt=dict(mean=float(dist.mean()),median=float(np.median(dist)),p95=float(np.percentile(dist,95))),hashes={str(p):sha(p) for p in [nag_path,Path(ply),ROOT/'B1.npz',ROOT/'B1.members.json',ROOT/'observations_level3.npz',gt_path,ROOT/'text_bank.npz']})
 (ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print('METRICS',json.dumps({a:v['mIoU'] for a,v in results.items()}),flush=True)
 del model;torch.cuda.empty_cache()
 render_masks(preds,classes)
 assert all(sha(p)==h for p,h in manifest['inputs'].items()),'frozen input changed'
 (ROOT/'COMPLETE').touch()

@torch.no_grad()
def render_masks(preds,classes):
 from argparse import Namespace,ArgumentParser
 from scene import Scene,GaussianModel
 from arguments import PipelineParams
 from gaussian_renderer import render
 import random
 random.seed(1234)
 args=Namespace(source_path=str(SRC),model_path=str(MODEL),images='images',resolution=-1,white_background=False,data_device='cuda',eval=True,sh_degree=3,render_items=['RGB','Alpha','Normal','Depth','Edge','Curvature'])
 g=GaussianModel(3,20);scene=Scene(args,g,load_iteration=0,shuffle=False,load_sem=False,load_img=False,load_test=True)
 # Production Blender loader deliberately returns an empty test list.
 # Read the genuine held-out transforms explicitly for diagnostics only.
 from scene.dataset_readers import readCamerasFromTransforms
 import scene.dataset_readers as readers
 import inspect
 from utils.camera_utils import cameraList_from_camInfos
 # A second production-loader bug leaves semantic unbound if load_sem=False.
 # Execute a local copy with only its optional-data initialization repaired.
 camera_source=inspect.getsource(readCamerasFromTransforms)
 assert '            if load_sem:' in camera_source
 camera_source=camera_source.replace('            if load_sem:','            semantic = None\n            if load_sem:')
 env=dict(readers.__dict__);exec(compile(camera_source,'isolated_camera_reader.py','exec'),env)
 infos=env['readCamerasFromTransforms'](str(SRC),'transforms_test.json',False,extension='.jpg',load_img=False,load_sem=False)
 cams=cameraList_from_camInfos(infos,1.0,args);assert len(cams)==len(json.loads((SRC/'transforms_test.json').read_text())['frames'])==25
 pipe=PipelineParams(ArgumentParser());bg=torch.zeros(3,device='cuda');rows=[]
 for cam in cams:
  for c in classes:
   masks={}
   for arm,p in preds.items():
    g._semantics=torch.from_numpy((p==c).astype(np.float32)).cuda()[:,None].expand(-1,20).contiguous()
    sem=render(cam,g,pipe,bg)['semantics'];masks[arm]=(sem[0]>.5).cpu().numpy()
   row=dict(image=cam.image_name,class_id=c,query=NYU40[c],off_pixels=int(masks['off'].sum()),fixed_pixels=int(masks['fixed'].sum()),adaptive_pixels=int(masks['adaptive'].sum()))
   for a,b in [('off','fixed'),('off','adaptive'),('fixed','adaptive')]:row[a+'_vs_'+b+'_mask_xor']=int(np.count_nonzero(masks[a]!=masks[b]))
   for arm,mask in masks.items():row[arm+'_mask_sha256']=hashlib.sha256(mask.tobytes()).hexdigest()
   rows.append(row)
  print('RENDER',cam.image_name,flush=True)
 writecsv(ROOT/'SCANNET_M3_MASK_XOR.csv',rows)
 (ROOT/'RENDER_PROTOCOL.json').write_text(json.dumps(dict(views=[c.image_name for c in cams],classes=classes,threshold=.5,gt_2d=False,camera_reader_sha256=hashlib.sha256(camera_source.encode()).hexdigest(),camera_adapter='explicit test transforms, initialize optional semantic=None',note='render diagnostics only; TP FP FN are genuine 3D Gaussian-vs-mesh GT metrics, not invented 2D GT'),indent=2))
if __name__=='__main__':
 if '--render-only' in sys.argv:
  with np.load(ROOT/'predictions_and_scores.npz') as z:preds={a:z['pred_'+a] for a in ['off','fixed','adaptive']}
  render_masks(preds,SUBSETS[19])
  manifest=json.loads((ROOT/'EXPORT_MANIFEST.json').read_text())
  assert all(sha(p)==h for p,h in manifest['inputs'].items())
  (ROOT/'COMPLETE').touch()
 else:main()
