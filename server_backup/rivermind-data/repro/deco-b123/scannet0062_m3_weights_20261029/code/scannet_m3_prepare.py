"""Freeze and verify scene0062_00 inputs, then export observations without clustering."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,zipfile,random
import numpy as np
import torch
from plyfile import PlyData

ROOT=Path('/root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029')
REL=Path('/root/rivermind-data/repro/deco-b123/frozen_ramen_relags')
PROD=Path('/root/rivermind-data/ReLaGS_Reproduction/ReLaGS')
MODEL=Path('/root/rivermind-data/output/scannet/scene0062_00')
SRC=Path('/root/rivermind-data/datasets/scannet/scene0062_00')
DECO=Path('/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
def xyz(p):
 v=PlyData.read(str(p))['vertex'];return np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float32)
def main():
 ROOT.mkdir(parents=True,exist_ok=True);code=ROOT/'code';code.mkdir(exist_ok=True)
 paths=[MODEL/'point_cloud/iteration_0/point_cloud.ply',MODEL/'point_cloud/iteration_30000/point_cloud.ply',MODEL/'nag-l1.pt',MODEL/'sai_nag_3.0.pt',MODEL/'saifine_labels.npy',MODEL/'scannetsaifine.ply',MODEL/'cfg_args',SRC/'transforms_train.json',SRC/'transforms_test.json']
 manifest={'inputs':{str(p):sha(p) for p in paths}}
 manifest['production_commit']=subprocess.check_output(['git','-C',str(PROD),'rev-parse','HEAD'],text=True).strip()
 manifest['production_status']=subprocess.check_output(['git','-C',str(PROD),'status','--porcelain'],text=True)
 nag=torch.load(MODEL/'sai_nag_3.0.pt',map_location='cpu');labels=nag['nag'];f=nag['nag_feat']
 x=xyz(paths[0]);old=xyz(paths[1]);packed=lambda a:np.ascontiguousarray(a).view(np.dtype('V12')).ravel()
 oi=np.argsort(packed(old));pos=np.searchsorted(packed(old)[oi],packed(x));assert (pos<len(old)).all()
 ix=oi[pos];assert np.array_equal(x,old[ix]) and (np.diff(ix)>0).all(),'pruned point order/source mismatch'
 assert np.array_equal(x,xyz(MODEL/'scannetsaifine.ply')),'semantic PLY point-order mismatch'
 assert all(len(a)==len(x) for a in labels)
 assert torch.equal(labels[0],torch.load(MODEL/'nag-l1.pt',map_location='cpu'))
 saved=np.load(MODEL/'saifine_labels.npy');assert np.array_equal(saved.reshape(-1)-1,labels[-1].numpy())
 assert all(torch.equal(torch.unique(a),torch.arange(int(a.max())+1)) for a in labels)
 assert all(int(labels[i+1].max())+1==len(f[i]) for i in range(len(f)))
 frames=json.loads((SRC/'transforms_train.json').read_text())['frames'];assert len(frames)==219
 feature_hash={}
 for frame in frames:
  name=Path(frame['file_path']).stem
  for suffix in ['_f.npy','_s.npy']:
   p=SRC/'language_features'/(name+suffix);assert p.exists();feature_hash[str(p)]=sha(p)
  feature=np.load(SRC/'language_features'/(name+'_f.npy'),mmap_mode='r')
  seg=np.load(SRC/'language_features'/(name+'_s.npy'),mmap_mode='r')
  assert feature.shape[-1]==512 and seg.shape[0]>3 and np.isfinite(feature).all()
 manifest.update(n_gaussian=len(x),trained_gaussians=len(old),pruned_is_ordered_subset=True,semantic_ply_xyz_equal=True,l1_equal=True,labels_equal_saifine=True,region_counts=[int(a.max())+1 for a in labels],train_views=len(frames),feature_hash=feature_hash)
 gt=ROOT/'scene0062_00_vh_clean_2.labels.ply'
 if not gt.exists():
  with zipfile.ZipFile('/root/rivermind-data/VALA/dataset/scannet.zip') as z:
   name='scannet/scans/scene0062_00/scene0062_00_vh_clean_2.labels.ply'
   with z.open(name) as src,gt.open('wb') as dst:shutil.copyfileobj(src,dst)
 manifest['gt_sha256']=sha(gt);manifest['gt_source']='existing VALA/dataset/scannet.zip exact labelled mesh'
 for src in [REL/'merge_proj.py',DECO/'lerf_ovs/evidence_modes.py',DECO/'lerf_ovs/build_bank_b123.py',DECO/'lerf_ovs/evidence_readout_v2.py',DECO/'lerf_ovs/evidence_readout_weight_ablation.py',DECO/'scannet/score_scannet.py',DECO/'scannet/eval_scannet_3d.py',DECO/'deco_ops.py']:
  dst=code/src.name
  if dst.exists():assert sha(dst)==sha(src),'snapshot conflict'
  else:shutil.copy2(src,dst)
  manifest.setdefault('source_sha256',{})[str(src)]=sha(src)
 manifest['production_projection_sha256']=sha(PROD/'merge_proj.py')
 (ROOT/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('PREFLIGHT PASS',manifest['region_counts'],flush=True)
 # Use the previously audited export sink, and deterministic training-camera order.
 sys.path[:0]=[str(code),str(REL),str(REL/'ext')]
 os.environ['HF_HUB_OFFLINE']='1';random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
 from argparse import Namespace
 import merge_proj
 args=Namespace(source_path=str(SRC),model_path=str(MODEL),iteration=0,images='images',resolution=-1,white_background=False,data_device='cuda',eval=True,sh_degree=3,vlm_type='clip',thres_connect=[.9,.7,.7],thres_merge=20,max_neighbor_distance=2,view_freq=5,dis_decay=.5,thres_dis=.15,thres_trunc=0.,seg_enhance=False,tau=3.,seed=1234,mix='soft',agg='mean',observation_out=str(ROOT/'observations_level3.npz'),render_items=['RGB','Alpha','Normal','Depth','Edge','Curvature'])
 agent=merge_proj.SAI3D(args);agent.init_data(args);views=agent.scene.getTrainCameras();assert len(views)==219
 (ROOT/'camera_order.json').write_text(json.dumps([v.image_name for v in views]))
 obs=ROOT/'observations_level3.npz'
 if obs.exists():raise FileExistsError(obs)
 pt=agent.extract_gaussian_features(agent.gaussians,views,3)
 assert pt.shape==(len(x),len(views))
 agent.proj_gaussian_features(agent.gaussians,views,3,labels[-1].cuda(),pt.T.cuda())
 with np.load(obs) as z:
  n=len(z['features']);assert z['features'].shape==(n,512) and np.isfinite(z['features']).all();assert np.isfinite(z['weights']).all() and (z['weights']>0).all()
  for prefix in ['visible','member']:
   ptr=z[prefix+'_gaussian_offsets'];ids=z[prefix+'_gaussian_ids'];assert len(ptr)==n+1 and ptr[0]==0 and ptr[-1]==len(ids) and (np.diff(ptr)>=0).all()
   assert (ids>=0).all() and (ids<len(x)).all()
   for i,r in enumerate(z['region_ids']):assert np.all(labels[-1].numpy()[ids[ptr[i]:ptr[i+1]]]==r),'cross-region CSR'
 manifest['observations']=n;manifest['observation_sha256']=sha(obs)
 assert all(sha(p)==h for p,h in manifest['inputs'].items()),'frozen input changed'
 (ROOT/'EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print('EXPORT PASS',n,flush=True)
 subprocess.run([sys.executable,str(code/'build_bank_b123.py'),'--observations',str(obs),'--nag',str(MODEL/'sai_nag_3.0.pt'),'--out',str(ROOT/'B1.npz'),'--b1','1','--b2','0','--geo-weight','.25','--max-modes','3'],check=True)
 (ROOT/'BANK_READY').touch()
if __name__=='__main__':main()
