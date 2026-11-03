#!/usr/bin/env python3
"""Deterministic frozen-NAG observation + trace/beta sidecar export.

This is an isolated exporter.  It never calls assign_label/everything_seg and
does not modify the frozen NAG or the production model.
"""
import argparse, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<22),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',required=True)
    ap.add_argument('--source',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--nag',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--iteration',type=int,default=0)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=False)
    src=Path(a.source); model=Path(a.model); nagp=Path(a.nag)
    frozen=torch.load(nagp,map_location='cpu',weights_only=False)
    root=frozen['nag'][-1].long().flatten()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    import sys
    sys.path.insert(0,a.repo)
    import merge_proj
    from gaussian_renderer import render_point, trace
    import torch.nn.functional as F
    from argparse import Namespace
    args=Namespace(source_path=str(src),model_path=str(model),iteration=a.iteration,images='images',resolution=-1,white_background=False,data_device='cuda',eval=False,sh_degree=3,vlm_type='clip',thres_connect=[.9,.7,.7],thres_merge=20,max_neighbor_distance=2,view_freq=5,dis_decay=.5,thres_dis=.15,thres_trunc=0.,seg_enhance=False,tau=3.,seed=a.seed,mix='soft',agg='mean',observation_out=str(out/'observations_level3.npz'),render_items=['RGB','Alpha','Normal','Depth','Edge','Curvature'])
    class Agent(merge_proj.SAI3D):
        @torch.no_grad()
        def extract_gaussian_features(self, gaussians, views, feature_level):
            pipe=self.pipeline; bg=self.background; labels=[]; confs=[]; vmeta=[]
            for view in views:
                seg=view.semantic['seg_map'][feature_level].clone(); mask=view.semantic['fg_mask'][feature_level]
                seg_min,seg_max=seg[mask].min().item(),seg[mask].max().item(); seg_num=seg_max-seg_min+2
                seg-=seg_min-1; seg[~mask]=0
                enc=torch.normal(mean=0,std=1,size=(seg_num,20),device='cuda'); enc=F.normalize(enc,p=2,dim=-1)
                rp=trace(view,gaussians,enc[seg],None,pipe,bg)
                sem=F.normalize(rp['gaussian_semantics'],p=2,dim=-1); sim=sem@enc.T
                lab=sim.argmax(-1); cf=sim.max(-1).values; lab[cf<=.85]=0
                labels.append(lab); confs.append(cf); vmeta.append((seg_min,seg_max,seg_num))
            self._sas_labels=labels; self._sas_conf=confs; self._sas_vmeta=vmeta
            return torch.stack(labels).transpose(0,1)
        @torch.no_grad()
        def proj_gaussian_features(self, gaussians, views, feature_level, gau2sp, pt_sp_label, device='cuda'):
            WT=1e-4; RT=.3
            sp_uni,sp_gnum=torch.unique(gau2sp,return_counts=True,sorted=True); D=views[0].semantic['sem'].to(device).shape[1]
            feats=[[] for _ in range(sp_uni.shape[0])]; members=[]
            for sp in range(len(feats)): members.append(torch.nonzero(gau2sp==sp,as_tuple=False).flatten().cpu().numpy().astype(np.int64))
            obs_f=[]; obs_w=[]; obs_r=[]; obs_v=[]; vo=[0]; vi=[]; mo=[0]; mi=[]
            mask_off=[0]; mask_rows=[]; mask_beta_raw=[]; mask_beta_kept=[]; mask_lab=[]
            sup_off=[0]; sup_ids=[]; sup_w=[]; nonbg=[]; viscnt=[]; regcnt=[]; raw_feat=[]; rec_feat=[]; rec_vis=[]
            pipe=self.pipeline; bg=self.background
            for vid,view in enumerate(views):
                rp=render_point(view,gaussians,pipe,bg); weight=rp['weight']; gmask=weight>WT
                gt=view.semantic['sem'].to(device); seg=view.semantic['seg_map'][feature_level]; im=view.semantic['fg_mask'][feature_level]
                smin,smax=seg[im].min().item(),seg[im].max().item(); vf=gt[smin:smax+1].float(); seg_num=smax-smin+2
                raw=self._get_superpoint_mask_ratio(gau2sp,pt_sp_label[vid],seg_num); kept=raw.clone(); kept[kept<RT]=0
                spfeat=kept@vf; spidx=gau2sp[gmask]; usp,cnt=torch.unique(spidx,return_counts=True)
                labels=self._sas_labels[vid]
                for qi,sp in enumerate(usp.tolist()):
                    vis=float(cnt[qi].item()/sp_gnum[sp].item()); sf=F.normalize(spfeat[sp],p=2,dim=-1)
                    if not sf.any(): continue
                    feats[sp].append(sf*vis); obs_f.append(sf.detach().float().cpu().numpy()); obs_w.append(vis); obs_r.append(sp); obs_v.append(vid)
                    ids=torch.nonzero(gmask & (gau2sp==sp),as_tuple=False).flatten().cpu().numpy().astype(np.int64); vi.extend(ids.tolist()); vo.append(len(vi)); mi.extend(members[sp].tolist()); mo.append(len(mi))
                    row=kept[sp].detach().cpu().numpy(); rr=np.flatnonzero(row>0).astype(np.int32); mask_lab.extend((rr+1).tolist()); mask_rows.extend((smin+rr).tolist()); mask_beta_kept.extend(row[rr].astype(np.float32).tolist()); mask_beta_raw.extend(raw[sp].detach().cpu().numpy()[rr].astype(np.float32).tolist()); mask_off.append(len(mask_lab))
                    labs=labels[ids].detach().cpu().numpy(); valid=(labs>0)
                    sid=ids[valid]; sl=labs[valid].astype(np.int64); sw=np.asarray([float(row[int(x)-1]) if int(x)-1 < len(row) else 0.0 for x in sl],np.float32); ok=sw>0
                    sid=sid[ok]; sw=sw[ok]; order=np.argsort(sid); sid=sid[order]; sw=sw[order]; sup_ids.extend(sid.tolist()); sup_w.extend(sw.tolist()); sup_off.append(len(sup_ids))
                    nonbg.append(int((labels[members[sp]]>0).sum().item())); viscnt.append(int(len(ids))); regcnt.append(int(len(members[sp]))); raw_feat.append(spfeat[sp].detach().float().cpu().numpy()); rec_feat.append(sf.detach().float().cpu().numpy()); rec_vis.append(vis)
            result=F.normalize(torch.stack([F.normalize(torch.stack(x),p=2,dim=-1).mean(0) if x else torch.zeros(D,device=device) for x in feats]),p=2,dim=-1)
            np.savez_compressed(out/'observations_level3.npz',features=np.asarray(obs_f,np.float32),weights=np.asarray(obs_w,np.float32),region_ids=np.asarray(obs_r,np.int64),view_ids=np.asarray(obs_v,np.int64),visible_gaussian_offsets=np.asarray(vo,np.int64),visible_gaussian_ids=np.asarray(vi,np.int64),member_gaussian_offsets=np.asarray(mo,np.int64),member_gaussian_ids=np.asarray(mi,np.int64))
            np.savez_compressed(out/'observation_semantic_sidecar.npz',observation_id=np.arange(len(obs_f),dtype=np.int64),region_id=np.asarray(obs_r,np.int64),view_id=np.asarray(obs_v,np.int64),level_id=np.full(len(obs_f),feature_level,np.int32),mask_offsets=np.asarray(mask_off,np.int64),mask_labels=np.asarray(mask_lab,np.int32),mask_feature_rows=np.asarray(mask_rows,np.int32),mask_beta_raw=np.asarray(mask_beta_raw,np.float32),mask_beta_kept=np.asarray(mask_beta_kept,np.float32),support_offsets=np.asarray(sup_off,np.int64),support_gaussian_ids=np.asarray(sup_ids,np.int64),support_weights=np.asarray(sup_w,np.float32),nonbackground_count=np.asarray(nonbg,np.int32),visible_count=np.asarray(viscnt,np.int32),region_gaussian_count=np.asarray(regcnt,np.int32),reconstructed_feature=np.asarray(rec_feat,np.float32),reconstructed_visibility=np.asarray(rec_vis,np.float32),trace_view_offsets=np.asarray([0]+[int((x>0).sum().item()) for x in labels],np.int64).cumsum() if False else np.asarray([0],np.int64))
            meta={'seed':a.seed,'scene':src.name,'n_observations':len(obs_f),'nag_sha256':sha(nagp),'observation_sha256':sha(out/'observations_level3.npz'),'sidecar_sha256':sha(out/'observation_semantic_sidecar.npz'),'n_gaussians':int(self.N),'n_views':len(views),'root_regions':int(root.max().item())+1}
            (out/'EXPORT_PROVENANCE.json').write_text(json.dumps(meta,indent=2))
            return result
    agent=Agent(args); agent.init_data(args)
    if agent.N!=len(root): raise RuntimeError(f'Gaussian/NAG mismatch {agent.N} vs {len(root)}')
    views=agent.scene.getTrainCameras(); labels=agent.extract_gaussian_features(agent.gaussians,views,3)
    agent.proj_gaussian_features(agent.gaussians,views,3,root.cuda(),labels.T.cuda())
    with np.load(out/'observations_level3.npz') as d, np.load(out/'observation_semantic_sidecar.npz') as s:
        n=len(d['features']); assert n==len(s['observation_id']); assert len(d['visible_gaussian_offsets'])==n+1
        assert np.isfinite(d['features']).all() and np.isfinite(s['mask_beta_kept']).all(); assert np.all(s['support_weights']>=0)
        for pfx in ('visible','member'):
            p=d[pfx+'_gaussian_offsets']; ids=d[pfx+'_gaussian_ids']; assert p[-1]==len(ids) and np.all(np.diff(p)>=0)
        p=s['support_offsets']; ids=s['support_gaussian_ids']; assert p[-1]==len(ids) and np.all(np.diff(p)>=0); assert len(ids)==0 or (ids.min()>=0 and ids.max()<agent.N)
    print(json.dumps({'scene':src.name,'n_observations':n,'nag_sha256':sha(nagp),'observation_sha256':sha(out/'observations_level3.npz'),'sidecar_sha256':sha(out/'observation_semantic_sidecar.npz')},indent=2))
if __name__=='__main__': main()
