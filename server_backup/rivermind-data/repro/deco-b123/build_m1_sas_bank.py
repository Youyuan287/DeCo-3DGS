#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from semantic_overlap import weighted_jaccard_csr, pairwise_weighted_jaccard

EPS=1e-10
def norm(x):
 x=np.asarray(x,np.float32); return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)
def csr(d,prefix,i):
 p=d[prefix+'_offsets']; key=prefix+'_gaussian_ids' if prefix in ('visible','member','support') else prefix+'_ids'; ids=d[key]; return ids[p[i]:p[i+1]]
def build(obs,side,out,mode='semantic',max_modes=3,geo_weight=.25):
 d=np.load(obs); s=np.load(side) if side else None
 f=norm(d['features']); w=d['weights'].astype(np.float32); r=d['region_ids'].astype(np.int64); v=d['view_ids'].astype(np.int64)
 if mode=='semantic' and s is None: raise ValueError('semantic mode requires sidecar')
 if mode=='semantic' and len(s['observation_id'])!=len(f): raise ValueError('sidecar observation mismatch')
 u=norm(f.mean(0)); h=norm(f-(f@u)[:,None]*u[None,:]); regions=np.unique(r); R=len(regions); D=f.shape[1]
 feat=np.zeros((R,max_modes,D),np.float32); res=np.zeros_like(feat); pri=np.zeros((R,max_modes),np.float32); valid=np.zeros((R,max_modes),bool); base=np.zeros((R,D),np.float32); counts=np.zeros(R,np.int32); members=[]; changed=0; pairs=0; diffs=[]
 for z,rid in enumerate(regions):
  ix=np.flatnonzero(r==rid); a=w[ix]/w[ix].sum(); ff=f[ix]; hh=h[ix]; base[z]=norm((a[:,None]*ff).sum(0)); base_sim=hh@hh.T
  sim=base_sim.copy()
  if len(ix)>1 and geo_weight>0:
   if mode=='semantic' and pairwise_weighted_jaccard is not None:
    ov=pairwise_weighted_jaccard(ix.astype(np.int64),s['support_offsets'],s['support_gaussian_ids'],s['support_weights'])
    pairs += len(ix)*(len(ix)-1)//2; diffs.extend(ov[np.triu_indices(len(ix),1)].tolist())
   else:
    ov=np.eye(len(ix),dtype=np.float32)
    for ii in range(len(ix)):
     for jj in range(ii+1,len(ix)):
      oi,oj=int(ix[ii]),int(ix[jj])
      if mode=='semantic': j=weighted_jaccard_csr(csr(s,'support',oi),s['support_weights'][s['support_offsets'][oi]:s['support_offsets'][oi+1]],csr(s,'support',oj),s['support_weights'][s['support_offsets'][oj]:s['support_offsets'][oj+1]])
      else:
       ai=csr(d,'visible',oi); aj=csr(d,'visible',oj); den=len(set(ai.tolist())|set(aj.tolist())); j=0.0 if den==0 else len(set(ai.tolist())&set(aj.tolist()))/den
      ov[ii,jj]=ov[jj,ii]=j; pairs+=1; diffs.append(j)
   sim=(1-geo_weight)*base_sim+geo_weight*(2*ov-1)
  centers=[int(np.argmax(sim@a))]
  for _ in range(min(max_modes,len(ix))-1):
   nov=1-np.max(sim[:,centers],axis=1); nov[centers]=-np.inf; q=int(np.argmax(nov))
   if nov[q]<=1e-6: break
   centers.append(q)
  assign=np.argmax(sim[:,centers],axis=1); groups=[]
  for k in range(int(assign.max())+1):
   sel=np.flatnonzero(assign==k); 
   if len(sel): groups.append((float(a[sel].sum()),sel))
  groups.sort(key=lambda x:-x[0]); counts[z]=len(groups)
  for k,(mass,sel) in enumerate(groups):
   aa=a[sel]; pri[z,k]=float(aa.sum()); feat[z,k]=norm((aa[:,None]*ff[sel]).sum(0)); res[z,k]=norm((aa[:,None]*hh[sel]).sum(0)); valid[z,k]=True
   members.append({'region_id':int(rid),'slot':k,'observation_indices':ix[sel].tolist(),'view_ids':v[ix[sel]].tolist()})
  pri[z]/=max(pri[z].sum(),EPS)
 prov=Path(obs).with_name('EXPORT_PROVENANCE.json'); pmeta=json.loads(prov.read_text()) if prov.exists() else {}
 meta={'overlap_mode':mode,'max_modes':max_modes,'geo_weight':geo_weight,'observation_sha256':hashlib.sha256(Path(obs).read_bytes()).hexdigest(),'sidecar_sha256':hashlib.sha256(Path(side).read_bytes()).hexdigest() if side else None,'nag_sha256':pmeta.get('nag_sha256'),'n_regions':R,'n_observations':len(f),'pair_count':pairs,'jaccard_mean':float(np.mean(diffs)) if diffs else 0.0}
 out=Path(out); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,features=feat,residuals=res,priors=pri,valid=valid,base_features=base,group_count=counts,region_ids=regions,metadata=json.dumps(meta,sort_keys=True),semantic_support_offsets=s['support_offsets'] if s is not None else np.array([],np.int64),semantic_support_ids=s['support_gaussian_ids'] if s is not None else np.array([],np.int64),semantic_support_weights=s['support_weights'] if s is not None else np.array([],np.float32),empty_support_flags=(np.diff(s['support_offsets'])==0) if s is not None else np.array([],bool))
 out.with_suffix('.members.json').write_text(json.dumps({'metadata':meta,'memberships':members},indent=2))
 return meta
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--observations',required=True); ap.add_argument('--sidecar'); ap.add_argument('--out',required=True); ap.add_argument('--mode',choices=['visible','semantic'],default='semantic'); ap.add_argument('--max-modes',type=int,default=3); ap.add_argument('--geo-weight',type=float,default=.25); a=ap.parse_args(); print(json.dumps(build(a.observations,a.sidecar,a.out,a.mode,a.max_modes,a.geo_weight),indent=2))
