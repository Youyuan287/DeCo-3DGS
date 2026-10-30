"""B1/B2 evidence construction for the complete DeCo-3DGS pipeline.
B1: residual-guided grouping, weighted within-mode full-CLIP descriptors.
B2: optional ROFA within each mode; never use majority consensus across modes.
No GT, prompts or dataset labels used. Explicit geometry support for optional
view overlap: sparse per-observation *visible* gaussian CSR (not all members).
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import numpy as np
EPS=1e-10

def norm(x):
    x=np.asarray(x,np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),EPS)

@dataclass(frozen=True)
class BuildConfig:
    max_modes:int=3
    min_mass:float=0.0
    b1:bool=True
    b2:bool=False
    grouping:str='debiased'  # "raw" only for controlled diagnostic
    geo_weight:float=0.25
    geometry_required:bool=True
    rofa_tau:float=3.0
    rofa_mode:str='raw'  # "raw" (released), "center", or "off"
    mode_selection:str='original'  # "original" or "rcs"


def compute_cross_view_support(residual_features, observation_weights, view_ids):
    """Compute view-deduplicated positive semantic support for each observation."""
    x = norm(residual_features)
    w = np.asarray(observation_weights, np.float32)
    v = np.asarray(view_ids, np.int64)
    n = len(x)
    score = np.zeros(n, np.float32)
    count = np.zeros(n, np.int32)
    if n == 0:
        return score, count
    finite = np.isfinite(x).all(1) & np.isfinite(w) & (w > 0)
    for i in range(n):
        if not finite[i]:
            continue
        other = (v != v[i]) & finite
        if not other.any():
            continue
        # One contribution per independent view: take its strongest positive match.
        view_scores = []
        total = 0.0
        for view in np.unique(v[other]):
            ids = np.flatnonzero(other & (v == view))
            c = np.maximum(x[ids] @ x[i], 0.0)
            best = float(np.max(c)) if len(c) else 0.0
            mass = float(np.sum(w[ids]))
            view_scores.append((best, mass))
            total += mass
        if total <= 0:
            continue
        score[i] = sum((mass / total) * best for best, mass in view_scores)
        count[i] = sum(best > 0 for best, _ in view_scores)
    return score, count


def select_rcs_seeds(affinity_matrix, support_score, support_view_count, max_modes=3):
    """Select supported seeds while preserving original affinity novelty priority."""
    a = np.asarray(affinity_matrix, np.float32)
    r = np.asarray(support_score, np.float32)
    c = np.asarray(support_view_count, np.int32)
    eligible = np.flatnonzero((c >= 1) & np.isfinite(r))
    if not len(eligible) or max_modes < 1:
        return []
    first = int(eligible[np.argmax(r[eligible])])
    centers = [first]
    while len(centers) < min(max_modes, len(eligible)):
        novelty = 1.0 - np.max(a[:, centers], axis=1)
        novelty[~np.isin(np.arange(len(a)), eligible)] = -np.inf
        novelty[centers] = -np.inf
        best = max((int(j) for j in eligible if j not in centers),
                   key=lambda j: (float(novelty[j]), float(r[j]), -j))
        if not np.isfinite(novelty[best]) or novelty[best] <= 1e-6:
            break
        centers.append(best)
    return centers


def rofa_keep(unit_features,tau=3.0):
    """Exact original mean-pairwise ROFA lower-tail, applied to ONE mode."""
    x=norm(unit_features)
    if len(x)<=1:return np.ones(len(x),bool)
    sim=x@x.T
    q=(sim.sum(1)-1)/(len(x)-1)
    cut=q.mean()-tau*q.std(ddof=0)
    keep=q>cut
    if not keep.any():keep[np.argmax(q)]=True
    return keep


def load_observations(path, *, require_geometry):
    with np.load(path,allow_pickle=False) as d:
        f=d['features'].astype(np.float32)
        w=d['weights'].astype(np.float32)
        r=d['region_ids'].astype(np.int64)
        v=d['view_ids'].astype(np.int64)
        keys=set(d.files)
        # Exported by audit projection sink (expected names are explicit; do not guess)
        ptr_key=next((k for k in ('visible_gaussian_offsets','visible_gaussian_indptr','visible_gaussian_ptr') if k in keys),None)
        ids_key='visible_gaussian_ids' if 'visible_gaussian_ids' in keys else None
        ptr=d[ptr_key].astype(np.int64) if ptr_key else None
        ids=d[ids_key].astype(np.int64) if ids_key else None
    if not (f.ndim==2 and w.shape==(len(f),) and r.shape==(len(f),) and v.shape==(len(f),)):
        raise ValueError('observation arrays have inconsistent length')
    if not np.isfinite(f).all() or not np.isfinite(w).all() or np.any(w<=0):
        raise ValueError('invalid production observations')
    if ptr is not None and (ptr.shape!=(len(f)+1,) or ptr[0]!=0 or ptr[-1]!=len(ids) or np.any(np.diff(ptr)<0)):
        raise ValueError('invalid visible-gaussian CSR')
    if require_geometry and ptr is None:
        raise ValueError('B1 geometry requested but observation visible_gaussian CSR is missing. Do not silently fall back. Inspect NPZ keys and adapt mapping explicitly.')
    return f,w,r,v,ptr,ids


def weighted_overlap(indices,ptr,visible):
    """Actual visibility Jaccard; duplicate view IDs do not count as geometry."""
    n=len(indices)
    sets=[set(visible[ptr[i]:ptr[i+1]].tolist()) for i in indices]
    overlap=np.eye(n,dtype=np.float32)
    for i in range(n):
        for j in range(i+1,n):
            union=len(sets[i]|sets[j]); ov=len(sets[i]&sets[j])/union if union else 0.
            overlap[i,j]=overlap[j,i]=ov
    return overlap


def build_bank(obs,output,nag_sha256, cfg=BuildConfig()):
    if cfg.max_modes<1 or cfg.min_mass<0 or cfg.min_mass>=1 or not (0<=cfg.geo_weight<=1):raise ValueError('invalid configuration')
    if cfg.mode_selection not in ('original','rcs'):raise ValueError('invalid mode_selection')
    if cfg.rofa_mode not in ('raw','center','off'):raise ValueError('invalid rofa_mode')
    f,w,r,v,ptr,visible=load_observations(obs,require_geometry=cfg.b1 and cfg.geometry_required and cfg.geo_weight>0)
    f=norm(f)
    vmean=np.zeros((int(v.max())+1,f.shape[1]),np.float32)
    for vid in np.unique(v):
        vmean[vid]=norm(f[v==vid].mean(0))
    u=norm(f.mean(0))  # per-scene no-label prototype; for frozen partition only
    h=norm(f-(f@u)[:,None]*u[None,:]) if cfg.grouping=='debiased' else f
    if cfg.grouping not in ('debiased','raw'):raise ValueError(cfg.grouping)
    if cfg.b1 and cfg.grouping!='debiased':raise ValueError('B1 requires debiased grouping')
    regions=np.unique(r);R=len(regions); K=cfg.max_modes;D=f.shape[1]
    feat=np.zeros((R,K,D),np.float32); res=np.zeros_like(feat)
    mass=np.zeros((R,K),np.float32); valid=np.zeros((R,K),bool)
    base=np.zeros((R,D),np.float32); count=np.zeros(R,np.int32)
    memberships=[]; dropped=[]; rcs_audit=[]
    for z,rid in enumerate(regions):
        ix=np.flatnonzero(r==rid); a=w[ix]/w[ix].sum();ff=f[ix];hh=h[ix]
        base[z]=norm((a[:,None]*ff).sum(0))
        if cfg.b1 and len(ix)>1 and K>1:
            # Fixed mode budget. Geometry combines shared spatial support with semantics
            sem=hh@hh.T
            if cfg.geo_weight>0 and ptr is not None:
                ov=weighted_overlap(ix,ptr,visible)
                similarity=(1-cfg.geo_weight)*sem+cfg.geo_weight*(2*ov-1)
            else:similarity=sem
            if cfg.mode_selection == 'rcs':
                support_score, support_view_count = compute_cross_view_support(hh, w[ix], v[ix])
                centers=select_rcs_seeds(similarity, support_score, support_view_count, K)
                rcs_audit.append({'region_id':int(rid),'supported_observations':int(np.sum(support_view_count>=1)),
                                  'seed_indices':[int(ix[j]) for j in centers],
                                  'seed_support_view_count':[int(support_view_count[j]) for j in centers],
                                  'seed_support_score':[float(support_score[j]) for j in centers],
                                  'original_seed_count':int(min(K,len(ix)))})
                if not centers:
                    count[z]=0
                    continue
                else:
                    assign=np.argmax(similarity[:,centers],axis=1)
            else:
                centers=[int(np.argmax(similarity@a))]
                for _ in range(min(K,len(ix))-1):
                    novelty=1-np.max(similarity[:,centers],axis=1)
                    novelty[centers]=-np.inf
                    j=int(np.argmax(novelty))
                    if novelty[j]<=1e-6:break
                    centers.append(j)
                assign=np.argmax(similarity[:,centers],axis=1)
        else: assign=np.zeros(len(ix),np.int64)
        groups=[]
        for j in range(int(assign.max())+1):
            members=np.flatnonzero(assign==j)
            if not len(members):continue
            prior=float(a[members].sum())
            if prior<cfg.min_mass and len(ix)>1:
                dropped.extend(ix[members].tolist());continue
            original=members.copy()
            if cfg.b2:
                # Robustness only INSIDE the candidate mode; minority mode survives
                if cfg.rofa_mode=='raw':
                    members=members[rofa_keep(ff[members],cfg.rofa_tau)]
                elif cfg.rofa_mode=='center':
                    cen=norm(ff[members]-vmean[v[ix[members]]])
                    members=members[rofa_keep(cen,cfg.rofa_tau)]
                if not len(members):raise AssertionError('ROFA empty group')
            groups.append((prior,members,original))
        if not groups:
            groups=[(1.0,np.arange(len(ix)),np.arange(len(ix)))]
        groups.sort(key=lambda t:-t[0])
        for k,(pr,sel,orig) in enumerate(groups):
            # Update prior to retained observation support after intra-mode ROFA
            actual=float(a[sel].sum())
            mass[z,k]=actual
            feat[z,k]=norm((a[sel,None]*ff[sel]).sum(0))
            res[z,k]=norm((a[sel,None]*hh[sel]).sum(0))
            valid[z,k]=True
            memberships.append({'region_id':int(rid),'slot':k,'view_ids':v[ix[sel]].tolist(),'observation_indices':ix[sel].tolist(),'n_before_rofa':int(len(orig)),'n_after_rofa':int(len(sel)),'rofa_mode':cfg.rofa_mode})
        if groups:
            mass[z] /= mass[z].sum()
            count[z] = len(groups)
        else:
            # Keep the bank loader's row-stochastic invariant without inventing
            # an auxiliary mode: slot 0 is a finite dummy, and remains invalid.
            mass[z, 0] = 1.0
            feat[z, 0] = 0.0
            res[z, 0] = 0.0
            count[z] = 0
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    obs_digest=hashlib.sha256(Path(obs).read_bytes()).hexdigest()
    metadata={'observation_sha256':obs_digest,'nag_sha256':nag_sha256,'source_observation':str(Path(obs).resolve()),'config':cfg.__dict__,'effective_geometry':bool(cfg.geo_weight>0 and ptr is not None),'b1_enabled':cfg.b1,'b2_enabled':cfg.b2,'dropped_observation_count':len(dropped),'rcs_seed_audit':rcs_audit}
    np.savez_compressed(path,features=feat,residuals=res,priors=mass,valid=valid,base_features=base,group_count=count,region_ids=regions,metadata=json.dumps(metadata,sort_keys=True))
    path.with_suffix('.members.json').write_text(json.dumps({'metadata':metadata,'memberships':memberships},indent=2),encoding='utf8')
    return metadata
