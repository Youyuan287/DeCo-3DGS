"""Only K-dimensional evidence fusion. C1/C2 and hierarchy come from complete DeCo."""
from __future__ import annotations
import hashlib,json,numpy as np,torch

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for z in iter(lambda:f.read(1<<20),b''):h.update(z)
    return h.hexdigest()

def load_bank(path,nag_path,root_count,D):
    with np.load(path,allow_pickle=False) as d:
        req=['features','priors','valid','base_features','region_ids','metadata']
        for k in req:
            if k not in d:raise ValueError('missing bank key '+k)
        bank={k:d[k].copy() for k in req if k!='metadata'}
        meta=json.loads(str(d['metadata'].item()))
    x=bank['features'];p=bank['priors'];val=bank['valid'].astype(bool);rid=bank['region_ids']
    if x.ndim!=3 or x.shape[-1]!=D or p.shape!=x.shape[:2] or val.shape!=p.shape:raise ValueError('shape mismatch')
    if len(np.unique(rid))!=len(rid) or (rid<0).any() or (rid>=root_count).any():raise ValueError('bad region ids')
    if not val.any(axis=1).all():raise ValueError('empty region evidence')
    if not np.isfinite(x).all() or not np.isfinite(p).all() or (p<0).any():raise ValueError('bad feature/priors')
    if not np.allclose(p.sum(1),1,atol=1e-4):raise ValueError('priors do not sum to one')
    if meta.get('nag_sha256')!=sha256(nag_path):raise ValueError('NAG SHA256 mismatch: do not compare results')
    return bank,meta

def combine(slot_scores,priors,valid,mode='fixed',temperature=.08):
    """Fixed: query-independent priors; adaptive: query-conditioned reweighting.
    Does not score images or modify hierarchical selection.
    """
    if mode not in ('fixed','adaptive'):raise ValueError(mode)
    if temperature<=0:raise ValueError('temperature')
    if not bool(valid.any(1).all()):raise ValueError('no valid mode')
    if mode=='fixed':
        w=priors*valid
        w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
    else:
        logits=slot_scores/temperature+torch.log(priors.clamp_min(1e-12))
        logits=logits.masked_fill(~valid,-1e9)
        w=torch.softmax(logits,1)
    return (w*slot_scores).sum(1)

def replace_root(sims,bank,slot_scores,b3=False,temperature=.08):
    """Only replace indexed top-level scores; missing regions keep SAME DeCo baseline score."""
    root=sims[-1].clone();device=root.device
    ids=torch.as_tensor(bank['region_ids'],device=device,dtype=torch.long)
    priors=torch.as_tensor(bank['priors'],device=device,dtype=root.dtype)
    valid=torch.as_tensor(bank['valid'],device=device,dtype=torch.bool)
    if slot_scores.shape!=priors.shape:raise ValueError('slot shape mismatch')
    # B3 query-conditioned vs B1/B2 prior-only score combination
    root[ids]=combine(slot_scores,priors,valid,'adaptive' if b3 else 'fixed',temperature)
    out=list(sims);out[-1]=root
    return out

def descriptors(bank, device, center=None,renorm=False):
    x=torch.from_numpy(bank['features']).to(device=device,dtype=torch.float32)
    if center is not None:
        x=x-center.reshape(1,1,-1)
        if renorm:x=torch.nn.functional.normalize(x,dim=-1)
    return x.reshape(-1,x.shape[-1])
