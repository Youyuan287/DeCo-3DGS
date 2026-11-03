"""Query-aware multi-evidence region representation; no training or GT required.
Input observations are *production* per-(region,view) features BEFORE ROFA.
Weights must be real production visibility_portion values (NOT binary CSR links).
All directions are computed from development data, never from GT labels.
"""
from dataclasses import dataclass
import numpy as np

EPS = 1e-10

def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)

@dataclass(frozen=True)
class EvidenceConfig:
    max_evidence: int = 3
    min_group_weight: float = 0.05
    temperature: float = 0.08
    # Fixed evidence budget for every region, but no artificial duplication.
    # k=1 recovers the original normalized weighted aggregation if source feature
    # conventions match ReLaGS.

@dataclass
class EvidenceBank:
    features: np.ndarray       # [R,K,D] evidence vectors, padded zeros
    residuals: np.ndarray      # [R,K,D] debiased mode signatures
    priors: np.ndarray         # [R,K], normalized within valid modes
    valid: np.ndarray          # [R,K] boolean
    base_features: np.ndarray  # [R,D], original weighted mean
    group_count: np.ndarray    # [R] integer
    region_ids: np.ndarray     # [R]


def _check_obs(feature, weight, region_id):
    feature = np.asarray(feature, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    region_id = np.asarray(region_id, dtype=np.int64)
    if feature.ndim != 2 or weight.shape != (len(feature),) or region_id.shape != (len(feature),):
        raise ValueError('features [N,D], weights [N], region_ids [N] required')
    if not np.isfinite(feature).all() or not np.isfinite(weight).all():
        raise ValueError('nonfinite observations')
    if (weight < 0).any():
        raise ValueError('negative visibility weight')
    if np.any(np.linalg.norm(feature,axis=1) < EPS):
        raise ValueError('zero input features: inspect production observation creation')
    if len(feature) == 0:
        raise ValueError('empty input')
    return feature, weight, region_id


def build_bank(features, weights, region_ids, shared_direction, cfg=EvidenceConfig(), mode='debiased'):
    """Deterministic weighted farthest-mode selection on normalized residuals.
    mode='raw': diversity selection in raw CLIP space (B2 control).
    mode='debiased': diversity selection in orthogonal residual space (B3).
    Evidence values always aggregate ORIGINAL per-observation vectors, not residuals.
    No GT, text bank, or query participates in mode construction.
    """
    f,w,r = _check_obs(features,weights,region_ids)
    if cfg.max_evidence < 1 or cfg.temperature <= 0 or not (0 <= cfg.min_group_weight < 1):
        raise ValueError('invalid configuration')
    if mode not in {'raw','debiased'}: raise ValueError(mode)
    u = np.asarray(shared_direction,dtype=np.float32)
    if u.shape != (f.shape[1],) or np.linalg.norm(u) < EPS: raise ValueError('invalid shared direction')
    u = l2n(u)
    nf = l2n(f)
    h = l2n(nf - np.outer(nf @ u,u)) if mode == 'debiased' else nf
    # Degenerate residuals: fall back to raw features per observation.
    if mode == 'debiased':
        hnorm = np.linalg.norm(nf - np.outer(nf @ u,u),axis=1)
        h[hnorm < 1e-6] = nf[hnorm < 1e-6]
    ids = np.unique(r)
    R,K,D=len(ids),cfg.max_evidence,f.shape[1]
    out=np.zeros((R,K,D),dtype=np.float32); resid=np.zeros_like(out)
    prior=np.zeros((R,K),dtype=np.float32); valid=np.zeros((R,K),dtype=bool)
    base=np.zeros((R,D),dtype=np.float32); counts=np.zeros(R,dtype=np.int32)
    for z,rid in enumerate(ids):
        idx=np.where((r==rid)&(w>0))[0]
        if not len(idx): raise ValueError(f'region {rid} has zero positive weight')
        a=w[idx]/w[idx].sum(); ff=nf[idx]; hh=h[idx]
        base[z]=l2n((a[:,None]*ff).sum(0))
        # Weighted medoid as seed, then farthest-first (weighted) seeds.
        # For K=1 directly use the full mean, not a single medoid.
        if K == 1 or len(idx)==1:
            out[z,0]=base[z]; resid[z,0]=l2n((a[:,None]*hh).sum(0)); prior[z,0]=1; valid[z,0]=True;counts[z]=1
            continue
        sim=hh@hh.T
        chosen=[int(np.argmax(sim@a))]
        for _ in range(min(K,len(idx))-1):
            d=1-np.max(sim[:,chosen],axis=1)
            d[chosen]=-np.inf
            candidate=int(np.argmax(d))
            if d[candidate] <= 1e-6:break
            chosen.append(candidate)
        assign=np.argmax(sim[:,chosen],axis=1)
        groups=[]
        for j in range(len(chosen)):
            mask=assign==j
            if not mask.any():continue
            mass=float(a[mask].sum())
            if mass < cfg.min_group_weight: continue
            groups.append((mass,l2n((a[mask,None]*ff[mask]).sum(0)),l2n((a[mask,None]*hh[mask]).sum(0))))
        if not groups:
            groups=[(1.,base[z],l2n((a[:,None]*hh).sum(0)))]
        # Groups not retained are not discarded from the fallback base vector.
        groups.sort(key=lambda t:-t[0])
        counts[z]=len(groups)
        masssum=sum(t[0] for t in groups)
        for j,(mass,p,d) in enumerate(groups):
            out[z,j]=p;resid[z,j]=d;prior[z,j]=mass/masssum;valid[z,j]=True
    return EvidenceBank(out,resid,prior,valid,base,counts,ids)


def score_bank(bank, text_features, *, temperature=0.08, readout_center=None):
    """Query-dependent convex evidence scoring, no topK/max reward.
    text_features are the exact certified text bank, already L2 normalized.
    readout_center: explicit development-sourced visual mean [D], or None.
    If provided, the SAME center is applied to modes and the base control.
    Alpha=1 readout must otherwise be provided by certified adapter; no
    claim of bitwise equivalence is made by this helper.
    """
    t=l2n(text_features)
    if temperature<=0: raise ValueError('temperature must be positive')
    x=bank.features.copy(); b=bank.base_features.copy()
    if readout_center is not None:
        c=np.asarray(readout_center,dtype=np.float32)
        if c.shape != (x.shape[-1],):raise ValueError('readout center dimension mismatch')
        x=l2n(x-c[None,None,:]);b=l2n(b-c[None,:])
    raw=x@t.T
    logits=raw/temperature+np.log(np.maximum(bank.priors,EPS))[:,:,None]
    logits=np.where(bank.valid[:,:,None],logits,-1e9)
    logits=logits-np.max(logits,axis=1,keepdims=True)
    prob=np.exp(logits);prob=prob/np.maximum(prob.sum(axis=1,keepdims=True),EPS)
    score=(prob*raw).sum(axis=1)
    return score,b@t.T


def save_npz(path, bank, metadata=None):
    # No pickle. Metadata is provenance only; never put labels or GT inside.
    import json
    np.savez_compressed(path,features=bank.features,residuals=bank.residuals,
        priors=bank.priors,valid=bank.valid,base_features=bank.base_features,
        group_count=bank.group_count,region_ids=bank.region_ids,
        metadata=np.asarray(json.dumps(metadata or {},sort_keys=True)))


def load_npz(path):
    with np.load(path,allow_pickle=False) as d:
        bank=EvidenceBank(*(d[k] for k in ['features','residuals','priors','valid','base_features','group_count','region_ids']))
    return bank


def score_bank_certified_style(bank, text_features, *, temperature=0.08, alpha=1.0):
    """Region-wise counterpart to certified ScanNet readout: normalize X,
    subtract alpha * mean(normalized region baseline), DO NOT re-normalize.
    Evidence branch is a NEW score function, NOT previously certified.
    It is only comparable after K=1 matches frozen alpha=0/1 outputs.
    """
    if not (0 <= alpha <= 1.0):raise ValueError('alpha outside fixed control range')
    if temperature <= 0:raise ValueError('temperature')
    t=l2n(text_features)
    base=l2n(bank.base_features)
    center=base.mean(axis=0)
    ev=l2n(bank.features)
    ev=ev-alpha*center[None,None,:]
    base=base-alpha*center[None,:]
    raw=ev@t.T
    logits=raw/temperature+np.log(np.maximum(bank.priors,EPS))[:,:,None]
    logits=np.where(bank.valid[:,:,None],logits,-1e9)
    logits=logits-np.max(logits,axis=1,keepdims=True)
    prob=np.exp(logits)
    prob=prob/np.maximum(prob.sum(axis=1,keepdims=True),EPS)
    return (prob*raw).sum(axis=1),base@t.T
