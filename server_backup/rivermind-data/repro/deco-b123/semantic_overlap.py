#!/usr/bin/env python3
import numpy as np
try:
    from numba import njit, prange
except Exception:
    njit = None

def _validate(ids, weights):
    ids=np.asarray(ids); weights=np.asarray(weights,dtype=np.float64)
    if ids.ndim!=1 or weights.ndim!=1 or len(ids)!=len(weights): raise ValueError('1D ids/weights with equal length required')
    if len(ids) and (not np.issubdtype(ids.dtype,np.integer) or np.any(np.diff(ids)<=0)): raise ValueError('ids must be sorted and unique')
    if not np.isfinite(weights).all() or np.any(weights<0): raise ValueError('weights must be finite and non-negative')
    return ids.astype(np.int64,copy=False),weights

def weighted_jaccard_csr(ids_a, weights_a, ids_b, weights_b):
    a,wa=_validate(ids_a,weights_a); b,wb=_validate(ids_b,weights_b)
    if len(a)==0 or len(b)==0: return 0.0
    ia=ib=0; num=den=0.0
    while ia<len(a) and ib<len(b):
        if a[ia]==b[ib]: num+=min(wa[ia],wb[ib]); den+=max(wa[ia],wb[ib]); ia+=1; ib+=1
        elif a[ia]<b[ib]: den+=wa[ia]; ia+=1
        else: den+=wb[ib]; ib+=1
    while ia<len(a): den+=wa[ia]; ia+=1
    while ib<len(b): den+=wb[ib]; ib+=1
    return 0.0 if den<=0 else float(num/den)

if njit is not None:
    @njit(parallel=True, cache=True)
    def pairwise_weighted_jaccard(obs_indices, offsets, ids, weights):
        n=len(obs_indices); out=np.zeros((n,n),np.float32)
        for i in prange(n):
            ai=obs_indices[i]; a0=offsets[ai]; a1=offsets[ai+1]
            out[i,i]=1.0 if a1>a0 else 0.0
            for j in range(i+1,n):
                bj=obs_indices[j]; ib0=offsets[bj]; ib1=offsets[bj+1]
                ia=a0; ib=ib0; num=0.0; den=0.0
                while ia<a1 and ib<ib1:
                    ga=ids[ia]; gb=ids[ib]
                    if ga==gb:
                        wa=weights[ia]; wb=weights[ib]
                        num += wa if wa<wb else wb; den += wa if wa>wb else wb; ia+=1; ib+=1
                    elif ga<gb: den+=weights[ia]; ia+=1
                    else: den+=weights[ib]; ib+=1
                while ia<a1: den+=weights[ia]; ia+=1
                while ib<ib1: den+=weights[ib]; ib+=1
                out[i,j]=0.0 if den<=0 else num/den; out[j,i]=out[i,j]
        return out
else:
    pairwise_weighted_jaccard=None
