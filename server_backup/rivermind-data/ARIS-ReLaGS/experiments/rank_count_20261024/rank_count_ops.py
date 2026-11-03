"""Frozen-pool rank/count intervention; no GT or GPU dependencies."""
import math
from support_ops import gap_select


def build_cells(record):
    pool=record['pool']
    scores=[record['modes'][m]['scores'] for m in ('baseline','spatial')]
    for ss in scores:
        if len(ss)!=len(pool) or not all(math.isfinite(s) for s in ss):
            raise ValueError('Invalid frozen score vector')
    selected=[gap_select(pool,ss) for ss in scores]
    for mode,chosen in zip(('baseline','spatial'),selected):
        if chosen!=record['modes'][mode]['selected']:
            raise ValueError('Original selector replay differs: '+mode)
    order0=sorted(range(len(pool)),key=lambda i:scores[0][i],reverse=True)
    order1=sorted(range(len(pool)),key=lambda i:scores[1][i],reverse=True)
    k0,k1=map(len,selected)
    return {'baseline':selected[0], 'spatial':selected[1],
            'rank1_count0':[pool[i] for i in order1[:k0]],
            'rank0_count1':[pool[i] for i in order0[:k1]]}
