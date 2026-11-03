import ast
import hashlib
import numpy as np

def validate_features(seg,feat):
    ids=seg[seg>=0]
    if feat.ndim!=2 or feat.shape[1]!=512 or not np.isfinite(feat).all(): raise ValueError('invalid features')
    if len(ids) and ids.max()>=len(feat): raise ValueError('seg feature ID out of bounds')

def validate_labels(labels,k,n):
    if labels.shape!=(n,) or labels.min()<0 or labels.max()>=k: raise ValueError('invalid NAG labels')
    if len(np.unique(labels))!=k: raise ValueError('noncontiguous NAG')

def check_budget(free,planned):
    if planned>3*2**30 or free-planned<10*2**30: raise ValueError('disk budget exceeded')

def geometry_groups(cameras):
    # Pre-registered, query/GT-free: greedy anchor, 2% camera trajectory diagonal and 5 degrees.
    centers=np.asarray([c['position'] for c in cameras])
    scale=float(np.linalg.norm(centers.max(0)-centers.min(0)))
    threshold=max(scale*.02,1e-9)
    rotations=np.asarray([c['rotation'] for c in cameras])
    anchors=[];ids=[]
    for i in range(len(cameras)):
        group=None
        for g,j in enumerate(anchors):
            angle=np.degrees(np.arccos(np.clip((np.trace(rotations[i].T@rotations[j])-1)/2,-1,1)))
            if np.linalg.norm(centers[i]-centers[j])<=threshold and angle<=5:
                group=g;break
        if group is None: group=len(anchors);anchors.append(i)
        ids.append(group)
    return {'rule':'sorted-view greedy-anchor; position <= .02 camera trajectory diagonal; rotation <= 5deg',
            'position_threshold':threshold,'ids':ids,'anchors':anchors,
            'sizes':np.bincount(ids).tolist()}

def audit_calls(source):
    forbidden={'assign_label','assign_seg_label_torch','merge_small_segs_torch','get_seg_adjacency','everything_seg','init_data'}
    for n in ast.walk(ast.parse(source)):
        if isinstance(n,ast.Call):
            name=n.func.attr if isinstance(n.func,ast.Attribute) else getattr(n.func,'id','')
            if name in forbidden: raise ValueError('forbidden construction call: '+name)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
