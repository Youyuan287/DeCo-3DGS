"""CPU-only copy of observed ROFA; original-source execution uses isolated method AST."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

SOURCE=Path('/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/merge_proj.py')

def rofa(features,tau=3.):
    if features.device.type!='cpu': raise ValueError('CPU only')
    feat_dim=features.shape[1]
    info={'fallback':False}
    if features.shape[0]==0: return torch.zeros(feat_dim,device='cpu'),info
    if features.shape[0]==1: return F.normalize(features[0],p=2,dim=-1),info
    cos_sim=F.cosine_similarity(features.unsqueeze(1),features.unsqueeze(0),dim=-1)
    mean_sim=(cos_sim.sum(dim=1)-1)/(cos_sim.shape[0]-1)
    mu,sigma=mean_sim.mean(),mean_sim.std()
    keep_mask=mean_sim>(mu-tau*sigma)
    if keep_mask.sum()==0:
        keep_mask[mean_sim.argmax()]=True
        info['fallback']=True
    info.update(mean_sim=mean_sim,mu=mu,sigma=sigma,keep=keep_mask)
    return F.normalize(features[keep_mask].mean(dim=0),p=2,dim=-1),info

def rofa_identity(features,source_ids,tau=3.):
    if len(features)!=len(source_ids): raise ValueError('source length mismatch')
    seen={}
    indices=[]
    for i,key in enumerate(source_ids):
        if key in seen:
            if not torch.equal(features[i],features[seen[key]]):
                raise ValueError('inconsistent source feature')
        else:
            seen[key]=i
            indices.append(i)
    out,info=rofa(features[indices],tau)
    info['unique_indices']=indices
    return out,info

def load_original():
    tree=ast.parse(SOURCE.read_text())
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_filter_feature_outliers']
    if len(matches)!=1: raise RuntimeError('ambiguous original source')
    method=matches[0]
    method.decorator_list=[]
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace={'torch':torch,'F':F}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    function=namespace['_filter_feature_outliers']
    return lambda x,tau: function(SimpleNamespace(args=SimpleNamespace(tau=tau)),x,device='cpu')
