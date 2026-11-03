import sys,os,glob,numpy as np,torch
sys.path[:0]=["/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet","/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"]
from eval_scannet_3d import *
import score_scannet as published
S="scene0062_00"; D="/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
nag=torch.load("/root/verify58_20261029/full10/"+S+"/model/sai_nag_3.0.pt",map_location="cpu")
lab=nag["nag"][-1].numpy().astype(np.int64); n=int(lab.max())+1
cls=SUBSETS[19]; names=[NYU40[c] for c in cls]
T=text_features(names,model_name="ViT-B-16",pretrained="laion2b_s34b_b88k",prompts=published.PROMPTS).cpu().numpy().astype(np.float32)
T=T/np.maximum(np.linalg.norm(T,axis=1,keepdims=True),1e-8)
R=np.load(D+"/ablation/v2_off_off/"+S+"/predictions_and_scores.npz")["scores_off"].astype(np.float64)
print("n_root",n,"R",R.shape,"T",T.shape)
o=np.load(D+"/"+S+"/export/observations_level3.npz")
F=o["features"].astype(np.float32); F=F/np.maximum(np.linalg.norm(F,axis=1,keepdims=True),1e-8)
reg=o["region_ids"].astype(np.int64); vw=o["view_ids"].astype(np.int64)
print("reg max",reg.max(),"views",len(np.unique(vw)))
Sx=F@T.T; key=reg*100000+vw; uk,inv=np.unique(key,return_inverse=True)
sums=np.zeros((len(uk),len(cls))); np.add.at(sums,inv,Sx); cnt=np.bincount(inv).astype(float); sv=sums/cnt[:,None]
vr=(uk//100000).astype(np.int64)
E=np.zeros((n,len(cls)))
for c in range(len(cls)):
    vals=sv[:,c]; order=np.lexsort((-vals,vr)); vr_s=vr[order]; vs=vals[order]
    st=np.r_[0,np.flatnonzero(np.diff(vr_s))+1]; ct=np.diff(np.r_[st,len(vr_s)])
    rank=np.arange(len(vr_s))-np.repeat(st,ct); kk=rank<3
    a=np.bincount(vr_s[kk],weights=vs[kk],minlength=n); b=np.bincount(vr_s[kk],minlength=n)
    E[:,c]=a/np.maximum(b,1)
z=lambda x:(x-x.mean(0))/(x.std(0)+1e-8)
zR=z(R); base=zR+0.5*z(E)
print("E colmean",np.round(E.mean(0)[:5],4),"E std",np.round(E.std(0)[:5],4))
print("zE colmean",np.round(z(E).mean(0)[:5],6))
print("base colmean",np.round(base.mean(0)[:5],4))
print("nan",np.isnan(base).sum(),"inf",np.isinf(base).sum())
print("argmax base[:5]",base.argmax(1)[:5],"argmax zR[:5]",zR.argmax(1)[:5])
