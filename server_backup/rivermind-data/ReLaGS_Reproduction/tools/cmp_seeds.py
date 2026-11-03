import torch, numpy as np, glob, os, sys
R='/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf'
sc = sys.argv[1] if len(sys.argv)>1 else 'figurines'
pat = sys.argv[2] if len(sys.argv)>2 else '/root/rivermind-data/ReLaGS_Reproduction/work_seed/'+sc+'_s*'
def load(p): return torch.load(p, map_location='cpu')
def feats(o):
    f=o.get('nag_feat')
    if torch.is_tensor(f): f=[f]
    return [x.float().reshape(x.shape[0],-1) for x in f]
def labs(o):
    n=o.get('nag')
    if torch.is_tensor(n): n=[n]
    return [x.reshape(-1).numpy().astype(np.int64) for x in n]
def ari(a,b):
    ka,kb=int(a.max())+1,int(b.max())+1
    if ka*kb>8_000_000: return float('nan')
    c=np.bincount(a*kb+b,minlength=ka*kb).reshape(ka,kb).astype(np.float64)
    ca,cb=c.sum(1),c.sum(0); comb=lambda x:(x*(x-1)/2).sum()
    n=a.size; sn=n*(n-1)/2; exp=comb(ca)*comb(cb)/sn; mx=(comb(ca)+comb(cb))/2
    return float((comb(c)-exp)/(mx-exp)) if mx!=exp else float('nan')
O=load(R+'/'+sc+'/sai_nag.pt'); fo,lo=feats(O),labs(O)
print('official '+sc+': L1/L2/L3 =', [f.shape[0] for f in fo], ' gauss=', lo[0].size)
print('%-26s%-24s%-30s%s' % ('build','L1/L2/L3','cos L1/L2/L3','ARI L1/L2/L3'))
for d in sorted(glob.glob(pat)):
    p=os.path.join(d,'sai_nag_3.0.pt')
    if not os.path.exists(p): continue
    M=load(p); fm,lm=feats(M),labs(M)
    cos=[]
    for a,b in zip(fo,fm):
        if a.shape[0]!=b.shape[0] or a.shape[1]!=b.shape[1]: cos.append(float('nan')); continue
        cn=(b/(b.norm(dim=1,keepdim=True)+1e-8))@(a/(a.norm(dim=1,keepdim=True)+1e-8)).T
        cos.append(float(cn.max(1).values.mean()))
    ar=[ari(a,b) for a,b in zip(lo,lm)] if lo[0].size==lm[0].size else [float('nan')]*3
    print('%-26s%-24s%-30s%s' % (os.path.basename(d), str([f.shape[0] for f in fm]),
          '/'.join('%.3f'%x for x in cos), '/'.join('%.3f'%x for x in ar)))