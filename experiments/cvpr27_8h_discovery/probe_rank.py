"""Rank, don't regress: does a within-query ranker recover the better candidate?

Ridge-on-IoU underperformed; features still have high per-query AUC. So test a
LOSO classifier on the 'is the best candidate' label, plus fixed training-free
rank sums, all on within-query percentile features.
"""
import csv, glob, os
import numpy as np
from sklearn.linear_model import LogisticRegression

G="analysis/results/20261101/cnone_gate_tf"; EV="analysis/results/20261101/cnone_ev"
FIELDS=['score','deficit','id_mean','id_top3','agree','agree_top','id_mean_c','agree_c',
        'n_points','area_ratio','cover_frac','cover_views','mwp_conf','vis_frac','mean_vis',
        'mem_frac','mean_mem','bbox_diag','flatness','height','obs_views','obs_sim_mean',
        'obs_sim_std','obs_sim_max']
ev={}
for f in glob.glob(os.path.join(EV,'*_evidence.csv')):
    for r in csv.DictReader(open(f)):
        ev[(r['scene'],r['image'],r['prompt'],int(r['rank']))]=r
q={}
for f in sorted(glob.glob(os.path.join(G,'*_segidentity.csv'))):
    for r in csv.DictReader(open(f)):
        k=(r['scene'],r['image'],r['prompt']); e=ev.get((k[0],k[1],k[2],int(r['rank'])),{})
        d={'iou':float(r['iou']),'rank':int(r['rank'])}
        for c in FIELDS:
            v=r.get(c) or e.get(c)
            try: d[c]=float(v)
            except: d[c]=np.nan
        q.setdefault(k,[]).append(d)

def pct(v):
    v=np.asarray(v,float); ok=np.isfinite(v); o=np.zeros_like(v)
    if ok.sum()<=1: return o
    o[ok]=v[ok].argsort().argsort()/max(ok.sum()-1,1)
    if (~ok).any(): o[~ok]=0.5
    return o

X=[];y=[];meta=[]
for k,rs in q.items():
    rs=sorted(rs,key=lambda z:-z['iou'])
    best=rs[0]['iou']
    cols={c:pct([z[c] for z in rs]) for c in FIELDS}
    for i,z in enumerate(rs):
        X.append([cols[c][i] for c in FIELDS]); y.append(1.0 if z['iou']>=best-1e-9 else 0.0)
        meta.append((k[0],k[1],k[2],z['rank'],z['iou']))
X=np.array(X);y=np.array(y)
print("rows",len(X),"pos",int(y.sum()))
scenes=sorted({m[0] for m in meta})
iou={(m[0],m[1],m[2],m[3]):m[4] for m in meta}
best_of={}
for m in meta:
    k=(m[0],m[1],m[2]); best_of[k]=max(best_of.get(k,-1),m[4])
def score(sel):
    per={}
    for k,rk in sel.items(): per.setdefault(k[0],{}).setdefault(k[1],[]).append(iou[(k[0],k[1],k[2],rk)])
    return float(np.mean([np.mean([np.mean(v) for v in d.values()]) for d in per.values()]))
sel={}
for s in scenes:
    tr=np.array([m[0]!=s for m in meta]); te=~tr
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(X[tr],y[tr])
    p=clf.predict_proba(X[te])[:,1]; j=0; qs={}
    for i,m in enumerate(meta):
        if not te[i]: continue
        k=(m[0],m[1],m[2])
        if k not in qs or p[j]>qs[k][0]: qs[k]=(p[j],m[3])
        j+=1
    sel.update({k:v[1] for k,v in qs.items()})
print("%-26s %.4f"%("LOSO logistic (all)",score(sel)))

# training-free fixed rank sums over the strongest families
GROUPS={
 "rank-only":['score'],
 "geom":['area_ratio','n_points','cover_views','obs_views','mean_vis'],
 "maskcons":['mwp_conf','agree','agree_top','cover_frac'],
 "evidence":['obs_sim_max','obs_sim_mean','obs_sim_std'],
 "id":['id_top3','id_mean_c'],
}
GROUPS["all-fixed"]=[c for g in GROUPS.values() for c in g]
for name,cols in GROUPS.items():
    ci=[FIELDS.index(c) for c in cols]
    s2={}
    for i,m in enumerate(meta):
        k=(m[0],m[1],m[2]); v=X[i][ci].mean()
        if k not in s2 or v>s2[k][0]: s2[k]=(v,m[3])
    print("%-26s %.4f"%("fixed rank-sum "+name,score({k:v[1] for k,v in s2.items()})))
print("%-26s %.4f"%("rank1",score({k:1 for k in {(m[0],m[1],m[2]) for m in meta}})))
print("%-26s %.4f"%("oracle",float(np.mean([best_of[k] for k in best_of]))))
