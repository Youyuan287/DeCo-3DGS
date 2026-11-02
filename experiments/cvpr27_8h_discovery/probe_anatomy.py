"""Anatomy of the missed candidate: what does the better-but-not-chosen candidate look like?

Within-query Spearman of each feature with IoU, plus a paired comparison of
rank1 vs the best candidate. If the missed candidate is systematically smaller /
less consistent / deeper, then every existing prior points the wrong way.
"""
import csv, glob, os
import numpy as np
from scipy.stats import spearmanr

G="analysis/results/20261101/cnone_gate_tf"; EV="analysis/results/20261101/cnone_ev"
F=['score','deficit','id_mean','id_top3','agree','agree_top','id_mean_c','agree_c',
   'n_points','area_ratio','cover_frac','cover_views','mwp_conf','vis_frac','mean_vis',
   'mem_frac','mean_mem','bbox_diag','flatness','height','obs_views','obs_sim_max']
ev={}
for f in glob.glob(os.path.join(EV,'*_evidence.csv')):
    for r in csv.DictReader(open(f)):
        ev[(r['scene'],r['image'],r['prompt'],int(r['rank']))]=r
q={}
for f in sorted(glob.glob(os.path.join(G,'*_segidentity.csv'))):
    for r in csv.DictReader(open(f)):
        k=(r['scene'],r['image'],r['prompt']); e=ev.get((k[0],k[1],k[2],int(r['rank'])),{})
        d={'rank':int(r['rank']),'level':int(r['level']),'iou':float(r['iou'])}
        for c in F:
            v=r.get(c) or e.get(c)
            try: d[c]=float(v)
            except: d[c]=np.nan
        q.setdefault(k,[]).append(d)

# 1) within-query Spearman(feature, IoU)
print("within-query Spearman with IoU (mean over queries, then share of queries with rho>0)")
print("%-13s %7s %7s %7s"%("feature","meanRho","frac+","n"))
for c in F+['level']:
    rs=[]
    for k,v in q.items():
        x=np.array([z[c] for z in v],float); y=np.array([z['iou'] for z in v],float)
        ok=np.isfinite(x)
        if ok.sum()<3 or np.std(y[ok])<1e-9: continue
        rho,_=spearmanr(x[ok],y[ok])
        if np.isfinite(rho): rs.append(rho)
    rs=np.array(rs)
    print("%-13s %7.3f %7.3f %7d"%(c,rs.mean(),(rs>0).mean(),len(rs)))

# 2) paired: rank1 vs best, only where they differ by >0.1 IoU
print("\nmissed candidate (best vs rank1, gap>0.1): paired differences")
print("%-13s %8s %8s %8s"%("feature","rank1","best","best-rank1"))
sel=[]
for k,v in q.items():
    v=sorted(v,key=lambda z:z['rank'])
    r1=v[0]; best=max(v,key=lambda z:z['iou'])
    if best['iou']-r1['iou']>0.1: sel.append((r1,best))
print("n queries:",len(sel))
for c in F+['level','iou']:
    a=np.array([x[c] for x,_ in sel],float); b=np.array([y[c] for _,y in sel],float)
    ok=np.isfinite(a)&np.isfinite(b)
    diff=b[ok]-a[ok]
    # normalise area-like features by rank1 value for interpretability
    print("%-13s %8.3f %8.3f %+8.3f"%(c,np.nanmean(a),np.nanmean(b),np.nanmean(diff)))

# 3) is the missed candidate deeper in the ranked list?
rk=np.array([y['rank'] for _,y in sel])
print("\nmissed candidate's rank distribution: median %d, %%rank>2 = %.2f, %%level<=2 = %.2f"%(
    np.median(rk),(rk>2).mean(),np.mean([y['level']<=2 for _,y in sel])))
