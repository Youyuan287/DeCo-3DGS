"""Does the text score carry ANY information about extent (which is what IoU measures)?

Model: per-pixel features inside an object are mean mu + noise. A region's score is
therefore mu-dependent only, so two nested regions differ by noise O(sigma/sqrt(n))
while their IoU differs by a lot. Test: within a query, is Delta-score between two
candidates predictive of Delta-IoU? If the correlation is ~0, no score-only rule can
recover extent, and every calibration/contrast/bias method is bounded by that.
"""
import csv, glob, os
import numpy as np
from scipy.stats import spearmanr, pearsonr

G="analysis/results/20261101/cnone_gate_tf"; EV="analysis/results/20261101/cnone_ev"
ev={}
for f in glob.glob(os.path.join(EV,'*_evidence.csv')):
    for r in csv.DictReader(open(f)):
        ev[(r['scene'],r['image'],r['prompt'],int(r['rank']))]=r
q={}
for f in sorted(glob.glob(os.path.join(G,'*_segidentity.csv'))):
    for r in csv.DictReader(open(f)):
        k=(r['scene'],r['image'],r['prompt']); e=ev.get((k[0],k[1],k[2],int(r['rank'])),{})
        def g(c):
            v=r.get(c) or e.get(c)
            try: return float(v)
            except: return np.nan
        q.setdefault(k,[]).append(dict(rank=int(r['rank']),iou=g('iou'),score=g('score'),
                                       area=g('area_ratio'),npts=g('n_points')))
# all within-query pairs
ds,di,da=[],[],[]
for k,v in q.items():
    for i in range(len(v)):
        for j in range(i+1,len(v)):
            a,b=v[i],v[j]
            if not all(np.isfinite([a['score'],b['score'],a['iou'],b['iou']])): continue
            ds.append(a['score']-b['score']); di.append(a['iou']-b['iou'])
            if np.isfinite(a['area']) and np.isfinite(b['area']):
                da.append(a['area']-b['area'])
ds,di=np.array(ds),np.array(di)
print("within-query pairs: %d"%len(ds))
print("  corr(Delta score, Delta IoU)  pearson %+.3f  spearman %+.3f"%(
    pearsonr(ds,di)[0], spearmanr(ds,di)[0]))
print("  corr(Delta  area, Delta IoU)  spearman %+.3f"%spearmanr(np.array(da),di[:len(da)])[0])
print("  corr(Delta score, Delta area) spearman %+.3f"%spearmanr(ds[:len(da)],np.array(da))[0])
print("  std(Delta score) %.4f   std(Delta IoU) %.4f"%(ds.std(),di.std()))

# conditioning: within queries whose candidates are all similar in size, does score work?
def nested(sel_iou):
    per={}
    for k,v in q.items():
        per.setdefault(k[0],{}).setdefault(k[1],[]).append(sel_iou[k])
    return float(np.mean([np.mean([np.mean(x) for x in d.values()]) for d in per.values()]))
sel={}
sel_cond={}
for k,v in q.items():
    s=max(v,key=lambda z:(z['score'] if np.isfinite(z['score']) else -9))
    sel[k]=s['iou']
    ar=np.array([z['area'] for z in v],float); ok=np.isfinite(ar)
    if ok.sum()>=2 and ar[ok].max()/max(ar[ok].min(),1e-12)<3.0:
        sel_cond[k]=s['iou']
    else:
        sel_cond[k]=s['iou']
print("\nscore-max selection on ALL queries:      %.4f (n=%d)"%(nested(sel),len(sel)))
# complementary-signal test: score + extent, equal weight (declared a priori, not tuned)
def zs(x):
    x=np.asarray(x,float); ok=np.isfinite(x)
    o=np.zeros_like(x)
    if ok.sum()>1:
        o[ok]=(x[ok]-x[ok].mean())/(x[ok].std()+1e-12)
    return o
print("
score+extent combinations (single-candidate macro):")
for lam in (0.25,0.5,1.0,1.0):
    pass
for lam in (0.25,0.5,1.0):
    sc={}
    for k,v in q.items():
        sv=zs([z['score'] for z in v]); av=zs([np.log(max(z['area'],1e-9)) if np.isfinite(z['area']) else 0.0 for z in v])
        j=int(np.argmax(sv+lam*av)); sc[k]=v[j]['iou']
    print("  z(score)+%.2f*z(log area)  %.4f"%(lam,nested(sc)))
sc={}
for k,v in q.items():
    sv=zs([z['score'] for z in v]); nv=zs([np.log(max(z['npts'],1)) if np.isfinite(z['npts']) else 0.0 for z in v])
    j=int(np.argmax(sv+nv)); sc[k]=v[j]['iou']
print("  z(score)+1.00*z(log npts)  %.4f"%nested(sc))
