"""Score + extent are weakly correlated (0.19) but individually predictive of IoU
(0.42 / 0.46). Test whether the sum beats either -- equal weight, declared a priori."""
import csv, glob, os
import numpy as np
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
def nested(sel):
    per={}
    for k,iou in sel.items():
        per.setdefault(k[0],{}).setdefault(k[1],[]).append(iou)
    return float(np.mean([np.mean([np.mean(x) for x in d.values()]) for d in per.values()]))
def zs(x):
    x=np.asarray(x,float); ok=np.isfinite(x); o=np.zeros_like(x)
    if ok.sum()>1: o[ok]=(x[ok]-x[ok].mean())/(x[ok].std()+1e-12)
    return o
def run(f):
    sel={}
    for k,v in q.items():
        j=int(np.argmax(f(v))); sel[k]=v[j]['iou']
    return nested(sel)
print("rank1                                  %.4f"%nested({k:v[0]['iou'] for k,v in q.items()}))
print("max score (rank1 by construction)      %.4f"%run(lambda v:zs([z['score'] for z in v])))
print("max log-area alone                     %.4f"%run(lambda v:zs([np.log(max(z['area'],1e-9)) if np.isfinite(z['area']) else 0.0 for z in v])))
for lam in (0.25,0.5,1.0):
    print("z(score)+%.2f*z(log area)            %.4f"%(lam,run(lambda v:zs([z['score'] for z in v])+lam*zs([np.log(max(z['area'],1e-9)) if np.isfinite(z['area']) else 0.0 for z in v]))))
print("z(score)+1.00*z(log npts)              %.4f"%run(lambda v:zs([z['score'] for z in v])+zs([np.log(max(z['npts'],1)) if np.isfinite(z['npts']) else 0.0 for z in v])))
print("oracle                                 %.4f"%float(np.mean([max(z['iou'] for z in v) for v in q.values()])))
