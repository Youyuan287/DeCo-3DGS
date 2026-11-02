"""Is the per-view decision signal informative only in the TAIL?

E14 shows per-view operators lose as unconditional selectors. The audit still
gains +3.13 by overriding only when the paired comparison is decisive. Test the
mechanism directly: override rank1 with the max-margin candidate only when its
margin exceeds rank1's by more than t; measure hit rate and macro as t grows.
"""
import csv, glob, os
import numpy as np
W="analysis/results/20261101/witness"; G="analysis/results/20261101/cnone_gate_tf"
wit={}
for f in glob.glob(os.path.join(W,"*_witness.csv")):
    sc=os.path.basename(f).split("_witness")[0]
    for r in csv.DictReader(open(f)):
        wit.setdefault((sc,r["image"],r["prompt"]),{}).setdefault(int(r["rank"]),{})[int(r["view_id"])]=float(r["sim"])
iou={}
for f in glob.glob(os.path.join(G,"*_segidentity.csv")):
    for r in csv.DictReader(open(f)):
        iou[(r["scene"],r["image"],r["prompt"],int(r["rank"]))]=float(r["iou"])
def marg(pr,r,ranks):
    mg=[]
    for v,s in pr[r].items():
        o=max([pr[x].get(v,-9) for x in ranks if x!=r] or [-9]); mg.append(s-o)
    return float(np.mean(mg)) if mg else -9.0
def macro(sel):
    per={}
    for k,rk in sel.items(): per.setdefault(k[0],{}).setdefault(k[1],[]).append(iou[(k[0],k[1],k[2],rk)])
    return float(np.mean([np.mean([np.mean(v) for v in d.values()]) for d in per.values()]))
qs=[]
for k,pr in wit.items():
    rs=sorted(pr)
    if len(rs)>=2: qs.append((k,rs,pr))
base={k:min(rs) for k,rs,_ in qs}
print("base (rank1)                 %.4f"%macro(base))
print("%10s %6s %7s %7s"%("t","fires","hit%","macro"))
for t in (0.0,0.005,0.01,0.02,0.04,0.08,0.15):
    sel=dict(base); fire=hit=0
    for k,rs,pr in qs:
        m={r:marg(pr,r,rs) for r in rs}
        top=max(rs,key=lambda r:m[r])
        inc=rs[0]
        if top!=inc and m[top]-m[inc]>t:
            fire+=1; sel[k]=top
            hit += iou[(k[0],k[1],k[2],top)]>iou[(k[0],k[1],k[2],inc)]
    print("%10.3f %6d %7.1f %7.4f"%(t,fire,100*hit/max(fire,1),macro(sel)))
