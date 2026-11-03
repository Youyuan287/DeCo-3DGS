"""Aggregate a V2 ablation run: pooled/macro per arm, per-scene mIoU, changed roots, correction vs decision margin."""
import sys,json,numpy as np,pandas as pd
B="/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
RUN=sys.argv[1]
S="scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00".split()
EXP_DECO=dict(zip(S,[51.28,82.50,35.78,43.38,51.98,49.77,43.47,44.54,52.17,46.92]))
arms=["off","m1_fixed","m1_m2_fixed","m1_m3_adaptive","m1_m2_m3_adaptive"]
inter={a:{} for a in arms};union={a:{} for a in arms};per={}
chg={a:[0,0] for a in arms[1:]};marg=[];corr=[];rel=[];NR=0
for s in S:
  R=f"{B}/ablation/{RUN}/{s}";sm=json.load(open(R+"/SUMMARY.json"))
  per[s]=[sm["metrics"][a]["mIoU"]*100 for a in arms]
  for a in arms:
    for c,v in sm["metrics"][a]["per_class"].items():
      inter[a][c]=inter[a].get(c,0)+v["inter"];union[a][c]=union[a].get(c,0)+v["union"]
  z=np.load(R+"/predictions_and_scores.npz");off=z["scores_off"];po=off.argmax(1);NR+=len(po)
  srt=np.sort(off,1);marg.append(srt[:,-1]-srt[:,-2])
  for a in arms[1:]:
    ch=np.flatnonzero(z["scores_"+a].argmax(1)!=po);chg[a][0]+=len(ch);chg[a][1]+=int(np.isin(z["labels"],ch).sum())
  t=pd.read_csv(R+"/SCANNET_M3_WEIGHT_TRACE.csv",usecols=["arm","eligible","reliability","correction"])
  corr.append(t[t.arm=="m1_m3_adaptive"].correction.values);rel.append(t[(t.arm=="m1_m2_fixed")&t.eligible].reliability.values)
print("scene | "+" | ".join(arms)+" | certified DeCo")
for s in S:print(s,"|"," | ".join("%.4f"%v for v in per[s]),"| %.2f"%EXP_DECO[s])
print()
for a in arms:
  p=np.mean([inter[a][c]/union[a][c] for c in inter[a] if union[a][c]>0])*100;m=np.mean([per[s][arms.index(a)] for s in S])
  print(f"{a:20s} pooled {p:.4f} macro {m:.4f}")
print("changed roots/gaussians vs off (of",NR,"roots):",chg)
m=np.concatenate(marg);c=np.abs(np.concatenate(corr))
print("top1-top2 margin q10/50/90 %s"%np.quantile(m,[.1,.5,.9]).round(4).tolist())
print("|correction| (adaptive) q50/90/99/max %s"%np.quantile(c,[.5,.9,.99,1]).round(4).tolist())
print("reliability q10/50/90 %s"%np.quantile(np.concatenate(rel),[.1,.5,.9]).round(4).tolist())
