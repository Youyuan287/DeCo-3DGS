import csv, json, os
from pathlib import Path
base=Path(os.environ.get('SCANNET_C1C2_ROOT','/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/c1c2_ablation_20261029'))
modes=['off_off','c1_on_c2_off','c1_off_c2_on','c1_on_c2_on']
scenes=['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0200_00','scene0347_00','scene0400_00','scene0590_00','scene0645_00']
arms=['off','m1_fixed','m1_m2_fixed','m1_m3_adaptive','m1_m2_m3_adaptive']
out=[]
for mode in modes:
 for arm in arms:
  sums={}
  for scene in scenes:
   p=base/mode/scene/'SCANNET_M3_PER_CLASS.csv'
   if not p.exists(): continue
   for r in csv.DictReader(p.open()):
    c=r['class_id']; sums.setdefault(c,{'gt':0,'tp':0,'fp':0,'fn':0})
    sums[c]['gt']+=int(r['gt_count'])
    sums[c]['tp']+=int(r[f'{arm}_TP']); sums[c]['fp']+=int(r[f'{arm}_FP']); sums[c]['fn']+=int(r[f'{arm}_FN'])
  vals=[]
  for s in sums.values():
   if s['gt']>0:
    den=s['tp']+s['fp']+s['fn']; vals.append(s['tp']/den if den else 0.0)
  out.append({'mode':mode,'arm':arm,'pooled_mIoU':sum(vals)/len(vals),'present_classes':len(vals)})
Path(base/'SCANNET_C1C2_M1M2M3_POOLED.json').write_text(json.dumps(out,indent=2))
for r in out: print(r['mode'],r['arm'],f"{r['pooled_mIoU']:.6f}",r['present_classes'])
