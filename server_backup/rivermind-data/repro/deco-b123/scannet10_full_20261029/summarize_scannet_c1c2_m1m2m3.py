import json, os, csv
from pathlib import Path

base=Path(os.environ.get('SCANNET_C1C2_ROOT','/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/c1c2_ablation_20261029'))
out=Path(os.environ.get('SCANNET_C1C2_REPORT',str(base)))
modes=['off_off','c1_on_c2_off','c1_off_c2_on','c1_on_c2_on']
scenes=['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0200_00','scene0347_00','scene0400_00','scene0590_00','scene0645_00']
arms=['off','m1_fixed','m1_m2_fixed','m1_m3_adaptive','m1_m2_m3_adaptive']
rows=[]
for mode in modes:
 for scene in scenes:
  p=base/mode/scene/'SUMMARY.json'
  if not p.exists(): continue
  d=json.loads(p.read_text())
  for arm in arms:
   rows.append({'mode':mode,'scene':scene,'arm':arm,'mIoU':d['metrics'][arm]['mIoU'],'mAcc':d['metrics'][arm]['mAcc'],'OA':d['metrics'][arm]['OA']})
out.mkdir(parents=True,exist_ok=True)
with (out/'SCANNET_C1C2_M1M2M3_PER_SCENE.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def mean(mode,arm,key='mIoU'):
 z=[r[key] for r in rows if r['mode']==mode and r['arm']==arm]
 return sum(z)/len(z) if z else float('nan')
lines=['# ScanNet C1/C2 off baseline 与 M1/M2/M3 消融（重算）','',f'输入目录：`{base}`','', '## 十场 scene macro mIoU','', '| 组合 | DeCo off | M1 fixed | M1+M2 fixed | M1+M3 adaptive | M1+M2+M3 adaptive |','|---|---:|---:|---:|---:|---:|']
for mode in modes:
 lines.append('| '+mode+' | '+' | '.join(f'{mean(mode,a):.6f}' for a in arms)+' |')
lines += ['', '## 严格 C1/C2 主路径消融（只看 off arm）','', '| 条件 | scene macro mIoU | 相对 C1 off/C2 off |','|---|---:|---:|']
b=mean('off_off','off')
for mode,label in [('off_off','C1 off / C2 off'),('c1_on_c2_off','C1 on / C2 off'),('c1_off_c2_on','C1 off / C2 on'),('c1_on_c2_on','C1 on / C2 on')]:
 lines.append(f'| {label} | {mean(mode,"off"):.6f} | {mean(mode,"off")-b:+.6f} |')
lines += ['', '## 解释','', '- `off_off` 是本轮新的严格 baseline：raw Gaussian feature + 单一类别文本分数。', '- C1/C2 四格使用完全相同的 Gaussian、NAG、GT 与最近点映射。', '- M1/M2/M3 只在 `off_off` 这一新 baseline 上解释；旧 `both_lse + renorm=0` 结果不混入本表。', '- 若某一组合缺失，报告将保留缺失而不会用旧结果填充。']
(out/'SCANNET_C1C2_M1M2M3_REVISED.md').write_text('\n'.join(lines),encoding='utf8')
print('WROTE',out/'SCANNET_C1C2_M1M2M3_REVISED.md', 'rows',len(rows))
