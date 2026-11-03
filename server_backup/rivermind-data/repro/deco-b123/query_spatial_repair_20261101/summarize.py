import csv,json
from pathlib import Path
import numpy as np

root=Path(__file__).parent
rows=[]
for s in ['figurines','ramen','teatime','waldo_kitchen']:
    with (root/'results'/f'{s}_spatial.csv').open() as f:
        for r in csv.DictReader(f):
            for k in r:
                if k.endswith('_iou') or k=='baseline_error' or k.endswith(('_views','_added','_removed')):r[k]=float(r[k])
            rows.append(r)
assert len(rows)==208,len(rows)
keys=['A_iou','B_iou','C_iou','D_iou','oracle_ABC_iou']
scenes=sorted({r['scene'] for r in rows})
def macro(rs,k):
    return float(np.mean([np.mean([np.mean([r[k] for r in rs if r['scene']==s and r['image']==im]) for im in sorted({r['image'] for r in rs if r['scene']==s})]) for s in scenes]))
summary=dict(n=len(rows),baseline_error_max=max(r['baseline_error'] for r in rows),macro={k:macro(rows,k) for k in keys},scenes={},families={},comparisons={})
for s in scenes:
    sub=[r for r in rows if r['scene']==s]
    images=sorted({r['image'] for r in sub})
    summary['scenes'][s]={k:float(np.mean([np.mean([r[k] for r in sub if r['image']==im]) for im in images])) for k in keys}
for fam in ['found','rank2','deep','recall']:
    sub=[r for r in rows if r['family']==fam]
    summary['families'][fam]=dict(n=len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in keys})
for first,second in [('B','A'),('C','A'),('D','A'),('C','B'),('C','D')]:
    differences=np.array([r[first+'_iou']-r[second+'_iou'] for r in rows])
    means=[]
    for s in scenes:
        sub=[r for r in rows if r['scene']==s]
        means.append(np.array([np.mean([r[first+'_iou']-r[second+'_iou'] for r in sub if r['image']==im]) for im in sorted({r['image'] for r in sub})]))
    rng=np.random.default_rng(20261030)
    boots=[np.mean([np.mean(a[rng.integers(0,len(a),len(a))]) for a in means]) for _ in range(2000)]
    summary['comparisons'][first+'-'+second]=dict(delta=summary['macro'][first+'_iou']-summary['macro'][second+'_iou'],ci95=np.percentile(boots,[2.5,97.5]).tolist(),wins=int((differences>1e-8).sum()),losses=int((differences< -1e-8).sum()),ties=int((np.abs(differences)<=1e-8).sum()))
summary['edits']={tag:dict(mean_views=float(np.mean([r[tag+'_views'] for r in rows])),queries_with_edits=sum(r[tag+'_added']+r[tag+'_removed']>0 for r in rows),mean_added=float(np.mean([r[tag+'_added'] for r in rows])),mean_removed=float(np.mean([r[tag+'_removed'] for r in rows]))) for tag in 'BCD'}
c=summary['comparisons']['C-A'];
summary['pass']=bool(c['delta']>=.01 and c['ci95'][0]>0 and all(v['C_iou']-v['A_iou']>=-.01 for v in summary['scenes'].values()) and summary['families']['found']['C_iou']-summary['families']['found']['A_iou']>=-.005 and summary['macro']['C_iou']>max(summary['macro']['B_iou'],summary['macro']['D_iou']))
(root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,indent=2,ensure_ascii=False))
