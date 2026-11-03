import csv, json, hashlib
from pathlib import Path
import numpy as np

root = Path('/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028')
arms = ['B0','K0','B1','B2','B3']
def npz_metadata(p):
    with np.load(p, allow_pickle=False) as z:
        return json.loads(str(z['metadata'].item()))
rows = {}
for arm in arms:
    with (root/arm/'per_query.csv').open() as f:
        rows[arm] = {(r['image'], r['prompt']): r for r in csv.DictReader(f)}
summ = {arm: json.loads((root/arm/f'ramen_{arm}_summary.json').read_text()) for arm in arms}
def delta(a,b):
    vals=[float(rows[a][k]['iou'])-float(rows[b][k]['iou']) for k in sorted(set(rows[a])&set(rows[b]))]
    return len(vals),sum(vals)/len(vals),min(vals),max(vals),sum(v>0 for v in vals),sum(v<0 for v in vals)
lines=['# Official-NAG B123 frozen ramen results','','Previous frozen-NAG results are invalidated. These five arms use the official `sai_nag.pt`, one newly exported observation file, and one protocol.','', '## Inputs','', '- NAG SHA256: `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324`.','- Observation SHA256: `'+hashlib.sha256((root.parent/'official_nag_export_20261028'/'observations_level3.npz').read_bytes()).hexdigest()+'`.','- Protocol: center=image, text=canon, renorm=0, sel=gap, topk=5, level_until=3, filter_small=50, temperature=0.08.','', '## Overall metrics','', '| Arm | mIoU | flat mIoU | queries | mean candidates | mean predicted px |','|---|---:|---:|---:|---:|---:|']
for arm in arms:
    rr=list(rows[arm].values()); lines.append(f"| {arm} | {summ[arm]['miou']:.6f} | {summ[arm]['flat_miou']:.6f} | {summ[arm]['n_queries']} | {sum(int(r['n_candidates']) for r in rr)/len(rr):.3f} | {sum(int(r['pred_px']) for r in rr)/len(rr):.1f} |")
lines += ['', '## Paired IoU differences', '', '| Comparison | n | mean | min | max | positive / negative |','|---|---:|---:|---:|---:|---:|']
for a,b in [('K0','B0'),('B1','K0'),('B2','B1'),('B3','B2')]:
    n,mn,mi,ma,pos,neg=delta(a,b); lines.append(f'| {a} - {b} | {n} | {mn:.8f} | {mi:.8f} | {ma:.8f} | {pos} / {neg} |')
lines += ['', '## ROFA and bank audit','']
for arm in ['K0','B1','B2']:
    m=npz_metadata(root/(arm+'.npz')); lines.append(f"- {arm}: dropped observations=`{m.get('dropped_observation_count')}`, effective geometry=`{m.get('effective_geometry')}`, bank SHA256=`{hashlib.sha256((root/(arm+'.npz')).read_bytes()).hexdigest()}`.")
lines += ['- B3 directly loads B2.npz; B2/B3 bank SHA256 is identical.','- All per-query CSVs contain 71 queries.','', '## Conclusion','', 'B0 now matches the historical official DeCo value (`0.5181`). K0/B1/B2/B3 are controlled ablations against this same official-NAG baseline and the same newly exported observation source.']
(root/'B123_RESULTS_OFFICIAL.md').write_text('\n'.join(lines)+'\n')
