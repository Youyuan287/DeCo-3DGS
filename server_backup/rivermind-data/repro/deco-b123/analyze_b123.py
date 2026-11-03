import csv, json, hashlib
from pathlib import Path

root = Path('/root/rivermind-data/repro/deco-b123/b123_eval_20261028')
arms = ['B0','K0','B1','B2','B3']
def npz_metadata(p):
    import numpy as np
    with np.load(p, allow_pickle=False) as z:
        return json.loads(str(z['metadata'].item()))
rows = {}
for arm in arms:
    with (root/arm/'per_query.csv').open() as f:
        rows[arm] = {(r['image'], r['prompt']): r for r in csv.DictReader(f)}
summ = {}
for arm in arms:
    summ[arm] = json.loads((root/arm/f'ramen_{arm}_summary.json').read_text())

def delta(a,b):
    vals=[]
    for k in sorted(set(rows[a]) & set(rows[b])):
        vals.append(float(rows[a][k]['iou'])-float(rows[b][k]['iou']))
    return len(vals), sum(vals)/len(vals), min(vals), max(vals), sum(v>0 for v in vals), sum(v<0 for v in vals)

lines = ['# B123 frozen ramen results', '', 'All five arms use the same frozen NAG and the newly exported observation file. No historical 304-region bank is used.', '', '## Protocol', '', '- scene: ramen; center=image; text=canon; renorm=0; selection=gap; topk=5; level_until=3; filter_small=50; temperature=0.08.', '- NAG SHA256: `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535`.', '- observation SHA256: `3c857c05d101029fd332fbdbb4e499b4d0c8d1b67a643121a9b7fdb7d32a7fc6`.', '', '## Overall metrics', '', '| Arm | mIoU | flat mIoU | queries | mean candidates | mean predicted px |', '|---|---:|---:|---:|---:|---:|']
for arm in arms:
    rr=list(rows[arm].values())
    lines.append(f"| {arm} | {summ[arm]['miou']:.6f} | {summ[arm]['flat_miou']:.6f} | {summ[arm]['n_queries']} | {sum(int(r['n_candidates']) for r in rr)/len(rr):.3f} | {sum(int(r['pred_px']) for r in rr)/len(rr):.1f} |")
lines += ['', '## Paired IoU differences (first arm minus second arm)', '', '| Comparison | n | mean | min | max | positive / negative |', '|---|---:|---:|---:|---:|---:|']
for a,b in [('K0','B0'),('B1','K0'),('B2','B1'),('B3','B2')]:
    n,mn,mi,ma,pos,neg=delta(a,b)
    lines.append(f'| {a} - {b} | {n} | {mn:.8f} | {mi:.8f} | {ma:.8f} | {pos} / {neg} |')
lines += ['', '## B2 ROFA audit', '']
for arm in ['K0','B1','B2']:
    meta=npz_metadata(root/(arm+'.npz'))
    lines.append(f"- {arm}: dropped observations = `{meta.get('dropped_observation_count')}`, effective geometry = `{meta.get('effective_geometry')}`, bank SHA256 = `{hashlib.sha256((root/(arm+'.npz')).read_bytes()).hexdigest()}`.")
lines += ['- B3 directly loads `B2.npz`; its bank SHA256 is identical to B2.', '- Candidate and per-query records are in `B0/per_query.csv`, `K0/per_query.csv`, `B1/per_query.csv`, `B2/per_query.csv`, and `B3/per_query.csv`.', '', '## Interpretation', '', 'B1-K0 isolates debiased semantic plus geometry-supported grouping under the supplied constants. B2-B1 isolates the supplied mode-internal ROFA. B3-B2 isolates query-conditioned weighting. Any near-zero B2-B1 is reported as the measured net ROFA effect; no threshold was changed.']
(root/'B123_RESULTS.md').write_text('\n'.join(lines)+'\n')

