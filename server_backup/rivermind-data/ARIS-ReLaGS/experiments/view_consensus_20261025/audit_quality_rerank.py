"""CPU-only candidate-A audit; fails closed without a frozen candidate pool."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED_POOL = {'scene','query','level','candidate_id','raw_score'}


def _sha_bytes(x):
    import numpy as np
    return hashlib.sha256(np.asarray(x, dtype=np.uint8).tobytes()).hexdigest()


def validate_pool(path):
    p = Path(path)
    if not p.exists(): raise RuntimeError('BLOCKED_QUALITY_A: frozen candidate pool is absent')
    with p.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    missing = sorted(REQUIRED_POOL - set(rows[0] if rows else []))
    if missing: raise RuntimeError('BLOCKED_QUALITY_A: pool missing fields ' + ','.join(missing))
    if any('gt' in k.lower() or 'iou' in k.lower() for k in (rows[0] if rows else {})): raise RuntimeError('BLOCKED_QUALITY_A: GT/IoU field in inference pool')
    return rows


def gaussian_union_hash(labels, selected):
    mask = np.zeros(len(labels[0]), dtype=np.uint8)
    for level, cid in selected:
        mask |= (np.asarray(labels[int(level)]) == int(cid)).astype(np.uint8)
    return _sha_bytes(mask), int(mask.sum())


def audit(pool_csv, model_path, output):
    try:
        rows = validate_pool(pool_csv)
        if not Path(model_path).exists(): raise RuntimeError('BLOCKED_QUALITY_A: frozen R0 sai_nag.pt is absent')
        import torch
        from quality_aware_scoring import quality_from_hierarchy
        obj = torch.load(model_path, map_location='cpu')
        labels = [x.detach().cpu().numpy() for x in obj['nag']]
        feats = [x.detach().cpu().numpy() for x in obj['nag_feat']]
        quality = {level: quality_from_hierarchy(feats, labels, level) for level in (2, 3)}
        report_rows=[]
        for query in sorted({r['query'] for r in rows}):
            pool=[r for r in rows if r['query']==query]
            scored=[]
            for r in pool:
                level=int(r['level']); cid=int(r['candidate_id']); raw=float(r['raw_score'])
                if level not in quality or cid >= len(quality[level]): raise RuntimeError('BLOCKED_QUALITY_A: candidate outside frozen hierarchy')
                scored.append((raw*float(quality[level][cid]), level, cid, raw))
            original=sorted(pool, key=lambda r:(-float(r['raw_score']), int(r['level']), int(r['candidate_id'])))[:3]
            new=sorted(scored, key=lambda x:(-x[0], x[1], x[2]))[:3]
            orig_pairs=[(int(r['level']),int(r['candidate_id'])) for r in original]; new_pairs=[(x[1],x[2]) for x in new]
            oh, on = gaussian_union_hash(labels, orig_pairs); nh, nn = gaussian_union_hash(labels, new_pairs)
            report_rows.append({'query':query,'original':orig_pairs,'quality':new_pairs,'original_top1_kept':orig_pairs[0]==new_pairs[0],'original_union_sha256':oh,'quality_union_sha256':nh,'original_gaussians':on,'quality_gaussians':nn,'whole_part_metadata':'unavailable; no legality claim'})
        Path(output).write_text(json.dumps({'status':'CPU_AUDIT_COMPLETE','rows':report_rows,'gt_used':False,'gpu_used':False}, indent=2))
    except RuntimeError as exc:
        Path(output).write_text(json.dumps({'status':'BLOCKED','reason':str(exc),'gt_used':False,'gpu_used':False}, indent=2))
        return {'status':'BLOCKED','reason':str(exc)}
    return {'status':'CPU_AUDIT_COMPLETE','rows':report_rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--pool-csv',required=True); p.add_argument('--model',required=True); p.add_argument('--output',required=True); a=p.parse_args(); audit(a.pool_csv,a.model,a.output)


if __name__=='__main__': main()
