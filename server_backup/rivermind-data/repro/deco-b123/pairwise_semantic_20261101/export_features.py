"""Export existing frozen NAG/text vectors; no RGB encoding or labels in export."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/rivermind-data/repro/deco-b123')
REPO = ROOT/'c1_b123_run_20261028/DeCo-3DGS'
sys.path[:0] = [str(REPO), str(REPO/'deco'), str(REPO/'deco/lerf_ovs')]
from deco_ops import center_torch
from utils.vlm_utils import ClipSimMeasure

SCENES = ('figurines', 'ramen', 'teatime', 'waldo_kitchen')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vlm = ClipSimMeasure()
    vlm.load_model()
    text_cache = {}
    metadata = {}
    with torch.no_grad():
        for scene in SCENES:
            blob = torch.load(f'/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/{scene}/sai_nag.pt',
                              map_location='cpu', weights_only=True)
            # Reproduce the existing reader's preprocessing; do not add a transform.
            features = center_torch([f.cuda().float() for f in blob['nag_feat']], renorm=False)
            records = {}
            path = ROOT/'score_readout_20261101/results'/f'{scene}_details.jsonl'
            for line in path.read_text().splitlines():
                detail = json.loads(line)
                prompt = detail['prompt']
                if prompt not in text_cache:
                    vlm.encode_text(prompt)
                    text_cache[prompt] = vlm.text_feature[0].detach().clone()
                for i, (level, cid) in enumerate(detail['candidates']):
                    key = (prompt, int(level), int(cid))
                    expected = float(detail['r'][i])
                    if key in records:
                        assert abs(records[key]['r']-expected) < 1e-7
                        continue
                    f = features[level-1][cid]
                    t = text_cache[prompt]
                    actual = float(f @ t)
                    if not np.isfinite(actual) or abs(actual-expected) > 1e-6:
                        raise RuntimeError(f'FROZEN_VECTOR_MISMATCH {scene} {key}: {actual} {expected}')
                    records[key] = dict(F=f.cpu().numpy(), T=t.cpu().numpy(),
                                        r=expected, error=abs(actual-expected))
            keys = sorted(records)
            F = np.stack([records[k]['F'] for k in keys]).astype(np.float32)
            T = np.stack([records[k]['T'] for k in keys]).astype(np.float32)
            assert np.isfinite(F).all() and np.isfinite(T).all()
            np.savez_compressed(out/f'{scene}.npz', F=F, T=T,
                prompts=np.asarray([k[0] for k in keys]),
                levels=np.asarray([k[1] for k in keys], dtype=np.int16),
                candidate_ids=np.asarray([k[2] for k in keys], dtype=np.int32))
            metadata[scene] = dict(candidate_records=len(keys), prompts=len({k[0] for k in keys}),
                dim=F.shape[1], max_dot_error=max(r['error'] for r in records.values()),
                feature_norm_min=float(np.linalg.norm(F,axis=1).min()),
                feature_norm_max=float(np.linalg.norm(F,axis=1).max()))
            print('EXPORTED', scene, metadata[scene], flush=True)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('ALL_EXPORTED', sum(v['candidate_records'] for v in metadata.values()),flush=True)

if __name__ == '__main__':
    main()
