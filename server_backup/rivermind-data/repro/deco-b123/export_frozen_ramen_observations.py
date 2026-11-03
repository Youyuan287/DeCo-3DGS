#!/usr/bin/env python3
"""Export ReLaGS production observations against a FROZEN level-3 NAG.

Run inside the ReLaGS repo (which must contain the audited observation-export
merge_proj.py and production_observation_adapter.py). Requires CUDA; no NAG
clustering, no training, no modification of pre-existing NAG or model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='LERF ramen source with language_features')
    p.add_argument('--model', required=True, help='Model directory with checkpoint and nag-l1.pt')
    p.add_argument('--nag', required=True, help='Existing FROZEN sai_nag_3.0.pt')
    p.add_argument('--output', required=True, help='NEW observations_level3.npz')
    p.add_argument('--iteration', type=int, default=0)
    p.add_argument('--seed', type=int, default=1234)
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for projection; NO outputs written')
    src, model, nag_path, out = map(Path, (a.source, a.model, a.nag, a.output))
    for f in [nag_path, model / 'nag-l1.pt']:
        if not f.is_file():
            raise FileNotFoundError(f)
    if not (src / 'language_features').exists():
        raise FileNotFoundError('source/language_features missing; use existing correct CLIP cache')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    nag_digest_before = sha256(nag_path)
    frozen = torch.load(nag_path, map_location='cpu', weights_only=False)
    labels = frozen['nag']
    if len(labels) < 4:
        raise ValueError(f'Expected level0..3 labels, got {len(labels)} levels')
    root_ids = labels[-1].detach().long().flatten()
    if root_ids.numel() == 0 or root_ids.min() < 0:
        raise ValueError('Invalid frozen root IDs')

    # Use exactly the SAI3D implementation from the audited observation-export
    # branch; do not call everything_seg()/assign_label() (they recluster NAG).
    import merge_proj
    from argparse import Namespace
    args = Namespace(
        source_path=str(src), model_path=str(model), iteration=a.iteration,
        images='images', resolution=-1, white_background=False,
        data_device='cuda', eval=False,
        sh_degree=3, vlm_type='clip',
        thres_connect=[0.9, 0.7, 0.7], thres_merge=20,
        max_neighbor_distance=2, view_freq=5,
        dis_decay=0.5, thres_dis=0.15, thres_trunc=0.0,
        seg_enhance=False, tau=3.0, seed=a.seed,
        mix='soft', agg='mean',
        observation_out=str(out),
        render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'],
    )
    agent = merge_proj.SAI3D(args)
    agent.init_data(args)
    if agent.N != root_ids.numel():
        raise ValueError(f'Frozen NAG has {root_ids.numel()} Gaussians, model has {agent.N}')

    # The original level-3 pass first builds pt_sp_label by ray tracing, then
    # projects 2D masks to FROZEN root regions. No partitioning or feature save.
    views = agent.scene.getTrainCameras()
    pt_sp_label = agent.extract_gaussian_features(agent.gaussians, views, 3)
    if tuple(pt_sp_label.shape) != (agent.N, len(views)):
        raise AssertionError(f'Unexpected pt_sp_label shape {pt_sp_label.shape}')
    try:
        agent.proj_gaussian_features(
            agent.gaussians, views, 3,
            root_ids.to(device='cuda'), pt_sp_label.T.to(device='cuda'))
        if not out.is_file():
            raise RuntimeError('No observation NPZ: ensure merge_proj has the audited observation-export hook')
        with np.load(out, allow_pickle=False) as d:
            required = ('features', 'weights', 'region_ids', 'view_ids',
                        'visible_gaussian_offsets', 'visible_gaussian_ids',
                        'member_gaussian_offsets', 'member_gaussian_ids')
            missing = [k for k in required if k not in d]
            if missing:
                raise RuntimeError(f'Incomplete observation export: {missing}')
            n = len(d['features'])
            assert d['features'].ndim == 2 and d['features'].shape[1] == 512
            assert all(len(d[k]) == n for k in ('weights','region_ids','view_ids'))
            assert len(d['visible_gaussian_offsets']) == n+1
            assert len(d['member_gaussian_offsets']) == n+1
            assert int(d['visible_gaussian_offsets'][-1]) == len(d['visible_gaussian_ids'])
            assert int(d['member_gaussian_offsets'][-1]) == len(d['member_gaussian_ids'])
            assert np.isfinite(d['features']).all() and np.isfinite(d['weights']).all()
            assert np.all(d['weights'] > 0)
            assert np.all(d['region_ids'] >= 0) and np.all(d['region_ids'] <= int(root_ids.max()))
            for k in ('member_gaussian_ids', 'visible_gaussian_ids'):
                assert len(d[k]) == 0 or (int(d[k].min()) >= 0 and int(d[k].max()) < agent.N)
            meta = dict(n_observations=n, n_observed_regions=len(np.unique(d['region_ids'])),
                        n_root_regions=int(root_ids.max())+1, n_views=len(views),
                        n_gaussians=agent.N, nag_sha256=nag_digest_before,
                        observation_sha256=sha256(out), source=str(src),model=str(model),
                        frozen_nag=str(nag_path), export_level=3, seed=a.seed)
        if sha256(nag_path) != nag_digest_before:
            raise AssertionError('Frozen NAG changed during export!')
        meta_path = out.with_suffix('.provenance.json')
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(json.dumps(meta, indent=2))
    except Exception:
        # Preserve failed output for forensic inspection, never masquerade it as valid.
        if out.exists():
            out.rename(out.with_suffix('.invalid.npz'))
        raise


if __name__ == '__main__':
    main()
