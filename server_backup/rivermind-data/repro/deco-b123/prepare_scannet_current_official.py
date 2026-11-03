#!/usr/bin/env python3
import json, os, shutil, zipfile
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

BASE = Path('/root/rivermind-data/repro/deco-b123/scannet10_full_20261029')
REF = BASE / 'reference_root'
MODELROOT = Path('/root/rivermind-data/output/scannet')
ZIP = Path('/root/rivermind-data/VALA/dataset/scannet.zip')
SCENES = ['scene0000_00','scene0062_00','scene0070_00','scene0097_00','scene0140_00','scene0200_00','scene0347_00','scene0400_00','scene0590_00','scene0645_00']

def xyz_labels(path):
    v = PlyData.read(str(path))['vertex']
    xyz = np.column_stack([np.asarray(v[k]) for k in ('x','y','z')]).astype(np.float64)
    lab = np.asarray(v['label']).astype(np.int64)
    return xyz, lab

def main():
    (REF/'clip_cache').mkdir(parents=True, exist_ok=True)
    (REF/'output/scannet').mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        for scene in SCENES:
            gt_name = f'scannet/scans/{scene}/{scene}_vh_clean_2.labels.ply'
            if gt_name not in names:
                raise FileNotFoundError(gt_name)
            gt_path = BASE/'gt'/f'{scene}_vh_clean_2.labels.ply'
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            if not gt_path.exists():
                with z.open(gt_name) as src, gt_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            gx, gl = xyz_labels(gt_path)
            ply = MODELROOT/scene/'point_cloud/iteration_0/point_cloud.ply'
            px, _ = xyz_labels(ply) if 'label' in [p.name for p in PlyData.read(str(ply))['vertex'].properties] else (None, None)
            if px is None:
                v = PlyData.read(str(ply))['vertex']
                px = np.column_stack([np.asarray(v[k]) for k in ('x','y','z')]).astype(np.float64)
            dist, idx = cKDTree(gx).query(px, k=1, workers=-1)
            keep = np.ones(len(px), dtype=bool)
            np.savez_compressed(REF/'clip_cache'/f'{scene}.npz', gt=gl[idx], keep=keep)
            link = REF/'output/scannet'/scene
            if link.exists() or link.is_symlink():
                if link.is_symlink() or link.is_file(): link.unlink()
                elif link.is_dir(): shutil.rmtree(link)
            link.symlink_to(MODELROOT/scene, target_is_directory=True)
            manifest[scene] = dict(n_gaussians=int(len(px)), n_gt=int(len(gx)), nearest_mean=float(dist.mean()), nearest_p95=float(np.percentile(dist,95)), gt=str(gt_path), model=str(ply))
            print(scene, manifest[scene], flush=True)
    (BASE/'CURRENT_OFFICIAL_INPUTS.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
