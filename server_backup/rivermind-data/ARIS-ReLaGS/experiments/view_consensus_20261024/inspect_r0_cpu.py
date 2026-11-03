"""Read-only model and bounded related-cache inspection. Output path is new-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import torch

BASE=Path('/root/rivermind-data')
MODEL=BASE/'ReLaGS_Reproduction/runs/readout_20261024/figurines/model_r0/sai_nag_3.0.pt'
ROOTS=[
 BASE/'ReLaGS_Reproduction/runs/readout_20261024/figurines',
 BASE/'ReLaGS_Reproduction/ReLaGS/runs/readout_20261024/figurines',
 BASE/'ReLaGS_Reproduction/ref/scene/figurines',
 BASE/'ReLaGS_Reproduction/ref/relags_hf/figurines',
 BASE/'VALA/dataset/3dgs/lerf_ovs/figurines',
]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def tensor_meta(t):
    t=t.detach().cpu().contiguous()
    return {'shape':list(t.shape),'dtype':str(t.dtype),
            'raw_tensor_sha256':hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            'finite':bool(torch.isfinite(t).all()),
            'min':t.min().item() if t.numel() else None,'max':t.max().item() if t.numel() else None}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before=digest(MODEL)
    obj=torch.load(MODEL,map_location='cpu')
    fields={}
    for key,value in obj.items():
        fields[key]=[tensor_meta(t) for t in value] if isinstance(value,(tuple,list)) else tensor_meta(value)
    inventories=[]
    keywords=('observ','pt_sp','sp_feat','visibility','rofa','projection','mapping','contrib','sidecar')
    for root in ROOTS:
        listing=[]
        for directory,subdirs,files in os.walk(root,followlinks=False):
            for name in sorted(files):
                path=Path(directory)/name
                if path.is_symlink():
                    listing.append({'path':str(path),'symlink':os.readlink(path)})
                elif path.suffix in ('.pt','.pth','.npz','.pkl') or any(k in name.lower() for k in keywords):
                    listing.append({'path':str(path),'bytes':path.stat().st_size})
        inventories.append({'root':str(root),'exists':root.exists(),'matches':listing})
    result={'python':platform.python_version(),'torch':torch.__version__,
      'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'default_dtype':str(torch.get_default_dtype()),
      'model':str(MODEL),'model_bytes':MODEL.stat().st_size,'sha256_before':before,'sha256_after':digest(MODEL),
      'keys':list(obj),'fields':fields,'bounded_inventory':inventories,
      'source_sha256':digest(BASE/'ReLaGS_Reproduction/ReLaGS/merge_proj.py'),
      'scope':'CPU only; nonrecursive symlink traversal; matched filenames only; absence not whole-host proof'}
    with args.output.open('x',encoding='utf-8') as file: json.dump(result,file,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
