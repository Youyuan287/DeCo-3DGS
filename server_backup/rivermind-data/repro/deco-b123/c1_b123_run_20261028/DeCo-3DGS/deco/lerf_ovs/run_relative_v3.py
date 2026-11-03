"""Remote fixed launcher; never builds observations, banks or features."""
import os,subprocess,sys,json,hashlib
from pathlib import Path
ROOT=Path('/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028')
CODE=ROOT/'DeCo-3DGS'
OLD=ROOT/'evidence_v2_four_20261028'
OUT=ROOT/'evidence_v3_relative_20261029'
PY='/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3'
REL='/root/rivermind-data/repro/deco-b123/frozen_ramen_relags'
os.environ.update(RELAGS_REPO=REL,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='0',
 PYTHONPATH=f'{CODE}:{REL}:{REL}/ext:{REL}/ext/spt/dependencies/grid_graph/python/bin')
os.chdir(REL)
OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 if scene=='ramen':
  b=Path('/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028')
  obs=Path('/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz')
 elif scene=='waldo_kitchen':
  b=Path('/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval')
  obs=b.parent/'observations_level3.npz'
 else:
  b=OLD/'generated_inputs'/scene;obs=b/'observations_level3.npz'
 model=Path('/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf')/scene
 nag=model/'sai_nag.pt'
 frozen={str(p):sha(p) for p in [nag,b/'B1.npz',obs,b/'B1.members.json']}
 # Rerender V2 only for spatial and candidate audit, plus single V3 arm.
 for arm in ['semantic_audit','reliable_audit','relative_v3']:
  out=OUT/scene/arm;out.mkdir(parents=True,exist_ok=True)
  cmd=[PY,str(CODE/'deco/lerf_ovs/eval_lerf_ovs_v3.py'),
   '--scene',scene,'--source',f'/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/{scene}',
   '--model',str(model),'--nag',str(nag),'--gt_root','/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label',
   '--out',str(out),'--csv',str(out/'per_query.csv'),'--tag',arm,
   '--topk','5','--level_until','3','--filter_small','50','--center','image',
   '--text','canon','--renorm','0','--sel','gap','--evidence-v2',
   'semantic' if arm=='semantic_audit' else 'reliable',
   '--evidence-bank',str(b/'B1.npz'),'--evidence-observations',str(obs),
   '--evidence-members',str(b/'B1.members.json'),'--evidence-temperature','0.08']
  if arm=='relative_v3':cmd+=['--relative-v3']
  (out/'manifest.json').write_text(json.dumps(dict(inputs=frozen,command=cmd),indent=2))
  with (out/'run.log').open('w') as log:
   result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'{scene}/{arm} failed; see {out}/run.log')
  assert all(sha(p)==h for p,h in frozen.items())
  print('DONE',scene,arm,flush=True)
(OUT/'COMPLETE').write_text('all fixed arms completed')
