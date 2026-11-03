#!/usr/bin/env python3
import argparse, hashlib, json
from evidence_modes_rcs import BuildConfig, build_bank

p=argparse.ArgumentParser()
p.add_argument('--observations',required=True); p.add_argument('--nag',required=True); p.add_argument('--out',required=True)
p.add_argument('--b1',type=int,choices=[0,1],required=True); p.add_argument('--b2',type=int,choices=[0,1],required=True)
p.add_argument('--geo-weight',type=float,default=.25); p.add_argument('--max-modes',type=int,default=3)
p.add_argument('--min-mass',type=float,default=0.0); p.add_argument('--tau',type=float,default=3.)
p.add_argument('--rofa-mode',choices=['raw','center','off'],default='raw')
p.add_argument('--mode-selection',choices=['original','rcs'],default='rcs')
a=p.parse_args()
if a.b2 and not a.b1:p.error('B2 requires B1')
sha=hashlib.sha256(open(a.nag,'rb').read()).hexdigest()
cfg=BuildConfig(max_modes=a.max_modes,min_mass=a.min_mass,b1=bool(a.b1),b2=bool(a.b2),geo_weight=a.geo_weight,rofa_tau=a.tau,rofa_mode=a.rofa_mode,mode_selection=a.mode_selection)
print(json.dumps(build_bank(a.observations,a.out,sha,cfg),indent=2))
