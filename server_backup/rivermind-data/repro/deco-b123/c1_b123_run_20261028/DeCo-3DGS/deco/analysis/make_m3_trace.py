import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--trace',required=True);ap.add_argument('--out',required=True);a=ap.parse_args(); rows=[]
for line in open(a.trace,encoding='utf8'):
 x=json.loads(line); s=np.asarray(x['slot_scores'],float); p=np.asarray(x['priors'],float); v=np.asarray(x['valid'],bool); s=s[v];p=p[v];
 if len(s)==0: continue
 log=s/0.08+np.log(np.maximum(p,1e-12)); log-=log.max(); w=np.exp(log);w/=w.sum();
 rows.append(dict(scene=x['scene'],image=x['image'],query=x['prompt'],region_id=x['region_id'],n_modes=len(s),fixed_score=x['fixed_score'],adaptive_score=x['adaptive_score'],adaptive_minus_fixed=x['adaptive_score']-x['fixed_score'],slot_scores=json.dumps(s.tolist()),priors=json.dumps(p.tolist()),adaptive_weights=json.dumps(w.tolist()),formula_max_error=float(abs(w-(np.exp(log)/np.exp(log).sum())).max())))
pd.DataFrame(rows).to_csv(a.out,index=False);print('rows',len(rows),'max_formula_error',max((r['formula_max_error'] for r in rows),default=0))
