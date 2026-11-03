#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--support',required=True); ap.add_argument('--trace',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    sup=pd.read_parquet(a.support); rec=[]
    for line in open(a.trace,encoding='utf8'):
        x=json.loads(line); active=np.asarray(x['valid'],bool); scores=np.asarray(x['slot_scores'],float)[active]; pri=np.asarray(x['priors'],float)[active]
        if len(scores):
            mi=int(np.argmax(scores)); pi=int(np.argmax(pri)); mx=float(scores.max()); maxprior=float(scores[pi]); fixed=float(x['fixed_score']); single=float(x['root_original'])
            rec.append(dict(scene=x['scene'],image=x['image'],query=x['prompt'],region_id=int(x['region_id']),single_score=single,max_slot_score=mx,max_prior_slot_score=maxprior,fixed_score=fixed,recover_gain=mx-single,n_modes=len(scores),prior_argmax=pi,score_argmax=mi,query_prefers_minority=bool(mi!=pi)))
    ev=pd.DataFrame(rec); keys=['scene','image','query','region_id']; df=sup.merge(ev,on=keys,how='left'); df['no_evidence']=df['single_score'].isna(); df['target_related']=df['intersection_px']>0; df.to_parquet(out/'evidence_recovery.parquet',index=False)
    df['recover_gain']=df['recover_gain'].astype(float); rows=[]
    for target,sub in df.groupby('target_related',dropna=False):
      have=sub[~sub['no_evidence']]; rows.append(dict(target_related=bool(target),rows=len(sub),with_evidence=len(have),mean_recover_gain=float(have['recover_gain'].mean()) if len(have) else np.nan,median_recover_gain=float(have['recover_gain'].median()) if len(have) else np.nan,mean_max_slot=float(have['max_slot_score'].mean()) if len(have) else np.nan,mean_single=float(have['single_score'].mean()) if len(have) else np.nan,mean_precision=float(sub['precision'].mean()),mean_recall=float(sub['recall'].mean()),mean_iou=float(sub['iou'].mean())))
    pd.DataFrame(rows).to_csv(out/'evidence_recovery_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
