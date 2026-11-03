#!/usr/bin/env python3
import argparse, json, hashlib, html
from pathlib import Path
import numpy as np, pandas as pd

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--obs',required=True); ap.add_argument('--b1',required=True); ap.add_argument('--b2',required=True); ap.add_argument('--members1',required=True); ap.add_argument('--members2',required=True); ap.add_argument('--support',required=True); ap.add_argument('--recovery',required=True); ap.add_argument('--gt_root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); (out/'semantic_evidence_cases').mkdir(parents=True,exist_ok=True)
  with np.load(a.obs,allow_pickle=False) as z: weights=z['weights']; views=z['view_ids']; rids=z['region_ids']; ptr=z['visible_gaussian_offsets']; gids=z['visible_gaussian_ids']; feats=z['features']
  m1=json.loads(Path(a.members1).read_text())['memberships']; m2=json.loads(Path(a.members2).read_text())['memberships']; m2map={(int(x['region_id']),int(x['slot'])):set(map(int,x['observation_indices'])) for x in m2}; sup=pd.read_parquet(a.support); rec=pd.read_parquet(a.recovery)
  origin=[]; removed=[]
  for x in m1:
    rid=int(x['region_id']); slot=int(x['slot']); ids=list(map(int,x['observation_indices'])); vg=sorted(set(int(g) for i in ids for g in gids[ptr[i]:ptr[i+1]])); post=m2map.get((rid,slot),set());
    origin.append(dict(region_id=rid,mode=slot,n_observations=len(ids),view_ids=json.dumps(sorted(set(int(views[i]) for i in ids))),member_observation_ids=json.dumps(ids),visible_gaussian_count=len(vg),visible_gaussian_ids=json.dumps(vg[:20000]),prior=float(x.get('prior',0.0))))
    for i in sorted(set(ids)-post): removed.append(dict(region_id=rid,mode=slot,observation_id=int(i),view_id=int(views[i]),weight=float(weights[i]),feature_norm=float(np.linalg.norm(feats[i])),visible_gaussian_count=int(ptr[i+1]-ptr[i]),visible_gaussian_ids=json.dumps(gids[ptr[i]:ptr[i+1]].astype(int).tolist())))
  pd.DataFrame(origin).to_csv(out/'mode_origin_summary.csv',index=False); pd.DataFrame(removed).to_csv(out/'m2_removed_observations.csv',index=False)
  rr=rec.copy() if 'precision' in rec.columns else rec.merge(sup,on=['scene','image','query','region_id'],how='left'); rr['spread']=rr['max_slot_score']-rr['max_prior_slot_score']; rr['query_prefers_minority']=rr['query_prefers_minority'].fillna(False)
  picks=[]
  for label,sub in [('target_recovery',rr[(rr.target_related)&(~rr.no_evidence)].sort_values('recover_gain',ascending=False).head(5)),('non_target_recovery',rr[(~rr.target_related)&(~rr.no_evidence)].sort_values('recover_gain',ascending=False).head(5)),('mode_spread',rr[~rr.no_evidence].sort_values('spread',ascending=False).head(5)),('minority_preference',rr[(~rr.no_evidence)&(rr.query_prefers_minority)].sort_values('recover_gain',ascending=False).head(5))]:
    for _,r in sub.iterrows(): picks.append(dict(case_group=label,scene=r['scene'],image=r['image'],query=r['query'],region_id=int(r['region_id']),recover_gain=float(r['recover_gain']),precision=float(r['precision']),recall=float(r['recall']),iou=float(r['iou']),spread=float(r['spread']),no_evidence=bool(r['no_evidence'])))
  pd.DataFrame(picks).drop_duplicates(['scene','image','query','region_id','case_group']).to_csv(out/'semantic_evidence_case_index.csv',index=False)
  for j,r in enumerate(picks):
    meta=next((x for x in origin if x['region_id']==r['region_id']),{})
    body='<html><body><h2>Case %03d: %s</h2>'% (j,r['case_group'])
    body+='<pre>'+html.escape(json.dumps(r,indent=2,ensure_ascii=False))+'</pre><h3>Origin</h3><pre>'+html.escape(json.dumps(meta,indent=2,ensure_ascii=False))+'</pre>'
    body+='<p>GT and root-region support are linked by the scene/image paths in the metadata. Visible Gaussian IDs are recorded in the origin CSV; this page does not treat them as semantic truth.</p></body></html>'
    (out/'semantic_evidence_cases'/('case_%03d.html'%j)).write_text(body,encoding='utf8')
  print('origin rows',len(origin),'removed',len(removed),'cases',len(picks))
if __name__=='__main__': main()
