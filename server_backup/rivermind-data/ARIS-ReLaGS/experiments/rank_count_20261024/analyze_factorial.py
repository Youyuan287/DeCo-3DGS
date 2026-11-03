"""Read-only source assets; independent weighted analysis in a fresh directory."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image

W=Path(__file__).resolve().parents[3]
OLD=W/'aris_relags/review-stage/spatial_support_20261024'
NEW=W/'aris_relags/review-stage/ovs_rank_count_20261024/full'
OUT=NEW.parent/'analysis'
MODES=['baseline','rank1_count0','rank0_count1','spatial']


def main():
    if OUT.exists():raise SystemExit('Refusing existing analysis output')
    old=list(csv.DictReader((OLD/'per_query.csv').open(encoding='utf-8-sig')))
    new=list(csv.DictReader((NEW/'per_query.csv').open(encoding='utf-8-sig')))
    rows=[r for r in old+new if r['mode'] in MODES]
    paired=defaultdict(dict)
    for r in rows:
        key=(r['image'],r['prompt']);assert r['mode'] not in paired[key]
        paired[key][r['mode']]=r
    assert len(paired)==56 and all(set(v)==set(MODES) for v in paired.values())
    images=Counter(im for im,q in paired)
    assert sorted(images.values())==[11,13,15,17]
    manifest=json.loads((NEW/'manifest.json').read_text())
    for f in ('query_diagnostics.json','per_query.csv'):
        expected=next(v for k,v in manifest['input_sha256'].items() if k.endswith('/'+f))
        assert hashlib.sha256((OLD/f).read_bytes()).hexdigest()==expected
    mask_count=0
    for key,rr in paired.items():
        assert len({r['pool_sha256'] for r in rr.values()})==1
        assert len({int(r['gt_px']) for r in rr.values()})==1
        for mode,r in rr.items():
            root=OLD if mode in ('baseline','spatial') else NEW
            safe=r['prompt'].replace(' ','_').replace('/','_')+'.png'
            mask=np.asarray(Image.open(root/mode/'masks'/r['image']/safe))>0
            assert int(mask.sum())==int(r['pred_px'])
            mask_count+=1
    summary=[];changes=[];factorial=[];contrib=defaultdict(lambda:defaultdict(float))
    def weight(key):return 1/len(images)/images[key[0]]
    for mode in MODES:
        miou=sum(weight(key)*float(rr[mode]['iou']) for key,rr in paired.items())
        summary.append(dict(mode=mode,miou_percent=miou*100,n_query_views=56,empty_masks=sum(int(rr[mode]['pred_px'])==0 for rr in paired.values())))
    base=summary[0]['miou_percent']
    for r in summary:r['delta_baseline_pp']=r['miou_percent']-base
    for key,rr in paired.items():
        vals={m:float(rr[m]['iou']) for m in MODES};w=weight(key)*100
        effects={'rank_effect_pp':w*(vals['rank1_count0']-vals['baseline']),
                 'count_effect_pp':w*(vals['rank0_count1']-vals['baseline']),
                 'interaction_pp':w*(vals['spatial']-vals['rank1_count0']-vals['rank0_count1']+vals['baseline']),
                 'total_spatial_pp':w*(vals['spatial']-vals['baseline'])}
        assert abs(effects['rank_effect_pp']+effects['count_effect_pp']+effects['interaction_pp']-effects['total_spatial_pp'])<1e-10
        factorial.append(dict(image=key[0],prompt=key[1],weight=w/100,**{m+'_iou':vals[m] for m in MODES},**effects))
        for m in ('rank1_count0','rank0_count1','spatial'):
            contrib[key[1]][m+'_delta_pp']+=w*(vals[m]-vals['baseline'])
        for effect,value in effects.items():contrib[key[1]][effect]+=value
    for mode in MODES[1:]:
        improved=regressed=unchanged=rescued=added_empty=union_changed=0
        nonempty_contribution=other_contribution=0.
        for key,rr in paired.items():
            b,r=rr['baseline'],rr[mode];d=float(r['iou'])-float(b['iou']);w=weight(key)*100
            improved+=d>1e-12;regressed+=d< -1e-12;unchanged+=abs(d)<=1e-12
            be,re=int(b['pred_px'])==0,int(r['pred_px'])==0
            rescued+=be and not re;added_empty+=re and not be
            union_changed+=b['gaussian_union_sha256']!=r['gaussian_union_sha256']
            if not be and not re:nonempty_contribution+=w*d
            else:other_contribution+=w*d
        changes.append(dict(mode=mode,improved=improved,regressed=regressed,unchanged=unchanged,
                            union_changed=union_changed,rescued_empty=rescued,newly_empty=added_empty,
                            both_nonempty_contribution_pp=nonempty_contribution,other_contribution_pp=other_contribution))
    totals={k:sum(r[k] for r in factorial) for k in ('rank_effect_pp','count_effect_pp','interaction_pp','total_spatial_pp')}
    queryrows=[dict(prompt=q,**values) for q,values in sorted(contrib.items())]
    status=json.loads((NEW/'FINAL_STATUS.json').read_text())
    for r in status['summary']:
        local=next(x for x in summary if x['mode']==r['mode'])
        assert abs(local['miou_percent']/100-r['miou'])<1e-12
    OUT.mkdir()
    for name,rs in [('summary.csv',summary),('paired_gains.csv',changes),('per_query_factorial.csv',factorial),('query_contributions.csv',queryrows)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(rs[0]));wr.writeheader();wr.writerows(rs)
    verification=dict(status='PASS',n_query_views=56,n_distinct_queries=len(contrib),images=dict(images),
                      mask_files_checked=mask_count,new_masks=112,prior_hashes_match=True,
                      pools_paired=True,factorial_additive_check=True,summary=summary,effect_totals=totals,
                      old_union_checks=status['old_union_checks'],identical_union_mask_checks=status['identical_union_mask_checks'],
                      analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      decision='STOP_CURRENT_SPATIAL_RULE_AND_GAP_RESCUE_NO_EXPANSION')
    (OUT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,ensure_ascii=False,indent=2))
    print('Changes:',changes)
    for r in queryrows:
        if r['prompt'] in ('red apple','pirate hat','jake','bag'):print(r)


if __name__=='__main__':main()
