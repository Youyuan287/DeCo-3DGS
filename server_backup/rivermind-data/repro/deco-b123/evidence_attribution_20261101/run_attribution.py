"""Exact per-view matching-feasibility control for single-query candidate evidence."""
from pathlib import Path
import time
helper=Path(__file__).with_name('spatial_base.py').read_text()
exec(compile(helper.split('family={}')[0],'spatial_base.py','exec'),globals())
from scipy.optimize import linear_sum_assignment

def matching(a):
    nr,nm=a.shape
    padded=np.concatenate([a,np.zeros((nr,nr),float)],axis=1)
    if nr==0:return 0.,{}
    rr,cc=linear_sum_assignment(-padded)
    real={int(r):int(c) for r,c in zip(rr,cc) if c<nm and a[r,c]>0}
    return float(sum(a[r,c] for r,c in real.items())),real

def candidate_support(c):
    level,_,cid=c[:3]
    lab=snag.labels[level].detach().cpu().numpy()
    return lab==int(cid)

def new_scores(p5,t,legacy):
    tic=time.perf_counter()
    roots=sorted({int(c[1]) for c in p5}); root_index={r:i for i,r in enumerate(roots)}
    root_lab=snag.labels[root_level].detach().cpu().numpy()
    gs=[candidate_support(c) for c in p5]
    vals=[[[],[],[]] for _ in p5]
    feasible_masks=competing_masks=0;retention=[];records=[]
    for vi,v in enumerate(cache):
        ids=v['ids'];mp=v['maps'][2]
        node_ids=np.array([j for j,n in enumerate(v['nodes']) if n[0]==3 and v['totals'][j]>0],int)
        if not len(node_ids):continue
        geom=np.zeros((len(roots),len(node_ids)),float)
        for ri,root in enumerate(roots):
            gb=root_lab[ids]==root; n=int(gb.sum())
            pp=mp[gb];inter=np.bincount(pp[pp>=0],minlength=len(v['totals']))[node_ids]
            iou_r=inter/np.maximum(n+v['totals'][node_ids]-inter,1)
            ok=(inter>=20)&(iou_r>=.1);geom[ri,ok]=iou_r[ok]
        best,binding=matching(geom)
        competitors=(geom>0).sum(0);feasible_masks+=int((competitors>0).sum());competing_masks+=int((competitors>1).sum())
        possible=np.zeros_like(geom,dtype=bool);forced_scores=np.zeros_like(geom)
        for ri in range(len(roots)):
            for mi in np.flatnonzero(geom[ri]>0):
                remainder=geom[np.arange(len(roots))!=ri][:,np.arange(len(node_ids))!=mi]
                score=geom[ri,mi]+matching(remainder)[0]
                forced_scores[ri,mi]=score
                possible[ri,mi]=score+1e-10>=.9*best
        for ci,c in enumerate(p5):
            ri=root_index[int(c[1])];gb=gs[ci][ids];n=int(gb.sum());pp=mp[gb]
            inter=np.bincount(pp[pp>=0],minlength=len(v['totals']))[node_ids]
            eligible=(inter>=20)&(inter>=.1*n)&(inter>=.1*v['totals'][node_ids])&(geom[ri]>0)
            mids=np.flatnonzero(eligible)
            if not len(mids):continue
            sims=v['features'][node_ids]@t
            if ri in binding and eligible[binding[ri]]:
                mi=binding[ri];vals[ci][0].append((float(sims[mi]),vi,int(node_ids[mi])))
            mi=mids[np.argmax(sims[mids])];vals[ci][1].append((float(sims[mi]),vi,int(node_ids[mi])))
            mids_j=np.flatnonzero(eligible&possible[ri])
            if len(mids_j):
                mi=mids_j[np.argmax(sims[mids_j])];vals[ci][2].append((float(sims[mi]),vi,int(node_ids[mi])))
                retention.append(float(forced_scores[ri,mi]/best) if best>0 else 1.)
    outputs=np.tile(legacy,(3,1));complete=0;changed=0
    for ci,c in enumerate(p5):
        tops=[sorted(vs,key=lambda a:(-a[0],a[1],a[2]))[:3] for vs in vals[ci]]
        common=all(len(x)>=3 for x in tops)
        if common:
            complete+=1
            for method in range(3):outputs[method,ci]=np.mean([x[0] for x in tops[method]])
            changed+=len({x[1:] for x in tops[1]} ^ {x[1:] for x in tops[2]})
        records.append(dict(candidate=ci,root=int(c[1]),level=int(c[0]),cand_id=int(c[2]),counts=[len(x) for x in vals[ci]],common_valid=common,top=tops))
    assert np.all(outputs[0]<=outputs[2]+1e-7),'geometry optimum must be feasible for J'
    assert np.all(outputs[2]<=outputs[1]+1e-7),'J can only restrict independent evidence choices'
    return outputs,dict(n_roots=len(roots),competition_fraction=competing_masks/max(feasible_masks,1),changed_tokens=changed,valid_candidates=complete,joint_fallbacks=len(p5)-complete,solver_seconds=time.perf_counter()-tic,geom_retention_min=min(retention,default=float('nan'))),records

def select_sets(p5,prompt,image,signal):
    n=len(p5);sc=np.array([c[3] for c in p5],float);g=gap_len(sc);S0=list(range(g))
    pv=np.array([preds[(image,prompt,i+1)] for i in range(n)])
    vv=[ev[(image,prompt,i+1)] for i in range(n)]
    area=np.array([v[0] for v in vv]);vis=np.array([v[1] for v in vv])
    mp=max(pv);prune=[i for i in S0 if i==0 or pv[i]>=.5*mp] or ([0] if S0 else [])
    scset=prune if mp>=.5 else S0
    champ=int(np.argmax(zs(area)+zs(vis)))
    if area[0]<.001 and area.max()>2*area[0] and champ!=0:scset=sorted(set(scset)|{champ})
    sm=list(np.argsort(-(zs(sc)+zs(signal)),kind='stable'))[:g]
    ec=max((signal[i] for i in scset),default=-1e9);em=max((signal[i] for i in sm),default=-1e9)
    return sm if set(sm)!=set(S0) and em>ec else scset

def render_set(p5,chosen,cam):
    rel=snag._get_related_gaussians([p5[i] for i in chosen]) if chosen else torch.zeros(N,1)
    return render_bool(rel.detach().cpu().ravel()>0,cam,cam.image_height,cam.image_width) if chosen else np.zeros((cam.image_height,cam.image_width),bool)

family={}
with (ROOT/'lerf_segidentity_20261030'/f'{args.scene}_segidentity.csv').open() as f:
    for r in csv.DictReader(f):family.setdefault((r['image'],r['prompt']),[]).append(r)
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
(out/f'{args.scene}_sources.json').write_text(json.dumps([v['name'] for v in cache]))
fields=['scene','image','prompt','family']+[f'{t}_iou' for t in 'AGIJ']+[f'{t}_single_iou' for t in 'AGIJ']+['baseline_error','n_roots','competition_fraction','changed_tokens','valid_candidates','joint_fallbacks','solver_seconds','geom_retention_min']
memo={}
with (out/f'{args.scene}_attribution.csv').open('w',newline='') as f,(out/f'{args.scene}_evidence.jsonl').open('w') as detail,torch.no_grad():
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for qi,r in enumerate(references):
        if args.limit_queries and qi>=args.limit_queries:break
        image,prompt=r['image'],r['prompt'];cam=cams[image]
        sim=sims_for(prompt);p5=pool(sim,5);n=len(p5)
        legacy=np.array([msig.get((image,prompt,i+1),np.nan) for i in range(n)],float)
        legacy=np.where(np.isfinite(legacy),legacy,np.nanmean(legacy) if np.isfinite(legacy).any() else 0.)
        if prompt not in memo:
            scores,stats,rec=new_scores(p5,text_vector(prompt),legacy)
            memo[prompt]=(scores,stats,rec,[(int(c[0]),int(c[1]),int(c[2])) for c in p5])
            detail.write(json.dumps(dict(prompt=prompt,records=rec,stats=stats),default=int)+'\n');detail.flush()
            print('SCORED',args.scene,prompt,stats,flush=True)
        scores,stats,rec,identities=memo[prompt]
        assert identities==[(int(c[0]),int(c[1]),int(c[2])) for c in p5]
        signals=np.vstack([legacy,scores]);selections=[select_sets(p5,prompt,image,x) for x in signals]
        # GT only enters evaluation after all four selections are computed.
        anno=json.load(open(Path(args.gt_root)/args.scene/(image+'.json')))
        gt=np.zeros((cam.image_height,cam.image_width),np.uint8)
        for obj in anno['objects']:
            if obj['category']==prompt:gt=np.maximum(gt,polygon_to_mask(gt.shape,obj['segmentation']))
        gt=gt.astype(bool)
        values=[iou(render_set(p5,s,cam),gt) for s in selections]
        single=[iou(render_set(p5,[int(np.argmax(signal))],cam),gt) for signal in signals]
        err=abs(values[0]-float(r['gatee_iou']))
        if err>1e-6:raise RuntimeError(f'BASELINE_MISMATCH {image} {prompt} {values[0]} {r["gatee_iou"]}')
        candidates=sorted(family[(image,prompt)],key=lambda x:int(x['rank']));best=max(candidates,key=lambda x:float(x['iou']))
        fam='recall' if float(best['iou'])<.3 else 'found' if int(best['rank'])==1 else 'rank2' if int(best['rank'])==2 else 'deep'
        row=dict(scene=args.scene,image=image,prompt=prompt,family=fam,baseline_error=err,**stats)
        row.update({tag+'_iou':x for tag,x in zip('AGIJ',values)});row.update({tag+'_single_iou':x for tag,x in zip('AGIJ',single)})
        writer.writerow(row);f.flush()
        print('QUERY',args.scene,qi+1,prompt,fam,dict(zip('AGIJ',values)),flush=True)
print('SCENE_DONE',args.scene,len(references),flush=True)
